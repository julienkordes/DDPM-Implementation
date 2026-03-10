import math
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from PIL import Image, ImageDraw


class DDPMScheduler:

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Schedule '{schedule}' inconnu. Utiliser 'linear' ou 'cosine'.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, arr, t, x_shape):
        out = arr.to(t.device).gather(0, t)
        for _ in range(len(x_shape) - 1):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def predict_x_start_from_noise(self, x_t, t, noise):
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    @torch.no_grad()
    def p_sample_cfg(self, model, x_t, t, class_label, guidance_scale=3.0):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_cond = model(x_t, t_tensor, class_label=class_label)
        null_label = torch.full_like(class_label, model.num_classes)
        eps_uncond = model(x_t, t_tensor, class_label=null_label)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x_start = self.predict_x_start_from_noise(x_t, t_tensor, eps)
        x_start = torch.clamp(x_start, -1, 1)
        mean, _, log_variance = self.q_posterior_mean_variance(x_start, x_t, t_tensor)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return mean + (0.5 * log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, guidance_scale=3.0, class_label=None):
        x = torch.randn(shape, device=device)
        if class_label is None:
            class_label = torch.randint(0, model.num_classes, (shape[0],), device=device)
        else:
            class_label = torch.tensor([class_label] * shape[0], device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample_cfg(model, x, t, class_label, guidance_scale)
        return x, class_label

    def compute_loss(self, model, x_start, t, class_label=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, class_label=class_label)
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, device, num_steps=50, eta=0.0, guidance_scale=3.0, class_label=None):
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps + 1, dtype=torch.long)
        timesteps = list(reversed(timesteps.tolist()))  # de T à 0

        if class_label is None:
            class_label = torch.randint(0, model.num_classes, (shape[0],), device=device)
        elif isinstance(class_label, int):
            class_label = torch.tensor([class_label] * shape[0], device=device)

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t      = int(timesteps[i])
            t_prev = int(timesteps[i + 1])

            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            eps_cond  = model(x, t_tensor, class_label=class_label)
            null_label = torch.full_like(class_label, model.num_classes)
            eps_uncond = model(x, t_tensor, class_label=null_label)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_bar_t      = self.alphas_cumprod[t]
            alpha_bar_t_prev = self.alphas_cumprod[t_prev]

            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            dir_xt = (1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)).sqrt() * eps

            # Bruit optionnel (eta=0 -> déterministe)
            noise = eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)).sqrt() * torch.randn_like(x)

            x = alpha_bar_t_prev.sqrt() * x0_pred + dir_xt + noise

        return x, class_label


def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "celeba":
        dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {args.dataset} non supporté.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

@torch.no_grad()
def sample_and_save(model, scheduler, args, epoch):
    """ Génère et sauvegarde 16 images par le DDPM """
    model.eval()
    shape = (args.num_samples, args.in_channels, args.image_size, args.image_size)
    
    samples, class_labels = scheduler.p_sample_loop(model, shape, device=args.device, 
                                       guidance_scale=args.guidance_scale)
    samples = (samples.clamp(-1, 1) + 1) / 2  # -> [0, 1]

    # Annoter chaque image avec son label
    LABEL_HEIGHT = 12  # pixels de hauteur pour le texte
    annotated = []
    for img, label_idx in zip(samples, class_labels):
        pil_img = TF.to_pil_image(img)
        annotated_img = Image.new("RGB", (pil_img.width, pil_img.height + LABEL_HEIGHT), (255, 255, 255))
        annotated_img.paste(pil_img, (0, 0))
        draw = ImageDraw.Draw(annotated_img)
        label_name = CIFAR10_CLASSES[label_idx.item()]
        text_x = pil_img.width // 2 - len(label_name) * 3
        draw.text((text_x, pil_img.height + 1), label_name, fill=(0, 0, 0))
        annotated.append(TF.to_tensor(annotated_img))

    annotated = torch.stack(annotated)
    grid = make_grid(annotated, nrow=4)
    path = os.path.join(args.output_dir, f"samples_epoch_{epoch:04d}.png")
    save_image(grid, path)
    print(f"  Samples sauvegardés : {path}")
    model.train()

def update_ema(ema_model, model, decay, step):
    # Les premiers steps, on copie directement sans lissage
    actual_decay = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(actual_decay).add_(param.data, alpha=1 - actual_decay)


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        # tensors : (N, 3, 32, 32) dans [0, 255] uint8
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, i):
        return self.tensors[i]
    

def show_metrics(model, scheduler, args, sampling_method="DDPM"):
    """ Calcule la FID et l'IS """
    all_samples = []
    for _ in tqdm(range(0, args.num_samples_FID, args.batch_size)):
        if sampling_method == "DDPM":
            samples, _ = scheduler.p_sample_loop(model, (args.batch_size, 3, 32, 32), args.device,
                                                  guidance_scale=args.guidance_scale)
        elif sampling_method == "DDIM":
            samples, _ = scheduler.ddim_sample_loop(model, (args.batch_size, 3, 32, 32), args.device,
                                                     num_steps=args.num_steps_DDIM,
                                                     eta=args.eta, guidance_scale=args.guidance_scale)
        samples = (samples.clamp(-1, 1) + 1) / 2
        samples = (samples * 255).byte().cpu()
        all_samples.append(samples)

    generated = torch.cat(all_samples, dim=0)
    dataset = GeneratedDataset(generated)
    metrics = calculate_metrics(input1=dataset, input2="cifar10-train", fid=True, isc=True)

    fid     = metrics['frechet_inception_distance']
    isc     = metrics['inception_score_mean']
    isc_std = metrics['inception_score_std']

    if sampling_method == "DDIM":
        steps_info = f"{args.num_steps_DDIM} steps (eta={args.eta})"
    else:
        steps_info = f"{args.num_timesteps} steps"

    col_width = 30
    sep = "+" + "-" * col_width + "+" + "-" * 20 + "+\n"

    def row(label, value):
        return f"| {label:<{col_width-2}} | {value:<18} |\n"

    table = "\n"
    table += sep
    table += row("Metric", "Value")
    table += sep
    table += row("Sampling method", sampling_method)
    table += row("Steps", steps_info)
    table += row("Guidance scale", f"{args.guidance_scale:.1f}")
    table += row("Num samples", str(args.num_samples_FID))
    table += sep
    table += row("FID ↓ (lower is better)", f"{fid:.2f}")
    table += row("IS  ↑ (higher is better)", f"{isc:.2f} ± {isc_std:.2f}")
    table += sep

    # Sauvegarde
    filename = f"metrics_{sampling_method}"
    if sampling_method == "DDIM":
        filename += f"_{args.num_steps_DDIM}steps"
    filename += ".txt"

    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"Métriques sauvegardées : {output_path}")

@torch.no_grad()
def visualize_denoising(model, scheduler, args, num_snapshots=10, class_label=None):
    """
    Génère une figure montrant le débruitage progressif d'une image.
    num_snapshots : nombre d'étapes intermédiaires à afficher
    """
    model.eval()
    device = args.device

    if class_label is None:
        class_label = torch.randint(0, model.num_classes, (1,), device=device)
    else:
        class_label = torch.tensor([class_label], device=device)

    snapshot_times = set(
        int(t) for t in torch.linspace(scheduler.num_timesteps - 1, 0, num_snapshots)
    )

    x = torch.randn(1, args.in_channels, args.image_size, args.image_size, device=device)
    snapshots = []
    snapshot_labels = []

    for t in reversed(range(scheduler.num_timesteps)):
        x = scheduler.p_sample_cfg(model, x, t, class_label, args.guidance_scale)
        if t in snapshot_times:
            img = (x.clamp(-1, 1) + 1) / 2  # -> [0, 1]
            snapshots.append(img.squeeze(0).cpu())
            snapshot_labels.append(f"t={t}")

    if 0 not in snapshot_times:
        img = (x.clamp(-1, 1) + 1) / 2
        snapshots.append(img.squeeze(0).cpu())
        snapshot_labels.append("t=0")

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.5))
    class_name = CIFAR10_CLASSES[class_label.item()]

    for ax, img, label in zip(axes, snapshots, snapshot_labels):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Denoising process — class: {class_name}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output_dir, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {args.output_dir}")
    model.train()