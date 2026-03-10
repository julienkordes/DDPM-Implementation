import os
import copy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import get_model
from utils import DDPMScheduler, get_dataloader, sample_and_save, update_ema


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Données
    dataloader = get_dataloader(args)

    model = get_model(args.model, 
            args.in_channels, 
            args.channels, 
            args.network_depth,
            args.num_res_block, 
            args.attention_resolution, 
            args.image_size, 
            args.time_emb_dim
            ).to(device)
    
    start_epoch = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(args.checkpoint_path.split("_")[-1].replace(".pt", "")) 
    
    ema_model = copy.deepcopy(model)       # copie exacte du modèle
    ema_model.requires_grad_(False)        # jamais entraîné directement
    ema_decay = 0.9999

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {num_params / 1e6:.1f}M")

    # Scheduler de diffusion
    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
    )

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Si on a loadé un checkpoint on fait avancer le scheduler manuellement jusqu'à la bonne valeur
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_scheduler.step()

    losses = []
    global_step = 0

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")

        for batch_idx, (x, class_label) in enumerate(pbar):
            x, class_label = x.to(device), class_label.to(device)

            # Dropout du conditioning : remplace le label par le null token
            drop_mask = torch.rand(x.shape[0], device=device) < 0.1  
            class_label[drop_mask] = model.num_classes  # null token = index 10

            t = torch.randint(0, args.num_timesteps, (x.shape[0],), device=device)

            optimizer.zero_grad()
            loss = scheduler.compute_loss(model, x, t, class_label=class_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, ema_decay, global_step) 

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} — Loss moyenne : {avg_loss:.4f} — LR : {lr_scheduler.get_last_lr()[0]:.6f}")

        # Génération de samples
        if epoch % args.sample_every == 0:
            sample_and_save(ema_model, scheduler, args, epoch)

        # Sauvegarde checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ddpm_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt_path)
            print(f"  Checkpoint sauvegardé : {ckpt_path}")

    # Courbe de loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Courbe d'entraînement DDPM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"))
    plt.close()
    print("Entraînement terminé.")

