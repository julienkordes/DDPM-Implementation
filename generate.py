import torch
from config.argparser import load_opts
from utils import DDPMScheduler, show_metrics, visualize_denoising, sample_grid
from models import get_model


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = get_model(args.model, 
            args.in_channels, 
            args.channels, 
            args.network_depth,
            args.num_res_block, 
            args.attention_resolution, 
            args.image_size, 
            args.time_emb_dim
            ).to(device)

    model.load_state_dict(ckpt["ema_model_state_dict"]) # On fait la génération avec le modèle EMA
    model.eval()
    print(f"Modèle chargé — Epoch {ckpt['epoch']} — Loss {ckpt['loss']:.4f}")
    return model


@torch.no_grad()
def generate(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint_path, device)

    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
    )

    sample_grid(model, scheduler, args)
    show_metrics(model, scheduler, args, sampling_method = args.sampling_method)
    visualize_denoising(model, scheduler, args)

if __name__ == "__main__":
    args = load_opts()
    generate(args)
