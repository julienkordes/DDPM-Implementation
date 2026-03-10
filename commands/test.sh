uv run generate.py \
    --checkpoint_path checkpoints/800_epochs/ddpm_epoch_0550.pt \
    --guidance_scale 2 \
    --eta 0. \
    --sampling_method "DDIM" \
    --num_steps_DDIM 50 \