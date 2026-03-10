import os

from config.argparser import load_opts, save_opts
from trainers import get_trainer

def main():
    args = load_opts()
    exp_dir = args.output_dir
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.txt")
    save_opts(args, config_path)
    trainer = get_trainer("trainer")
    trainer(args)

if __name__ == "__main__":
    main()
