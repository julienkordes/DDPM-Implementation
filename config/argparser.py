import configargparse
from fractions import Fraction


def save_opts(args, fn):
    with open(fn, "w") as fw:
        for items in vars(args):
            fw.write("%s %s\n" % (items, vars(args)[items]))


def str2bool(v):
    v = v.lower()
    if v in ("yes", "true", "t", "1"):
        return True
    elif v in ("no", "false", "f", "0"):
        return False
    raise ValueError(
        "Boolean argument needs to be true or false. " "Instead, it is %s." % v
    )


def fraction_to_float(value):
    try:
        # Try to parse the input as a fraction
        fraction_value = Fraction(value)
        # Convert the fraction to a float
        float_value = float(fraction_value)
        return float_value
    except ValueError:
        # If parsing fails, raise an error
        raise configargparse.argparse.ArgumentTypeError(
            f"{value} is not a valid fraction."
        )

def load_opts():
    get_parser()
    return get_parser().parse_args()


def get_parser():
    parser = configargparse.ArgumentParser(description="main")
    parser.register("type", bool, str2bool)

    parser.add("-c", "--config", is_config_file=True, help="config file path")

    parser.add_argument("--data_paths", type=str, default="config/paths.json", help="path to data paths json file")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64, help="Taille de sortie de la conv_init")
    parser.add_argument("--num_res_block", type=int, default=2)
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--attention_resolution", type=int, default=16, help="resolution à laquelle on applique l'attention")
    parser.add_argument("--time_emb_dim", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="dir pour save les checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="dir pour load les checkpoints")
    parser.add_argument("--mixed_precision", type=bool, default=True)
    parser.add_argument("--guidance_scale", type=int, default=3, help="Coefficient pour le conditionning")
    parser.add_argument("--num_samples_FID", type=int, default=10000)
    parser.add_argument("--eta", type=float, default = 0.)
    parser.add_argument("--num_steps_DDIM", type=int, default=50)
    parser.add_argument("--sampling_method", type=str, default="DDPM")

    return parser