import os
import sys
import argparse

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.enhsamp import run_unbiased_dynamics
from eabfgamd.misc import load_config


def argument_parser():
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str, default="config.yaml")
    args.add_argument("--force_restart", action="store_true", default=False)
    args = args.parse_args()

    # load config file
    config = load_config(args.config_file)

    print(config["enhsamp"]["traj_dir"])
    run_unbiased_dynamics(config=config, force_restart=args.force_restart)


if __name__ == "__main__":
    argument_parser()
