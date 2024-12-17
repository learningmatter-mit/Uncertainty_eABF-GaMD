import os
import sys
import argparse

sys.path.append(f"{os.getenv('PROJECTSDIR')}/uncertainty_eABF-GaMD")
from eabfgamd.enhsamp import handle_multiproc_enhsamp, run
from eabfgamd.misc import load_config


def argument_parser():
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str, default="config.yaml")
    args.add_argument("--cpu_count", type=int, default=3)
    args.add_argument("--force_restart", action="store_true", default=False)
    args.add_argument("--individual", action='store_true', default=False)
    args = args.parse_args()

    # load config file
    config = load_config(args.config_file)

    if args.individual is True:
        print(config['enhsamp']['traj_dir'])
        run(config=config, force_restart=args.force_restart)
    else:
        handle_multiproc_enhsamp(config, args.cpu_count, args.force_restart)


if __name__ == "__main__":
    argument_parser()
