import argparse
import os
import sys

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.ala import Paths as AlaPaths
from eabfgamd.misc import load_config
from eabfgamd.sampling import handle_sampling
from eabfgamd.silica import Paths as SilicaPaths

paths = {"ala": AlaPaths, "silica": SilicaPaths}


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    # load config file
    config = load_config(args.config_file)

    sampled_dset = handle_sampling(config)

    if sampled_dset is None:
        os.makedirs(f"{paths[config['system']].params_dir}/error", exist_ok=True)
        os.rename(args.config_file, args.config_file.replace("inbox", "error"))


if __name__ == "__main__":
    argument_parser()
