import argparse
import os
import sys

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.ala import Paths as AlaPaths
from eabfgamd.misc import load_config
from eabfgamd.saving import handle_save
from eabfgamd.silica import Paths as SilicaPaths

paths = {"ala": AlaPaths, "silica": SilicaPaths}


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="path to config file")
    args = parser.parse_args()

    # read config file
    config = load_config(args.config_file)

    combined_dset = handle_save(config)
    if combined_dset is None:
        os.makedirs(f"{paths[config['system']].params_dir}/error", exist_ok=True)
        os.rename(args.config_file, args.config_file.replace("inbox", "error"))


if __name__ == "__main__":
    argument_parser()
