import os
import sys
import argparse

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.calc import handle_calc
from eabfgamd.misc import load_config


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="path to config file")
    args = parser.parse_args()

    # read config file
    config = load_config(args.config_file)

    handle_calc(config)


if __name__ == "__main__":
    argument_parser()
