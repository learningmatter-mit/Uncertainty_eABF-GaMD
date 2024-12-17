import os
import sys
import argparse

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.dihedrals import handle_dihedral_setting
from eabfgamd.misc import load_config


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--num_bins", type=int, default=361)
    parser.add_argument("--new_geom_path", type=str, default="new_geom.xyz")
    args = parser.parse_args()

    # load the config file
    config = load_config(args.config_file)

    handle_dihedral_setting(
        config=config,
        num_bins=args.num_bins,
        new_geom_file=args.new_geom_file,
    )


if __name__ == "__main__":
    argument_parser()
