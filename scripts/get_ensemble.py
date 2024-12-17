import os
import sys
import argparse
import torch

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.misc import load_config
from eabfgamd.ensemble import Ensemble, get_model_paths, get_ensemble_path, make_ensemble


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config_path)

    model_paths = get_model_paths(config)

    ensemble = make_ensemble(model_paths)

    ensemble_path = get_ensemble_path(config['model']['outdir'])

    torch.save(ensemble, ensemble_path)
    print(f"AGG: Ensemble model saved to {ensemble_path}")


if __name__ == "__main__":
    argument_parser()
