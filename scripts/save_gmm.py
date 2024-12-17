import argparse
import os
import sys

from ase.io import read
from nff.io.gmm import GaussianMixture

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.ensemble import load_ensemble
from eabfgamd.misc import load_config
from eabfgamd.prediction import get_prediction
from eabfgamd.ala import Paths as AlaPaths
from eabfgamd.silica import Paths as SilicaPaths


def initialize_gmm(
    n_clusters: int,
    covariance_type: str = "full",
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    n_init: int = 1,
    init_params: str = "kmeans",
) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
    )

    return gmm


def fit_and_save_gmm(
    config: dict,
) -> None:
    model = load_ensemble(
        model_path=config["model"]["outdir"],
        model_type=config["model"]["model_type"],
        device=config["train"]["device"],
    )

    train_dset = read(config["dset"]["path"], index=":")

    _, train_pred = get_prediction(
        model=model,
        dset=train_dset,
        batch_size=config["dset"]["batch_size"],
        device="cuda",
        requires_grad=False,
    )

    train_embedding = train_pred["embedding"][0].detach().cpu().numpy()

    gmm = initialize_gmm(n_clusters=config["uncertainty"]["n_gmm"])

    gmm.fit(train_embedding)

    gmm.save(config["uncertainty"]["gmm_path"])
    print(f"GMM saved to {config['uncertainty']['gmm_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, choice=['Ala', 'Silica'])
    parser.add_argument("--model_id", "-m", type=str, required=True)
    parser.add_argument("--gen", "-g", type=int, required=True)
    parser.add_argument(
        "--n_gmm",
        "-n",
        type=int,
        help="Number of GMM components. Overwrite config value if value provided",
    )
    parser.add_argument(
        "--gmm_path",
        "-p",
        type=str,
        help="Path to save GMM. Overwrite config value if value provided",
    )
    args = parser.parse_args()

    if args.system == "Ala":
        model_dir = f"{AlaPaths.results_dir}/mace_gmm/{args.model_id}/gen{args.gen}/model"
    elif args.system == "Silica":
        model_dir = f"{SilicaPaths.results_dir}/mace_gmm/{args.model_id}/gen{args.gen}/model"

    config = load_config(f"{model_dir}/config.yaml")

    if args.n_gmm is not None:
        config["uncertainty"]["n_gmm"] = args.n_gmm

    if args.gmm_path is not None:
        config["uncertainty"]["gmm_path"] = args.gmm_path

    fit_and_save_gmm(config)


if __name__ == "__main__":
    main()
