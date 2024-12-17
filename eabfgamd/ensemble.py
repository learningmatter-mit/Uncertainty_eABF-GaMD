import os
import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, networks: list):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def to(self, device):
        for network in self.networks:
            network.to(device)
        return self

    def __getattr__(self, name):
        if name == "networks":
            return super().__getattr__(name)
        return getattr(self.networks[0], name)

    def forward(self, *args, **kwargs):
        outputs = [network(*args, **kwargs) for network in self.networks]

        stacked_outputs = {}
        for key in outputs[0].keys():
            if key == "xyz":
                stacked_outputs[key] = outputs[0][key]
                continue

            val = []
            for out in outputs:
                if out[key] is not None:
                    val.append(out[key])
                else:
                    val = None
                    break

            if key not in ["embedding", "node_feats"]:
                value = torch.stack(val, dim=-1) if val is not None else None
            else:
                value = val

            stacked_outputs[key] = value

        return stacked_outputs


def load_ensemble(
    model_path: str,
    model_type: str,
    device: str,
):
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "ensemble")
    else:
        if not model_path.endswith("/ensemble"):
            model_path = os.path.join(os.path.dirname(model_path), "ensemble")

    assert os.path.exists(
        model_path), f"Ensemble model not found at {model_path}."

    model = torch.load(model_path, map_location=device).to(device)

    return model


def get_model_paths(config: dict) -> list:
    mconfig = config["model"]
    model_type = mconfig["model_type"]

    # Determine model path name based on model type
    if model_type in ["schnet", "painn"]:
        model_path_name = "best_model"
    elif model_type in ["mace", "mace_mp"]:
        if config["train"]["swa"]:
            model_path_name = "mace_swa.model"
        else:
            model_path_name = "mace.model"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Helper function to check and append valid paths
    def append_model_path(outdir, model_path_name):
        path = f"{outdir}/{model_path_name}"
        if os.path.exists(path):
            model_paths.append(path)

    # Initialize list to store valid model paths
    model_paths = []

    # Check for model paths based on outdir type
    if isinstance(mconfig["outdir"], str) and mconfig["num_networks"] == 1:
        append_model_path(mconfig["outdir"], model_path_name)
    elif isinstance(mconfig["outdir"], list):
        for p in mconfig["outdir"]:
            append_model_path(p, model_path_name)

    # Ensure at least one valid model path was found
    assert model_paths, "No model found"

    return model_paths


def get_ensemble_path(path: str) -> str:
    if isinstance(path, str):
        ensemble_path = f"{path}/ensemble"
    elif isinstance(path, (list, tuple)) and len(path) == 1:
        ensemble_path = f"{path[0]}/ensemble"
    else:
        common_prefix = os.path.commonprefix(path)
        if not common_prefix.endswith("model/"):
            common_prefix = common_prefix[: common_prefix.rfind("model/") + 5]
        ensemble_path = f"{common_prefix}/ensemble"

    return ensemble_path


def make_ensemble(model_paths: list) -> Ensemble:
    from mace.modules.models import MACE

    if len(model_paths) == 1:
        model = torch.load(model_paths[0], map_location="cuda")
        if isinstance(model, Ensemble):
            return model
        ensemble = Ensemble([model])
        return ensemble

    model_list = []
    for model_path in model_paths:
        model = torch.load(model_path, map_location="cuda")
        model_list.append(model)
    ensemble = Ensemble(model_list)

    return ensemble
