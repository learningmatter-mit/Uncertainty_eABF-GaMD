from typing import List, Tuple, Union

import numpy as np
import torch
from ase.atoms import Atoms
from nff.utils.cuda import batch_detach, batch_to
from mace import data
from mace.tools import torch_geometric, utils

from .misc import get_atoms

__all__ = [
    "get_prediction",
    "get_errors",
    "get_prediction_and_errors",
    "get_atoms",
]


def get_mace_prediction(
    model,
    dset: List[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
):
    configs = [data.config_from_atoms(atoms) for atoms in dset]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    _predicted = []
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=requires_grad)
        pred = {
            "energy": output["energy"],
            "energy_grad": -output["forces"],
            "embedding": output["node_feats"],
            "xyz": output["xyz"],
        }

        if not requires_grad:
            pred = batch_detach(pred)

        _predicted.append(pred)

        batch = batch.cpu()
        output = batch_detach(output)
        pred = batch_detach(pred)

    # Concatenate data
    predicted = {}
    for k in _predicted[0].keys():
        value = [p[k] for p in _predicted]
        if k not in ["embedding", "node_feats"]:
            value = torch.cat(value, dim=0)
        else:
            # if the model is an ensemble that has multiple networks
            if hasattr(model, "networks"):
                new_value = [[] for _ in range(len(value[0]))]
                for val in value:
                    for i, v in enumerate(val):
                        new_value[i].append(v)
                new_value = [torch.cat(v) for v in new_value]
                value = new_value
                del new_value
            # if the model is a single model (non-ensemble)
            else:
                value = torch.cat([v for v in value], dim=0)

        predicted[k] = value

    predicted["num_atoms"] = [len(atoms) for atoms in dset]

    return predicted


def get_prediction(
    model,
    dset: List[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
    **kwargs,
) -> Tuple[dict, dict]:
    predicted = get_mace_prediction(
        model,
        dset,
        batch_size=batch_size,
        device=device,
        requires_grad=requires_grad,
        **kwargs,
    )
    predicted["identifier"] = [at.info["identifier"] for at in dset]

    target = {
        "energy": np.array([at.get_potential_energy() for at in dset]),
        "energy_grad": np.concatenate(
            [-at.get_forces(apply_constraint=False) for at in dset]
        ),
    }

    target["energy"] = torch.tensor(
        target["energy"]).to(predicted["energy"].device)
    target["energy_grad"] = torch.tensor(target["energy_grad"]).to(
        predicted["energy_grad"].device
    )

    return target, predicted


def to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert data to NumPy array if it is a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def calculate_mae(pred, targ):
    return np.mean(np.abs(pred - targ))


def calculate_rmse(pred, targ):
    return np.sqrt(np.mean((pred - targ) ** 2))


def calculate_r2(pred, targ):
    return 1 - np.sum((pred - targ) ** 2) / np.sum((targ - np.mean(targ)) ** 2)


def calculate_max_error(pred, targ):
    return np.max(np.abs(pred - targ))


def get_errors(
    predicted: dict,
    target: dict,
    metrics=("mae", "rmse", "r2", "max_error"),
    keys=("energy", "energy_grad"),
) -> dict:
    errors = {}

    for key in keys:
        if key in predicted and key in target:
            pred = to_numpy(predicted[key])
            targ = to_numpy(target[key])

            if key == "energy_grad":
                pred, targ = -pred, -targ
                key = "forces"

            errors[key] = errors.get(key, {})

            # Adjust dimensionality if needed
            if pred.ndim > targ.ndim:
                pred = pred.mean(axis=-1)
            elif targ.ndim > pred.ndim:
                targ = targ.mean(axis=-1)

            # Compute requested metrics
            if "mae" in metrics:
                errors[key]["mae"] = calculate_mae(pred, targ)
            if "rmse" in metrics:
                errors[key]["rmse"] = calculate_rmse(pred, targ)
            if "r2" in metrics:
                errors[key]["r2"] = calculate_r2(pred, targ)
            if "max_error" in metrics:
                errors[key]["max_error"] = calculate_max_error(pred, targ)

    return errors


def get_prediction_and_errors(
    model, dset: List[Atoms], batch_size: int, device: str
) -> Tuple[dict, dict, dict]:
    target, predicted = get_prediction(model, dset, batch_size, device)

    target = batch_detach(target)
    predicted = batch_detach(predicted)

    metrics = ("mae", "rmse", "r2", "max_error")
    keys = ("energy", "energy_grad")
    errors = get_errors(predicted, target, metrics=metrics, keys=keys)

    return target, predicted, errors
