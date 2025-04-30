"""Prediction module for MACE models. This module provides functions to
load MACE models, make predictions on datasets, and calculate errors. These
functions are used to
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from ase.atoms import Atoms
from mace import data
from mace.tools import torch_geometric, utils
from nff.utils.cuda import batch_detach
from tqdm import tqdm

from .misc import get_atoms

__all__ = [
    "get_atoms",
    "get_errors",
    "get_prediction",
    "get_prediction_and_errors",
]


def get_mace_prediction(
    model: torch.nn.Module,
    dset: List[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
) -> dict:
    """Get predictions from a MACE model. Gathers the predictions for a list of
    ASE Atoms objects. The predictions include energy, forces, and node features (embeddings).
    The output dict also contains the corresponding atomic numbers and positions (xyz) in each
    prediction.

    Args:
        model (torch.nn.Module): MACE model.
        dset (List[Atoms]): List of ASE Atoms objects.
        batch_size (int): Batch size for predictions. Default is 10.
        device (str): Device to use for predictions (e.g., 'cuda' or 'cpu').
        requires_grad (bool): Whether to compute gradients.

    Returns:
        predicted (dict): predictions from MACE
    """
    predicted = {}
    predicted["num_atoms"] = [len(atoms) for atoms in dset]
    configs = [data.config_from_atoms(atoms) for atoms in dset]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    _predicted = []
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        output = model(batch.to_dict(), training=requires_grad)
        pred = {
            "energy": output["energy"],
            "energy_grad": -output["forces"],
            "embedding": output["node_feats"],
            # "xyz": output["xyz"],
            "xyz": batch.positions,
        }

        if not requires_grad:
            pred = batch_detach(pred)

        _predicted.append(pred)

        batch = batch.cpu()
        output = batch_detach(output)
        pred = batch_detach(pred)

    # Concatenate data
    for k in _predicted[0]:
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
                # Rui method
                value = torch.cat(list(value), dim=0)

        predicted[k] = value

    return predicted


def get_prediction(
    model: torch.nn.Module,
    dset: List[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
    **kwargs,
) -> Tuple[dict, dict]:
    """Get predictions from a MACE model and the target values for the corresponding
    ASE Atoms objects. The predictions include energy, forces, and node features (embeddings).
    This function calls the `get_mace_prediction` function to obtain the predictions and
    then retrieves the target values from the ASE Atoms objects.

    Args:
        model (torch.nn.Module): MACE model.
        dset (List[Atoms]): List of ASE Atoms objects.
        batch_size (int): Batch size for predictions. Default is 10.
        device (str): Device to use for predictions (e.g., 'cuda' or 'cpu').
        requires_grad (bool): Whether to compute gradients.
        **kwargs: Additional arguments for the prediction function.

    Returns:
        target (dict): Target values (energy and forces).
        predicted (dict): Predictions from MACE.
    """
    predicted = get_mace_prediction(
        model,
        dset,
        batch_size=batch_size,
        device=device,
        requires_grad=requires_grad,
        **kwargs,
    )
    predicted["identifier"] = [at.info.get("identifier", None) for at in tqdm(dset)]

    try:
        target = {
            "energy": np.array([at.get_potential_energy() for at in dset]),
            "energy_grad": np.concatenate([-at.get_forces(apply_constraint=False) for at in dset]),
        }
    except TypeError:
        target = {
            "energy": np.array([at.get_potential_energy().detach().cpu() for at in dset]),
            "energy_grad": np.concatenate(
                [-at.get_forces(apply_constraint=False).detach().cpu() for at in dset]
            ),
        }

    target["energy"] = torch.tensor(target["energy"]).to(predicted["energy"].device)
    target["energy_grad"] = torch.tensor(target["energy_grad"]).to(predicted["energy_grad"].device)

    return target, predicted


def to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert data to NumPy array if it is a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def calculate_mae(pred: np.ndarray, targ: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) between predicted and target values."""
    return np.mean(np.abs(pred - targ))


def calculate_rmse(pred: np.ndarray, targ: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE) between predicted and target values."""
    return np.sqrt(np.mean((pred - targ) ** 2))


def calculate_r2(pred: np.ndarray, targ: np.ndarray) -> float:
    """Calculate R-squared (R2) score between predicted and target values."""
    return 1 - np.sum((pred - targ) ** 2) / np.sum((targ - np.mean(targ)) ** 2)


def calculate_max_error(pred: np.ndarray, targ: np.ndarray) -> float:
    """Calculate maximum absolute error between predicted and target values."""
    return np.max(np.abs(pred - targ))


def get_errors(
    predicted: dict,
    target: dict,
    metrics: Tuple[str] = ("mae", "rmse", "r2", "max_error"),
    keys: Tuple[str] = ("energy", "energy_grad"),
) -> dict:
    """Calculate errors between predicted and target values for specified metrics and keys.

    Args:
        predicted (dict): Predicted values.
        target (dict): Target values.
        metrics (tuple): Tuple of metrics to calculate. Default is
            ("mae", "rmse", "r2", "max_error").
        keys (tuple): Tuple of keys to calculate errors for. Default is
            ("energy", "energy_grad").

    Returns:
        errors (dict): Dictionary containing calculated errors for each key and metric.
    """
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
    model: torch.nn.Module, dset: List[Atoms], batch_size: int, device: str
) -> Tuple[dict, dict, dict]:
    """Get predictions and errors from a MACE model for a list of ASE Atoms objects.

    Args:
        model (torch.nn.Module): MACE model.
        dset (List[Atoms]): List of ASE Atoms objects.
        batch_size (int): Batch size for predictions.
        device (str): Device to use for predictions (e.g., 'cuda' or 'cpu').

    Returns:
        target (dict): Target values (energy and forces).
        predicted (dict): Predictions from MACE.
        errors (dict): Calculated errors between predicted and target values.
    """
    target, predicted = get_prediction(model, dset, batch_size, device)

    target = batch_detach(target)
    predicted = batch_detach(predicted)

    metrics = ("mae", "rmse", "r2", "max_error")
    keys = ("energy", "energy_grad")
    errors = get_errors(predicted, target, metrics=metrics, keys=keys)

    return target, predicted, errors
