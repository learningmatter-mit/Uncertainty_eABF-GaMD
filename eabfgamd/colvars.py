"""This module contains the ColVar class, which is used to compute collective variables (CVs) for
molecular simulations. The class supports various types of CVs, including dihedral angles and
uncertainty measures that are particularly relevant for the work in the paper associated with
this repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import torch
from nff.io.ase import AtomsBatch
from nff.train.builders.model import load_model
from nff.utils.scatter import compute_grad

from .prediction import get_prediction
from .uncertainty import (
    EnsembleUncertainty,
    EvidentialUncertainty,
    GMMUncertainty,
    MVEUncertainty,
)

if TYPE_CHECKING:
    from ase import Atoms

UNC_DICT = {
    "ensemble": EnsembleUncertainty,
    "evidential": EvidentialUncertainty,
    "mve": MVEUncertainty,
    "gmm": GMMUncertainty,
}


# THIS IS A COPY OF THE COLVAR CLASS FROM NFF, BUT ONLY WITH SUPPORT FOR
# UNCERTAINTY AND DIHEDRAL COLLECTIVE VARIABLES FOR LIMITED IMPORT MEMORY
class ColVar(torch.nn.Module):
    """Collective variable class.
    Computes cv and its Cartesian gradient.
    """

    def __init__(self, info_dict: dict):
        """Initialization of many class variables to avoid recurrent assignment
        with every forward call

        Args:
            info_dict (dict): dictionary that contains all the definitions of the CV,
                              the common key is type, which defines the CV function
                              all other keys are specific to each CV
        """
        super().__init__()
        self.info_dict = info_dict

        self.type = self.info_dict["type"]
        assert self.type in ["uncertainty", "dihedral", "gmm_bias"]

        if self.type == "dihedral":
            self._init_dihedral()

        elif self.type == "uncertainty":
            self._init_uncertainty()

        elif self.type == "gmm_bias":
            self._init_gmm_bias()

    def _init_dihedral(self):
        if len(self.info_dict["index_list"]) != 4:
            raise ValueError(f"CV ERROR: Invalid number of centers in definition of {self.type}!")

    def _init_uncertainty(self):
        self.device = self.info_dict["device"]
        if self.info_dict.get("model"):
            self.model = self.info_dict["model"].to(self.device)
        else:
            self.model = load_model(
                path=self.info_dict["model_path"],
                model_type=self.info_dict["model_type"],
            )
            self.model = self.model.to(self.device)

        self.model.eval()

        self.unc_class = UNC_DICT[self.info_dict["uncertainty_type"]](
            **self.info_dict["uncertainty_params"]
        )
        # turn off calibration for now in CP for initial fittings
        self.unc_class.calibrate = False

        if self.info_dict.get("uncertainty_type") == "gmm" and self.unc_class.is_fitted() is False:
            # if the unc_class already has a gm_model, then we don't need
            # to refit it
            print("COLVAR: Doing train prediction")
            _, train_predicted = get_prediction(
                model=self.model,
                dset=self.info_dict["train_dset"],
                batch_size=self.info_dict["batch_size"],
                device=self.device,
                requires_grad=False,
            )

            train_embedding = train_predicted["embedding"].detach().cpu().squeeze()
            train_atomic_numbers = torch.cat(
                [torch.LongTensor(at.get_atomic_numbers()) for at in self.info_dict["train_dset"]]
            )

            print("COLVAR: Fitting GMM")
            self.unc_class.fit_gmm(train_embedding, train_atomic_numbers)

        self.calibrate = self.info_dict["uncertainty_params"].get("calibrate", False)
        if self.calibrate:
            print("COLVAR: Fitting ConformalPrediction")
            calib_target, calib_predicted = get_prediction(
                model=self.model,
                dset=self.info_dict["calib_dset"],
                batch_size=self.info_dict["batch_size"],
                device=self.device,
                requires_grad=False,
            )
            # calib_predicted["embedding"] = calib_predicted["embedding"][0]

            # get atomic numbers
            calib_predicted["test_atomic_numbers"] = torch.cat(
                [torch.LongTensor(at.get_atomic_numbers()) for at in self.info_dict["calib_dset"]]
            )

            # get neighbor list
            nbr_lists, adjust_idx = [], 0
            for at in self.info_dict["calib_dset"]:
                at = AtomsBatch(
                    at,
                    cutoff=self.info_dict.get("cutoff", 5.0),
                    cutoff_skin=self.info_dict.get("cutoff_skin", 0.0),
                )
                at.update_nbr_list()
                nbr_list = at.nbr_list + adjust_idx
                adjust_idx = nbr_list.max().item() + 1

                nbr_lists.append(torch.LongTensor(nbr_list))
            calib_predicted["nbr_list"] = torch.cat(nbr_lists).to(self.device)

            calib_uncertainty = (
                self.unc_class(
                    results=calib_predicted,
                    num_atoms=calib_predicted["num_atoms"],
                    reset_min_uncertainty=True,
                )
                .detach()
                .cpu()
            )

            # set minimum uncertainty to scale to
            print(f"COLVAR: Setting min_uncertainty to {self.unc_class.umin}")

            calib_res = (
                get_residual(
                    targ=calib_target,
                    pred=calib_predicted,
                    num_atoms=calib_predicted["num_atoms"],
                    quantity=self.info_dict["uncertainty_params"].get("quantity", "energy_grad"),
                    order=self.info_dict["uncertainty_params"].get("order", "system_mean"),
                )
                .detach()
                .cpu()
            )
            self.unc_class.fit_conformal_prediction(
                calib_res,
                calib_uncertainty,
            )  # fit CP manually since calibration is turned off
            # turn on the calibration again
            self.unc_class.calibrate = True

    def _init_gmm_bias(self):
        self.device = self.info_dict["device"]
        self.kB = self.info_dict["kB"]  # Boltzmann constant: must be in the same unit as the energy
        self.T = self.info_dict["T"]

        if self.info_dict.get("model"):
            self.model = self.info_dict["model"].to(self.device)
        else:
            self.model = load_model(
                path=self.info_dict["model_path"],
                model_type=self.info_dict["model_type"],
            )
            self.model = self.model.to(self.device)

        self.model.eval()

        self.unc_class = UNC_DICT["gmm"](**self.info_dict["uncertainty_params"])

        # calibration is always off for gmm_bias
        self.unc_class.calibrate = False

        if self.unc_class.is_fitted() is False:
            print("COLVAR: Doing train prediction")
            _, train_predicted = get_prediction(
                model=self.model,
                dset=self.info_dict["train_dset"],
                batch_size=self.info_dict["batch_size"],
                device=self.device,
                requires_grad=False,
            )

            train_embedding = train_predicted["embedding"][0].detach().cpu().squeeze()

            print("COLVAR: Fitting GMM")
            self.unc_class.fit_gmm(train_embedding)

    def _get_com(
        self, indices: Union[int, list], xyz: torch.Tensor, masses: torch.Tensor = None
    ) -> torch.Tensor:
        """Get center of mass (com) of group of atoms"""
        if hasattr(indices, "__len__"):
            # compute center of mass for group of atoms
            center = torch.matmul(xyz[indices].T, masses[indices])
            m_tot = masses[indices].sum()
            com = center / m_tot

        else:
            # only one atom
            atom = int(indices)
            com = xyz[atom]

        return com

    def dihedral(
        self,
        index_list: list[Union[int, list]],
        xyz: torch.Tensor,
        return_grad: bool = True,
    ) -> torch.Tensor:
        """Torsion angle between four mass centers in range(-pi,pi)

        Args:
            index_list (list): list of indices of atoms to compute the dihedral
                index_list[0] = [ind00, ind01, ...]
                index_list[1] = [ind10, ind11, ...]
                index_list[2] = [ind20, ind21, ...]
                index_list[3] = [ind30, ind31, ...]
            xyz (torch.tensor): coordinates of atoms
            return_grad (bool): whether to return gradient or not

        Returns:
            cv (float): computed torsional angle
        """
        p1 = self._get_com(index_list[0], xyz)
        p2 = self._get_com(index_list[1], xyz)
        p3 = self._get_com(index_list[2], xyz)
        p4 = self._get_com(index_list[3], xyz)

        # get dihedral
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3

        q23_u = q23 / torch.linalg.norm(q23)

        n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
        n2 = q34 - torch.dot(q34, q23_u) * q23_u

        cv = torch.atan2(torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2))

        if return_grad is False:
            return cv, None

        cv_grad = compute_grad(inputs=xyz, output=cv)

        return cv, cv_grad

    def uncertainty(
        self,
        atoms: AtomsBatch,
        pred: Union[dict, None] = None,
        return_grad: bool = True,
    ):
        """Calculate uncertainty and its gradient

        Args:
            atoms (AtomsBatch): atoms object
            pred (dict | None): prediction dictionary
            return_grad (bool): whether to return gradient or not

        Returns:
            uncertainty (torch.Tensor): uncertainty value
            uncertainty_grad (torch.Tensor): uncertainty gradient
        """
        if pred is None:
            print("No prediction provided, generating one...")
            _, pred = get_prediction(
                self.model,
                dset=[atoms],
                batch_size=self.info_dict["batch_size"],
                device=self.device,
                requires_grad=True,
            )

        if not isinstance(pred["xyz"], torch.Tensor):
            pred["xyz"] = torch.tensor(
                pred["xyz"], dtype=torch.float32, device=self.device, requires_grad=True
            )

        # get neighbor list
        try:
            atoms.update_nbr_list()
            pred["nbr_list"] = torch.LongTensor(atoms.nbr_list).to(self.device)
        except Exception as e:
            print(f"ERROR getting nbr_list: {e}")
            raise

        # get atomic numbers
        try:
            atomic_nums = atoms.get_atomic_numbers()
            pred["test_atomic_numbers"] = torch.LongTensor(atomic_nums)
        except Exception as e:
            print(f"ERROR getting atomic numbers: {e}")
            raise

        try:
            uncertainty = self.unc_class(
                results=pred,
                num_atoms=pred["num_atoms"],
                reset_min_uncertainty=False,
                device=self.device,
            )
        except Exception as e:
            import traceback

            print(f"ERROR calculating uncertainty: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

        if return_grad is False:
            return uncertainty, None

        try:
            uncertainty_grad = compute_grad(
                inputs=pred["xyz"],
                output=uncertainty,
                allow_unused=True,
            )
            if uncertainty_grad is None:
                uncertainty_grad = torch.zeros_like(pred["xyz"])
        except Exception:
            import traceback

            raise

        return uncertainty, uncertainty_grad

    def gmm_bias(
        self, atoms: Atoms, pred: Union[dict, None] = None, return_grad: bool = True
    ) -> tuple:
        """Calculate GMM bias and its gradient

        Args:
            atoms (Atoms): atoms object
            pred (dict | None): prediction dictionary
            return_grad (bool): whether to return gradient or not

        Returns:
            bias (torch.Tensor): GMM bias value
            bias_grad (torch.Tensor): GMM bias gradient
        """
        if pred is None:
            _, pred = get_prediction(
                self.model,
                dset=[atoms],
                batch_size=self.info_dict["batch_size"],
                device=self.device,
                requires_grad=True,
            )

        if "embedding" in pred:
            pred["embedding"] = pred["embedding"][0]

        logP = self.unc_class.log_likelihood(pred["embedding"])
        bias = -self.kB * self.T * logP.mean()

        if return_grad is False:
            return bias, None

        bias_grad = compute_grad(
            inputs=pred["xyz"],
            output=bias,
            allow_unused=True,
        )
        if bias_grad is None:
            bias_grad = torch.zeros_like(pred["xyz"])

        return bias, bias_grad

    def forward(
        self, atoms: Atoms, pred: Union[dict, None] = None, return_grad: bool = True
    ) -> tuple:
        """Switch function to call the right CV-func

        Args:
            atoms (Atoms): atoms object
            pred (dict | None): prediction dictionary
            return_grad (bool): whether to return gradient or not

        Returns:
            cv (torch.Tensor): collective variable value
            cv_grad (torch.Tensor): collective variable gradient
        """
        xyz = torch.from_numpy(atoms.get_positions())
        xyz.requires_grad = True

        assert self.type in ["uncertainty", "dihedral", "gmm_bias"]

        if self.type == "uncertainty":
            cv, cv_grad = self.uncertainty(atoms, pred, return_grad=return_grad)
        elif self.type == "dihedral":
            cv, cv_grad = self.dihedral(self.info_dict["index_list"], xyz, return_grad=return_grad)
        elif self.type == "gmm_bias":
            cv, cv_grad = self.gmm_bias(atoms, pred, return_grad=return_grad)

        if return_grad is False:
            return cv.detach().cpu().numpy(), None

        return cv.detach().cpu().numpy(), cv_grad.detach().cpu().numpy()


def get_residual(
    targ: dict,
    pred: dict,
    num_atoms: List[int],
    quantity: str = "energy_grad",
    order: str = "system_mean",
) -> torch.Tensor:
    """Get the residual between the target and prediction.

    Args:
        targ (dict): target values
        pred (dict): predicted values
        num_atoms (List[int]): number of atoms in each system
        quantity (str): quantity to compute the residual for
        order (str): order of the residual
            - local_mean: mean of the residual for each atom
            - local_sum: sum of the residual for each atom
            - system_mean: mean of the residual for each system
            - system_sum: sum of the residual for each system
            - system_max: max of the residual for each system
            - system_min: min of the residual for each system
            - system_mean_squared: mean of the squared residual for each system
            - system_root_mean_squared: root mean squared of the residual for each system

    Returns:
        torch.Tensor: residual
    """
    if pred[quantity].shape != targ[quantity].shape:
        pred[quantity] = pred[quantity].mean(-1)

    res = targ[quantity] - pred[quantity]
    res = abs(res)

    if quantity == "energy":
        return res

    # force norm
    res = torch.linalg.norm(res, dim=-1)

    # get residual based on the order
    splits = torch.split(res, num_atoms)
    res = torch.stack(splits, dim=0)

    if "local" in order and "system" not in order:
        device = res.device
        dtype = res.dtype
        size = res.size(0)

        nbr_list = pred["nbr_list"].to(device)

        nbr_count = torch.zeros(size, dtype=dtype, device=device)
        nbr_count.scatter_add_(
            0, nbr_list[:, 0], torch.ones(nbr_list.size(0), dtype=dtype, device=device)
        )
        nbr_count.scatter_add_(
            0, nbr_list[:, 1], torch.ones(nbr_list.size(0), dtype=dtype, device=device)
        )

        res_sum = torch.zeros(size, dtype=dtype, device=device)
        res_sum.scatter_add_(0, nbr_list[:, 0], res[nbr_list[:, 1]])
        res_sum.scatter_add_(0, nbr_list[:, 1], res[nbr_list[:, 0]])

        if "local_mean" in order:
            local_res = res_sum / nbr_count
        elif "local_sum" in order:
            local_res = res_sum
        else:
            raise ValueError(f"Invalid order {order}")

        # reshape the res to (num_systems, num_atoms)
        splits = torch.split(local_res, list(num_atoms))
        res = torch.stack(splits, dim=0)

    if "system" in order:
        if "system_mean" in order:
            res = torch.stack([i.mean() for i in res])
        elif "system_sum" in order:
            res = torch.stack([i.sum() for i in res])
        elif "system_max" in order:
            res = torch.stack([i.max() for i in res])
        elif "system_min" in order:
            res = torch.stack([i.min() for i in res])
        elif "system_mean_squared" in order:
            res = torch.stack([(i**2).mean() for i in res])
        elif "system_root_mean_squared" in order:
            res = torch.stack([torch.sqrt((i**2).mean()) for i in res])
        else:
            raise ValueError(f"Invalid order {order}")

    else:
        raise ValueError(f"Invalid order {order}")

    return res
