import os
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch

from sklearn.mixture import GaussianMixture
from amptorch.uncertainty import ConformalPrediction

__all__ = [
    "Uncertainty",
    "EnsembleUncertainty",
    "EvidentialUncertainty",
    "MVEUncertainty",
    "GMMUncertainty",
    "ConformalPrediction",
]


class Uncertainty:
    """
    Base class for uncertainty estimation.

    Args:
        order (str): Order of the uncertainty estimation (atomic, local, system).
            Non-hybrid choices include:
            * atomic: Uncertainty is calculated for each atom (n_systems, n_atoms)
            * local_mean: Mean of uncertainties of the neighbors of each atom (n_systems, n_atoms)
            * local_sum: Sum of uncertainties of the neighbors of each atom (n_systems, n_atoms)
            * system_mean: Mean of uncertainties of all atoms in the system (n_systems,)
            * system_sum: Sum of uncertainties of all atoms in the system (n_systems,)
            * system_max: Maximum of uncertainties of all atoms in the system (n_systems,)
            * system_min: Minimum of uncertainties of all atoms in the system (n_systems,)
            * system_mean_squared: Mean squared of uncertainties of all atoms in the system (n_systems,)
            * system_root_mean_squared: Root mean squared of uncertainties of all atoms in the system (n_systems,)
            Hybrid choices include:
            * atomic_system_sum: Sum of atomic uncertainties in the system (n_systems,). Equivalent to system_sum
            * atomic_system_mean: Mean of atomic uncertainties in the system (n_systems,). Equivalent to system_mean
            * atomic_system_max: Maximum of atomic uncertainties in the system (n_systems,). Equivalent to system_max
            * atomic_system_min: Minimum of atomic uncertainties in the system (n_systems,). Equivalent to system_min
            * atomic_system_mean_squared: Mean squared of atomic uncertainties in the system (n_systems,). Equivalent to system_mean_squared
            * atomic_system_root_mean_squared: Root mean squared of atomic uncertainties in the system (n_systems,). Equivalent to system_root_mean_squared
            * local_mean_system_sum: Sum of mean uncertainties of the neighbors of each atom in the system (n_systems,).
            * local_mean_system_mean: Mean of mean uncertainties of the neighbors of each atom in the system (n_systems,). Equivalent to just doing system_mean.
            * local_sum_system_sum: Sum of sum uncertainties of the neighbors of each atom in the system (n_systems,). Equivalent to just doing system_sum.
            * local_sum_system_mean: Mean of sum uncertainties of the neighbors of each atom in the system (n_systems,).
        calibrate (bool): Calibrate the uncertainty using Conformal Prediction
        cp_alpha (float): Significance level for the Conformal Prediction model
        min_uncertainty (float): Minimum uncertainty value
    """

    def __init__(
        self,
        order: str,
        calibrate: bool,
        cp_alpha: Union[None, float] = None,
        min_uncertainty: Union[float, dict, None] = None,
        set_min_uncertainty_at_level: str = "system",
        device="cuda",
        nbr_key="nbr_list",
        *args,
        **kwargs,
    ):
        self._check_and_set_order(order)
        self.calibrate = calibrate
        self.umin = min_uncertainty
        self.set_min_uncertainty_at_level = set_min_uncertainty_at_level
        self.device = device
        self.nbr_key = nbr_key

        if self.calibrate:
            assert cp_alpha is not None, "cp_alpha must be specified for calibration"

            self.CP = ConformalPrediction(alpha=cp_alpha)

    def __call__(self, *args, **kwargs):
        return self.get_uncertainty(*args, **kwargs)

    def _check_and_set_order(self, order: str) -> None:
        """
        Check if the order is valid.
        """
        assert order in [
            "atomic",
            "atomic_system_sum",
            "atomic_system_mean",
            "atomic_system_max",
            "atomic_system_min",
            "atomic_system_mean_squared",
            "atomic_system_root_mean_squared",
            "local_sum",
            "local_mean",
            "local_mean_system_sum",
            "local_mean_system_mean",
            "local_mean_system_max",
            "local_mean_system_min",
            "local_mean_system_mean_squared",
            "local_mean_system_root_mean_squared",
            "system_sum",
            "system_mean",
            "system_max",
            "system_min",
            "system_mean_squared",
            "system_root_mean_squared",
        ], f"{order} not implemented"

        if "atomic" in order and "system" in order:
            print(f"{order} is equivalent to {order[order.find('system'):]}")

        if "local_sum" in order and "system_sum" in order:
            print(f"{order} is equivalent to {order[order.find('system'):]}")

        if "local_mean" in order and "system_mean" in order:
            print(f"{order} is equivalent to {order[order.find('system'):]}")

        self.order = order

    def _update_min_uncertainty(
        self,
        min_uncertainty: float,
        key: Union[str, int, None] = None,
        force: bool = False,
    ) -> None:
        """
        Update the minimum uncertainty value to be used for scaling the uncertainty.
        Args:
            min_uncertainty (float): Minimum uncertainty value
            key (str, int, None): Key for the minimum uncertainty value if a
                dictionary is used for storing the minimum uncertainty values
                for different applications (e.g. atomic numbers, species, etc.).
                If None, then the minimum uncertainty value is set for all
                applications.
            force (bool): Forcefully overwrite the minimum uncertainty value
        """

        if getattr(self, "umin", None) is None:
            if key is not None:
                self.umin = {key: min_uncertainty}
            else:
                self.umin = min_uncertainty
            return

        if isinstance(self.umin, dict):
            if key is None:
                raise ValueError(
                    "Key must be provided when using dictionary for umin.")
            if key in self.umin and force is False:
                return
            if key in self.umin and force is True:
                warnings.warn(
                    f"Uncertainty: min_uncertainty for {key} already set to {self.umin[key]}. Overwriting to {min_uncertainty}"
                )

            self.umin[key] = min_uncertainty

        else:
            if key is not None:
                raise ValueError(
                    "Key must be None when not using dictionary for umin.")
            if force is False:
                return
            if force is True:
                warnings.warn(
                    f"Uncertainty: min_uncertainty already set to {self.umin}. Overwriting to {min_uncertainty}"
                )

            self.umin = min_uncertainty

    def set_min_uncertainty(
        self, uncertainty: Union[np.ndarray, torch.Tensor], force: bool = False
    ) -> None:
        """
        Set the minimum uncertainty value.
        """

        # if uncertainty is per atom and the element indices are given
        if hasattr(self, "element_indices") is True:
            uncertainty = uncertainty.flatten()
            for key, idx in self.element_indices.items():
                min_u = uncertainty[idx].min().item()
                self._update_min_uncertainty(min_u, key=key, force=force)

        # everything else takes the minimum uncertainty of the entire array
        else:
            min_u = uncertainty.min().item()
            self._update_min_uncertainty(min_u, key=None, force=force)

    def normalize_to_min_uncertainty(
        self, uncertainty: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale the uncertainty to the minimum value.
        """

        # if the minimum uncertainty has not been defined yet, then just return the uncertainty
        if hasattr(self, "umin") is False:
            return uncertainty

        # if the minimum uncertainty is not type/key/condition-dependent
        if isinstance(self.umin, float):
            if "system_mean_squared" in self.order:
                uncertainty = uncertainty - self.umin**2
            else:
                uncertainty = uncertainty - self.umin

        # if the minimum uncertainty is type/key/condition-dependent
        elif isinstance(self.umin, dict):
            assert hasattr(
                self, "element_indices") is True, "Element indices not saved"

            num_system, num_atoms = uncertainty.shape
            uncertainty = uncertainty.flatten()

            if "system_mean_squared" in self.order:
                for Z, min_u in self.umin.items():
                    idx = self.element_indices[Z]
                    uncertainty[idx] = uncertainty[idx] - min_u**2

            else:
                for Z, min_u in self.umin.items():
                    idx = self.element_indices[Z]
                    uncertainty[idx] = uncertainty[idx] - min_u

            uncertainty = uncertainty.reshape(num_system, num_atoms)

        return uncertainty

    def save_indices_of_atom_types(
        self,
        key: Union[str, int],
        indices: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """
        Save the indices of atoms so that the uncertainty can be scaled to the
        type/key/condition-dependent minimum uncertainty. Only needed if there
        is need to scale the uncertainty to a minimum value that is dependent
        on the atomic number, species, etc.
        """
        if not hasattr(self, "element_indices"):
            self.element_indices = {}
            self.element_indices[key] = indices
        else:
            self.element_indices[key] = indices

    def fit_conformal_prediction(
        self,
        residuals_calib: Union[np.ndarray, torch.Tensor],
        heuristic_uncertainty_calib: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Fit the Conformal Prediction model to the calibration data.
        """
        self.CP.fit(residuals_calib, heuristic_uncertainty_calib)

    def calibrate_uncertainty(
        self, uncertainty: Union[np.ndarray, torch.Tensor], *args, **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calibrate the uncertainty using Conformal Prediction.
        """
        if hasattr(self.CP, "qhat") is False:
            raise Exception("Uncertainty: ConformalPrediction not fitted.")

        cp_uncertainty, qhat = self.CP.predict(uncertainty)

        return cp_uncertainty

    def get_atomic_uncertainty(
        self,
        results: dict,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get the uncertainty for each atom.
        """
        return NotImplementedError

    def get_local_uncertainty(
        self,
        atomic_uncertainty: torch.Tensor,
        num_atoms: List[int],
        nbr_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Get the uncertainty for local environment within a certain cutoff
        sphere of atoms.
        """
        assert "local" in self.order, f"{self.order} does not contain 'local'"

        atomic_uncertainty = atomic_uncertainty.to(self.device)
        dtype = atomic_uncertainty.dtype

        nbr_list = nbr_list.to(self.device)
        size = nbr_list.max().item() + 1

        # assert that the shape of atomic_uncertainty is (num_systems, num_atoms)
        assert (
            len(atomic_uncertainty) == len(num_atoms)
        ), f"Number of systems do not match. Lengths of atomic_uncertainty and num_atoms are {len(atomic_uncertainty)} and {len(num_atoms)}, respectively."

        atoms_match = np.array(
            [len(u) == n for u, n in zip(atomic_uncertainty, num_atoms)]
        )
        if not all(atoms_match):
            not_match = np.where(~atoms_match)[0]
            raise AssertionError(
                f"Number of atoms in systems {not_match} do not match. Expected numbers of atoms in atomic_uncertainty are {[num_atoms[i] for i in not_match]} but got {atomic_uncertainty[not_match]}."
            )

        # assert that the atomic uncertainty has the same number of atoms as the neighbor list
        atomic_uncertainty = atomic_uncertainty.flatten()
        assert (
            len(atomic_uncertainty) == size
        ), f"Size of atomic uncertainty does not match. Expected size is {size} but got {len(atomic_uncertainty)}."

        nbr_count = torch.zeros(size, dtype=dtype, device=self.device)
        nbr_count.scatter_add_(
            0,
            nbr_list[:, 0],
            torch.ones(nbr_list.size(0), dtype=dtype, device=self.device),
        )
        nbr_count.scatter_add_(
            0,
            nbr_list[:, 1],
            torch.ones(nbr_list.size(0), dtype=dtype, device=self.device),
        )

        uncertainty_sum = torch.zeros(size, dtype=dtype, device=self.device)
        uncertainty_sum.scatter_add_(
            0, nbr_list[:, 0], atomic_uncertainty[nbr_list[:, 1]]
        )
        uncertainty_sum.scatter_add_(
            0, nbr_list[:, 1], atomic_uncertainty[nbr_list[:, 0]]
        )

        # Get the local uncertainty based on the order
        if self.order.startswith("local_mean"):
            local_uncertainty = uncertainty_sum / nbr_count
        elif self.order.startswith("local_sum"):
            local_uncertainty = uncertainty_sum
        else:
            raise TypeError(f"{self.order} not implemented")

        # reshape the uncertainty to (num_systems, num_atoms)
        splits = torch.split(local_uncertainty, list(num_atoms))
        local_uncertainty = torch.stack(splits, dim=0)

        return local_uncertainty

    def get_system_uncertainty(
        self, uncertainty: torch.Tensor, num_atoms: List[int]
    ) -> torch.Tensor:
        """
        Get the uncertainty for the entire system.
        """
        assert "system" in self.order, f"{self.order} does not contain 'system'"

        # assert that the shape of atomic_uncertainty is (num_systems, num_atoms)
        assert (
            len(uncertainty) == len(num_atoms)
        ), f"Number of systems do not match. Lengths of uncertainty and num_atoms are {len(uncertainty)} and {len(num_atoms)}, respectively."

        atoms_match = np.array(
            [len(u) == n for u, n in zip(uncertainty, num_atoms)])
        if not all(atoms_match):
            not_match = np.where(~atoms_match)[0]
            raise AssertionError(
                f"Number of atoms in systems {not_match} do not match. Expected numbers of atoms in uncertainty are {[num_atoms[i] for i in not_match]} but got {uncertainty[not_match]}."
            )

        # get the desired system uncertainty based on the order
        if self.order.endswith("system_sum"):
            system_uncertainty = uncertainty.sum(dim=-1)
        elif self.order.endswith("system_mean"):
            system_uncertainty = uncertainty.mean(dim=-1)
        elif self.order.endswith("system_max"):
            system_uncertainty = uncertainty.max(dim=-1).values
        elif self.order.endswith("system_min"):
            system_uncertainty = uncertainty.min(dim=-1).values
        elif self.order.endswith("system_mean_squared"):
            system_uncertainty = (uncertainty**2).mean(dim=-1)
        elif self.order.endswith("system_root_mean_squared"):
            system_uncertainty = (uncertainty**2).mean(dim=-1) ** 0.5

        return system_uncertainty

    def get_uncertainty(
        self, results: dict, reset_min_uncertainty: bool = False, *args, **kwargs
    ):
        """
        Get the uncertainty from the results.
        """
        uncertainty = self.get_atomic_uncertainty(
            results=results, *args, **kwargs)
        num_atoms = results["num_atoms"]

        # Get the uncertainty based on the order
        # If the order is "atomic", then the uncertainty is calculated for each atom, no action needed here
        if "atomic" in self.order:
            pass

            if self.set_min_uncertainty_at_level == "atomic":
                self.set_min_uncertainty(
                    uncertainty, force=reset_min_uncertainty)
                uncertainty = self.normalize_to_min_uncertainty(uncertainty)

        # If the order contains "local", then the uncertainty is calculated for the local environment
        if "local" in self.order:
            assert (
                self.nbr_key in results
            ), "Neighbor list must be provided for local uncertainty"

            uncertainty = self.get_local_uncertainty(
                atomic_uncertainty=uncertainty,
                num_atoms=num_atoms,
                nbr_list=results[self.nbr_key],
            )

            if self.set_min_uncertainty_at_level == "local":
                self.set_min_uncertainty(
                    uncertainty, force=reset_min_uncertainty)
                uncertainty = self.normalize_to_min_uncertainty(uncertainty)

        # If the order contains "system", then the uncertainty is calculated for the entire system
        if "system" in self.order:
            uncertainty = self.get_system_uncertainty(
                uncertainty=uncertainty, num_atoms=num_atoms
            ).squeeze()

            if self.set_min_uncertainty_at_level == "system":
                self.set_min_uncertainty(
                    uncertainty, force=reset_min_uncertainty)
                uncertainty = self.normalize_to_min_uncertainty(uncertainty)

        # Calibrate the uncertainty using Conformal Prediction if calibrate is True
        if self.calibrate:
            uncertainty = self.calibrate_uncertainty(uncertainty)

        return uncertainty


class EnsembleUncertainty(Uncertainty):
    """
    Ensemble uncertainty estimation using the variance or standard deviation of the
    predictions from the model ensemble.
    """

    def __init__(
        self,
        quantity: str,
        order: str,
        std_or_var: str = "var",
        min_uncertainty: Union[float, None] = None,
        set_min_uncertainty_at_level: str = "system",
        orig_unit: Union[str, None] = None,
        targ_unit: Union[str, None] = None,
        nbr_key: str = "nbr_list",
        calibrate: bool = False,
        cp_alpha: Union[float, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            min_uncertainty=min_uncertainty,
            set_min_uncertainty_at_level=set_min_uncertainty_at_level,
            nbr_key=nbr_key,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            *args,
            **kwargs,
        )
        assert std_or_var in [
            "std",
            "var",
        ], f"{std_or_var} not implemented. Choices only include 'std' or 'var'"
        self.q = quantity
        self.orig_unit = orig_unit
        self.targ_unit = targ_unit
        self.std_or_var = std_or_var

    def convert_units(
        self, value: Union[float, np.ndarray], orig_unit: str, targ_unit: str
    ):
        """
        Convert the energy/forces units of the value from orig_unit to targ_unit.
        """
        if orig_unit == targ_unit:
            return value

        conversion = {
            "eV": {"kcal/mol": 23.0605, "kJ/mol": 96.485},
            "kcal/mol": {"eV": 0.0433641, "kJ/mol": 4.184},
            "kJ/mol": {"eV": 0.0103641, "kcal/mol": 0.239006},
        }

        converted_val = value * conversion[orig_unit][targ_unit]
        return converted_val

    def get_energy_uncertainty(
        self,
        results: dict,
    ):
        """
        Get the uncertainty for the energy.
        """
        if self.orig_unit is not None and self.targ_unit is not None:
            results[self.q] = self.convert_units(
                results[self.q], orig_unit=self.orig_unit, targ_unit=self.targ_unit
            )

        if self.std_or_var == "std":
            val = results[self.q].std(-1)
        elif self.std_or_var == "var":
            val = results[self.q].var(-1)

        return val

    def get_forces_uncertainty(
        self,
        results: dict,
        num_atoms: List[int],
        *args,
        **kwargs,
    ):
        """
        Get the uncertainty for the forces.
        """
        if self.orig_unit is not None and self.targ_unit is not None:
            results[self.q] = self.convert_units(
                results[self.q], orig_unit=self.orig_unit, targ_unit=self.targ_unit
            )

        splits = torch.split(results[self.q], list(num_atoms))
        stack_split = torch.stack(splits, dim=0)

        if self.std_or_var == "std":
            val = stack_split.std(-1)
        elif self.std_or_var == "var":
            val = stack_split.var(-1)

        val = torch.norm(val, dim=-1)

        return val

    def get_atomic_uncertainty(self, results: dict, *args, **kwargs):
        return self.get_forces_uncertainty(results=results, *args, **kwargs)

    def get_uncertainty(
        self,
        results: dict,
        num_atoms: Union[List[int], None] = None,
        reset_min_uncertainty: bool = False,
        *args,
        **kwargs,
    ):
        if self.q == "energy":
            val = self.get_energy_uncertainty(results=results)
            val = self.normalize_to_min_uncertainty(val)

        elif self.q in ["energy_grad", "forces"]:
            val = self.get_atomic_uncertainty(
                results=results,
                num_atoms=num_atoms,
                reset_min_uncertainty=reset_min_uncertainty,
            )
        else:
            raise TypeError(f"{self.q} not yet implemented")

        if self.calibrate:
            val = self.calibrate_uncertainty(val)

        return val


class EvidentialUncertainty(Uncertainty):
    """
    Evidential Uncertainty estimation using the Evidential Deep Learning framework.
    """

    def __init__(
        self,
        order: str = "atomic",
        shared_v: bool = False,
        source: str = "epistemic",
        calibrate: bool = False,
        cp_alpha: Union[float, None] = None,
        min_uncertainty: float = None,
        set_min_uncertainty_at_level: str = "system",
        nbr_key: str = "nbr_list",
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            min_uncertainty=min_uncertainty,
            set_min_uncertainty_at_level=set_min_uncertainty_at_level,
            nbr_key=nbr_key,
            *args,
            **kwargs,
        )
        self.shared_v = shared_v
        self.source = source

    def check_params(
        self, results: dict, num_atoms=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check if the parameters are present in the results, if the shapes are
        correct. If the order is "atomic" and shared_v is True, then the v
        parameter is averaged over the systems.
        """
        v = results["v"].squeeze()
        alpha = results["alpha"].squeeze()
        beta = results["beta"].squeeze()
        assert (
            alpha.shape == beta.shape
        ), f"Shape of alpha {alpha.shape} and beta {beta.shape} do not match"

        num_systems = len(num_atoms)
        total_atoms = torch.sum(num_atoms)
        if self.order == "atomic" and self.shared_v:
            assert v.shape[0] == num_systems and alpha.shape[0] == total_atoms
            v = torch.split(v, list(num_atoms))
            v = torch.stack(v, dim=0)
            v = v.mean(-1, keepdims=True)
            v = v.repeat_interleave(num_atoms)

        return v, alpha, beta

    def get_atomic_uncertainty(
        self, results: dict, num_atoms: Union[List[int], None] = None, *args, **kwargs
    ) -> torch.Tensor:
        v, alpha, beta = self.check_params(
            results=results, num_atoms=num_atoms)

        if self.source == "aleatoric":
            uncertainty = beta / (alpha - 1)
        elif self.source == "epistemic":
            uncertainty = beta / (v * (alpha - 1))
        else:
            raise TypeError(f"{self.source} not implemented")

        # Reshape the uncertainty to (num_systems, num_atoms)
        splits = torch.split(uncertainty, list(num_atoms))
        uncertainty = torch.stack(splits, dim=0)

        return uncertainty


class MVEUncertainty(Uncertainty):
    """
    Mean Variance Estimation (MVE) based uncertainty estimation.
    """

    def __init__(
        self,
        variance_key: str = "var",
        quantity: str = "forces",
        order: str = "atomic",
        min_uncertainty: float = None,
        set_min_uncertainty_at_level: str = "system",
        nbr_key: str = "nbr_list",
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            min_uncertainty=min_uncertainty,
            set_min_uncertainty_at_level=set_min_uncertainty_at_level,
            nbr_key=nbr_key,
            *args,
            **kwargs,
        )
        self.vkey = variance_key
        self.q = quantity

    def get_atomic_uncertainty(
        self, results: dict, num_atoms: Union[List[int], None] = None, *args, **kwargs
    ) -> torch.Tensor:
        var = results[self.vkey].squeeze()
        assert results[self.q].shape[0] == var.shape[0]

        if self.q == "energy":
            pass

        else:
            splits = torch.split(var, list(num_atoms))
            var = torch.stack(splits, dim=0)

        return var

    def get_uncertainty(
        self,
        results: dict,
        num_atoms: Union[List[int], None] = None,
        reset_min_uncertainty: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        var = self.get_atomic_uncertainty(
            results=results, num_atoms=num_atoms, *args, **kwargs
        )

        if self.q == "energy":
            var = self.normalize_to_min_uncertainty(var)

        else:
            var = super().get_uncertainty(
                results=results,
                num_atoms=num_atoms,
                reset_min_uncertainty=reset_min_uncertainty,
                *args,
                **kwargs,
            )

        if self.calibrate:
            var = self.calibrate_uncertainty(var)

        return var


class GMMUncertainty(Uncertainty):
    """
    Gaussian Mixture Model (GMM) based uncertainty estimation.

    Args:
        train_embed_key (str): Key for the training embedding in the results dictionary
        test_embed_key (str): Key for the test embedding in the results dictionary
        n_clusters (int, dict): Number of clusters for the GMM model. If dict,
            then the number of clusters for each atomic number. If n_clusters
            is a dict, then the atomic_numbers key must be present in the results
            and gm_model will be a dict with keys as atomic numbers.
        order (str): Order of the uncertainty estimation (atomic, local, system)
        covariance_type (str): Type of covariance matrix for the GMM model
        tol (float): Tolerance for convergence of the GMM model
        max_iter (int): Maximum number of iterations for the GMM model
        n_init (int): Number of initializations for the GMM model
        init_params (str): Initialization method for the GMM model
        verbose (int): Verbosity level for the GMM model
        device (str): Device for the GMM model
        calibrate (bool): Calibrate the uncertainty using Conformal Prediction
        cp_alpha (float): Significance level for the Conformal Prediction model
        min_uncertainty (float): Minimum uncertainty value
        gmm_path (str): Path to the saved GMM model
        nbr_key (str): Key for the neighbor list in the results dictionary
    """

    def __init__(
        self,
        train_embed_key: str = "train_embedding",
        test_embed_key: str = "embedding",
        n_clusters: Union[int, dict] = 5,
        order: str = "atomic",
        covariance_type: str = "full",
        tol: float = 1e-3,
        max_iter: int = 100000,
        n_init: int = 1,
        init_params: str = "kmeans",
        verbose: int = 0,
        device: str = "cuda",
        calibrate: bool = False,
        cp_alpha: Union[float, None] = None,
        min_uncertainty: Union[float, None] = None,
        set_min_uncertainty_at_level: str = "system",
        gmm_path: Union[str, None] = None,
        nbr_key: str = "nbr_list",
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            min_uncertainty=min_uncertainty,
            set_min_uncertainty_at_level=set_min_uncertainty_at_level,
            device=device,
            *args,
            **kwargs,
        )
        self.train_key = train_embed_key
        self.test_key = test_embed_key
        self.n_clusters = n_clusters
        self.covar_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.verbose = verbose
        self.nbr_key = nbr_key
        self.gmm_path = gmm_path

        if gmm_path is not None and os.path.exists(gmm_path):
            import pickle

            with open(gmm_path, "rb") as f:
                self.gm_model = pickle.load(f)

            # Set the GMM parameters if the model is loaded
            self._set_gmm_params()

    def fit_gmm(
        self, Xtrain: torch.Tensor, atomic_numbers: Union[None, torch.Tensor] = None
    ) -> None:
        """
        Fit the GMM model to the embedding of training data.
        """
        self.Xtrain = Xtrain
        if isinstance(self.n_clusters, int):
            self.gm_model = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type=self.covar_type,
                tol=self.tol,
                max_iter=self.max_iter,
                n_init=self.n_init,
                init_params=self.init_params,
                verbose=self.verbose,
            )
            self.gm_model.fit(self.Xtrain.squeeze().cpu().numpy())

        elif isinstance(self.n_clusters, dict):
            assert (
                atomic_numbers is not None
            ), "Atomic numbers must be provided if n_clusters is a dict"

            self.gm_model = {}
            for i, (Z, n_clusters) in enumerate(self.n_clusters.items()):
                idx = atomic_numbers == Z
                Xtrain_Z = self.Xtrain[idx]
                gm_model = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type=self.covar_type,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    init_params=self.init_params,
                    verbose=self.verbose,
                )
                gm_model.fit(Xtrain_Z.squeeze().cpu().numpy())
                self.gm_model[Z] = gm_model

        else:
            raise TypeError(
                f"n_clusters must be an int or a dict, not a {type(self.n_clusters)}"
            )

        # Save the fitted GMM model if gmm_path is specified
        if (
            getattr(self, "gmm_path", None) is not None
            and os.path.exists(getattr(self, "gmm_path", None)) is False
        ):
            import pickle

            with open(self.gmm_path, "wb") as f:
                pickle.dump(self.gm_model, f)

            print(f"Saved fitted GMM model to {self.gmm_path}")

        # Set the GMM parameters
        self._set_gmm_params()

    def is_fitted(self) -> bool:
        """
        Check if the GMM model is fitted.
        """
        return hasattr(self, "gm_model") is True

    def _check_tensor(
        self,
        X: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Check if the input is a tensor and convert to torch.Tensor if not.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X = X.squeeze().double().to(self.device)

        return X

    def _set_gmm_params(self) -> None:
        """
        Get the means, precisions_cholesky, and weights from the GMM model.
        """
        if getattr(self, "gm_model", None) is None:
            raise Exception("GMMUncertainty: GMM does not exist/is not fitted")

        if isinstance(self.gm_model, GaussianMixture):
            self.means = self._check_tensor(self.gm_model.means_)
            self.precisions_cholesky = self._check_tensor(
                self.gm_model.precisions_cholesky_
            )
            self.weights = self._check_tensor(self.gm_model.weights_)

        elif isinstance(self.gm_model, dict):
            self.means = {}
            self.precisions_cholesky = {}
            self.weights = {}
            for Z, gm_model in self.gm_model.items():
                self.means[Z] = self._check_tensor(gm_model.means_)
                self.precisions_cholesky[Z] = self._check_tensor(
                    gm_model.precisions_cholesky_
                )
                self.weights[Z] = self._check_tensor(gm_model.weights_)

        else:
            raise Exception("GMMUncertainty: GMM model not recognized")

    def estimate_log_prob(
        self, X: torch.Tensor, means: torch.Tensor, precisions_cholesky: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate the log probability of the given embedding.
        """
        X = self._check_tensor(X)

        n_samples, n_features = X.shape
        n_clusters, _ = means.shape

        log_det = torch.sum(
            torch.log(
                precisions_cholesky.reshape(
                    n_clusters, -1)[:, :: n_features + 1]
            ),
            dim=1,
        )

        log_prob = torch.empty((n_samples, n_clusters)).to(X.device)
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
            y = torch.matmul(X, prec_chol) - \
                (mu.reshape(1, -1) @ prec_chol).squeeze()
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        log2pi = torch.log(torch.tensor([2 * torch.pi])).to(X.device)
        return -0.5 * (n_features * log2pi + log_prob) + log_det

    def estimate_weighted_log_prob(
        self,
        X: torch.Tensor,
        means: torch.Tensor,
        precisions_cholesky: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the weighted log probability of the given embedding.
        """
        log_prob = self.estimate_log_prob(X, means, precisions_cholesky)
        log_weights = torch.log(weights)
        weighted_log_prob = log_prob + log_weights

        return weighted_log_prob

    def log_likelihood(
        self,
        X: torch.Tensor,
        means: torch.Tensor,
        precisions_cholesky: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log likelihood of the embedding under the GMM model.
        """
        weighted_log_prob = self.estimate_weighted_log_prob(
            X, means, precisions_cholesky, weights
        )

        weighted_log_prob_max = weighted_log_prob.max(dim=1).values
        # logsumexp is numerically unstable for big arguments
        # below, the calculation below makes it stable
        # log(sum_i(a_i)) = log(exp(a_max) * sum_i(exp(a_i - a_max))) = a_max + log(sum_i(exp(a_i - a_max)))
        wlp_stable = weighted_log_prob - weighted_log_prob_max.reshape(-1, 1)
        logsumexp = weighted_log_prob_max + torch.log(
            torch.sum(torch.exp(wlp_stable), dim=1)
        )

        return logsumexp

    def probability(
        self,
        X: torch.Tensor,
        means: torch.Tensor,
        precisions_cholesky: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Probability of the embedding under the GMM model.
        """
        logP = self.log_likelihood(X, means, precisions_cholesky, weights)

        return torch.exp(logP)

    def negative_log_likelihood(
        self,
        X: torch.Tensor,
        atomic_numbers: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Negative log likelihood of the embedding under the GMM model.
        """
        if isinstance(self.gm_model, GaussianMixture):
            neglogP = -self.log_likelihood(
                X, self.means, self.precisions_cholesky, self.weights
            )

        else:
            assert (
                atomic_numbers is not None
            ), "Atomic numbers must be provided if gm_model is a dict"

            neglogP = torch.zeros(
                len(atomic_numbers), dtype=torch.float64, device=self.device
            )
            for Z, gm_model in self.gm_model.items():
                idx = np.where(atomic_numbers == Z)[0]
                X_Z = X[idx]
                logP_Z = self.log_likelihood(
                    X_Z, self.means[Z], self.precisions_cholesky[Z], self.weights[Z]
                )
                neglogP[idx] = -logP_Z

                self.save_indices_of_atom_types(key=Z, indices=idx)

        return neglogP

    def get_atomic_uncertainty(
        self,
        results: dict,
        num_atoms: Union[List[int], None] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get the uncertainty from the GMM model for the test embedding.
        """

        # Check if the GMM model is fitted and fit if not
        if self.is_fitted() is False:
            train_embedding = self._check_tensor(results[self.train_key])
            train_atomic_numbers = results.get("train_atomic_numbers", None)
            self.fit_gmm(train_embedding, train_atomic_numbers)

        # Get negative log likelihood for the test embedding
        test_embedding = self._check_tensor(results[self.test_key])
        test_atomic_numbers = results.get("test_atomic_numbers", None)
        uncertainty = self.negative_log_likelihood(
            test_embedding, test_atomic_numbers)

        # Reshape the uncertainty to (num_systems, num_atoms)
        splits = torch.split(uncertainty, list(num_atoms))
        uncertainty = torch.stack(splits, dim=0)

        return uncertainty
