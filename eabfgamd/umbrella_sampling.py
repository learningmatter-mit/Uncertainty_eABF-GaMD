import torch
import numpy as np
from typing import Union, Tuple
from ase import Atoms
from ase.calculators.calculator import all_changes

from .mace_calculators import MACECalculator
from .colvars import ColVar as CV


DEFAULT_PROPERTIES = [
    "energy",
    "forces",
    "energy_unbiased",
    "forces_unbiased",
    "cv_vals",
    "cv_invmass",
    "grad_length",
    "cv_grad_lengths",
    "cv_dot_PES",
]


def wrap_angle(
    values: Union[float, np.ndarray, torch.tensor],
) -> Union[float, np.ndarray, torch.tensor]:
    if hasattr(values, "__iter__"):
        while (values >= np.pi).any() or (values < -np.pi).any():
            values[values < -np.pi] += 2 * np.pi
            values[values >= np.pi] -= 2 * np.pi

    else:
        while values >= np.pi or values < -np.pi:
            if values < -np.pi:
                values += 2 * np.pi
            else:
                values -= 2 * np.pi

    return values


class MACEUmbrellaSampling(MACECalculator):
    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energy_unbiased",
        "forces_unbiased",
        "cv_vals",
        "cv_invmass",
        "grad_length",
        "cv_grad_lengths",
        "cv_dot_PES",
    ]

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        device="cpu",
        cvs: list[CV] = None,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="Qs",
        model_type="MACE",
        **kwargs,
    ):
        MACECalculator.__init__(
            self,
            model_path=model_path,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            model_type=model_type,
            **kwargs,
        )

        if cvs:
            self.the_cv = cvs
        else:
            self.the_cv = []
            for cv_def in cv_defs:
                self.the_cv.append(CV(cv_def["definition"]))

        self.num_cv = len(self.the_cv)
        self.cv_names = [cv_def.get("name", "i") for i, cv_def in enumerate(cv_defs)]

        self.ks = [cv_def["k"] for cv_def in cv_defs]
        self.centers = []
        for cv_def in cv_defs:
            if cv_def["definition"]["type"] in ["angle", "dihedral"]:
                self.centers.append(wrap_angle(cv_def["center"]))
            else:
                self.centers.append(cv_def["center"])

    def diff(
        self,
        a: Union[np.ndarray, torch.tensor, float],
        b: Union[np.ndarray, torch.tensor, float],
        cv: CV,
    ) -> Union[np.ndarray, torch.tensor, float]:
        diff = a - b

        if cv.type in ["angle", "dihedral"]:
            diff = wrap_angle(diff)

        return diff

    def step_bias(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bias_grad = np.zeros_like(grad_xi[0])
        bias_ener = 0.0

        for i in range(self.num_cv):
            dxi = self.diff(xi[i], self.centers[i], self.the_cv[i])
            bias_grad += self.ks[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ks[i] * dxi**2

        return bias_ener, bias_grad

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str] = None,
        system_changes=all_changes,
    ):
        MACECalculator.calculate(self, atoms)

        if properties is None:
            properties = DEFAULT_PROPERTIES

        # for backwards compatability
        self.implemented_properties = DEFAULT_PROPERTIES

        self.results["num_atoms"] = [len(atoms)]
        self.results["energy_grad"] = -self.results["forces"]

        # get model prediction
        model_energy = self.results["energy"]
        model_grad = self.results["energy_grad"]
        # remove dimension if needed
        if model_energy.ndim == 2:
            model_energy = model_energy.mean(-1)
        if model_grad.ndim == 3:
            model_grad = model_grad.mean(-1)

        inv_masses = 1.0 / atoms.get_masses()
        M_inv = np.diag(np.repeat(inv_masses, 3).flatten())

        cvs = np.zeros(shape=(self.num_cv, 1))
        cv_grads = np.zeros(
            shape=(
                self.num_cv,
                atoms.get_positions().shape[0],
                atoms.get_positions().shape[1],
            )
        )
        cv_grad_lens = np.zeros(shape=(self.num_cv, 1))
        cv_invmass = np.zeros(shape=(self.num_cv, 1))
        cv_dot_PES = np.zeros(shape=(self.num_cv, 1))
        for ii, cv in enumerate(self.the_cv):
            xi, xi_grad = cv(atoms)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.matmul(
                xi_grad.flatten(), np.matmul(M_inv, xi_grad.flatten())
            )
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        self.results = {
            "energy": energy.reshape(-1),
            "forces": -grad.reshape(-1, 3),
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
        }
