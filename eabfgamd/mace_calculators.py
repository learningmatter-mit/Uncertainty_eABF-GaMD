"""
NOTE: The code here for MACECalculator is copied from the original MACE github repo
(https://github.com/ACEsuit/mace/blob/main/mace/calculators/mace.py) and modified
to fit our application. Please refer to the original repo for original code.
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.stress import full_3x3_to_voigt_6_stress

# mace imports
from mace import data
from mace.modules.utils import extract_equivariant, extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from nff.train.uncertainty import (
    EnsembleUncertainty,
    EvidentialUncertainty,
    GMMUncertainty,
    MVEUncertainty,
)

from ala.colvars import ColVar as CV
from ala.ensemble import Ensemble

DEFAULT_CUTOFF = 5.0
DEFAULT_SKIN = 1.0
DEFAULT_PROPERTIES = [
    "energy",
    "forces",
    "stress",
    "energy_unbiased",
    "forces_unbiased",
    "cv_vals",
    "ext_pos",
    "cv_invmass",
    "grad_length",
    "cv_grad_lengths",
    "cv_dot_PES",
    "const_vals",
]
UNC_DICT = {
    "ensemble": EnsembleUncertainty,
    "evidential": EvidentialUncertainty,
    "mve": MVEUncertainty,
    "gmm": GMMUncertainty,
}

local_model_path = f"{os.getenv('PROJECTSDIR')}/mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model"


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """
    NOTE: This code is copied from the original MACE repo (https://github.com/ACEsuit/mace/blob/main/mace/calculators/mace.py)
    and modified slightly to fit our application.

    MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="Qs",
        model_type="MACE",
        cv_defs: Union[list[dict], None] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type

        if model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]
        elif model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]
        else:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE], {model_type} not supported"
            )

        self.model = torch.load(f=model_path).to(device)
        if model_path.endswith("ensemble") is False:
            self.model = Ensemble([self.model])

        self.r_max = float(self.model.r_max.cpu().item())

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.charges_key = charges_key
        model_dtype = get_model_dtype(self.model)
        if model_dtype != default_dtype:
            print(
                f"Changing default dtype to {model_dtype} to match model dtype, save a new version of the model to overload the type."
            )
            default_dtype = model_dtype
        torch_tools.set_default_dtype(default_dtype)
        for param in self.model.parameters():
            param.requires_grad = False

        if cv_defs is not None:
            self._init_cv(cv_defs)
            self.implemented_properties.append("cv_vals")

    def _init_cv(self, cv_defs: list[dict]):
        self.cv_defs = cv_defs
        self.the_cvs = []
        for i, cv_def in enumerate(self.cv_defs):
            if cv_def["definition"]["type"] == "uncertainty":
                self.cv_defs[i]["definition"]["model"] = self.model
                the_cv = CV(cv_def["definition"])
            else:
                the_cv = CV(cv_def["definition"])

            self.the_cvs.append(the_cv)

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        """
        Create tensors to store the results of the committee
        :param model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]
        :param num_models: int, number of models in the committee
        :return: tuple of torch tensors
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        return dict_of_tensors

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.model.atomic_energies_fn(batch["node_attrs"])
            compute_stress = True
        else:
            compute_stress = False

        batch_base = next(iter(data_loader)).to(self.device)
        ret_tensors = {}
        batch = batch_base.clone()
        out = self.model(batch.to_dict(), training=True, compute_stress=compute_stress)
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            ret_tensors["energy"] = out["energy"] * self.energy_units_to_eV
            ret_tensors["node_energy"] = (out["node_energy"].T - node_e0).T
            ret_tensors["forces"] = (
                out["forces"] * self.energy_units_to_eV / self.length_units_to_A
            )
            if out["stress"] is not None:
                ret_tensors["stress"] = (
                    out["stress"] * self.energy_units_to_eV / self.length_units_to_A**3
                )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            ret_tensors["dipole"] = out["dipole"]

        self.results = {}
        self.results["num_atoms"] = [len(atoms)]
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energy"], dim=-1).detach().cpu().numpy()
            )
            self.results["energy_var"] = (
                torch.var(ret_tensors["energy"], dim=-1).detach().cpu().numpy()
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"] - node_e0.reshape(-1, 1), dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=-1).detach().cpu().numpy()
            )
            self.results["forces_var"] = (
                torch.var(ret_tensors["forces"], dim=-1).detach().cpu().numpy()
            )
            self.results["embedding"] = [
                o.detach().cpu().numpy() for o in out["node_feats"]
            ]
            self.results["xyz"] = out["xyz"].detach().cpu().numpy()
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=-1).detach().cpu()
                )

        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=-1).detach().cpu().numpy()
            )

        if getattr(self, "the_cvs", None) is not None:
            cvs = np.zeros(shape=(len(self.the_cvs), 1))
            for ii, the_cv in enumerate(self.the_cvs):
                xi, _ = the_cv(atoms, pred=self.results, return_grad=False)
                cvs[ii] = xi

            self.results["cv_vals"] = cvs

    def get_descriptors(self, atoms=None, shift_variance="all", num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param shift_variance: if all or invariant or equivariant descriptors are return
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if num_layers == -1:
            num_layers = int(self.model.num_interactions)
        assert shift_variance in [
            "all",
            "invariant",
            "equivariant",
        ], "shift_variance must be one of ['all', 'invariant', 'equivariant']"

        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        descriptors = self.model(batch.to_dict())["node_feats"]
        if shift_variance == "invariant":
            irreps_out = self.model.products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_invariant(
                    d,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for d in descriptors
            ]
        elif shift_variance == "equivariant":
            irreps_out = self.model.products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_equivariant(
                    d,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for d in descriptors
            ]

        descriptors = [d.detach().cpu().numpy() for d in descriptors]

        return descriptors


def MACEMPCalculator(
    model: Union[str, Path] = None,
    device: str = "",
    default_dtype: str = "float32",
    dispersion: bool = False,
    dispersion_xc="pbe",
    dispersion_cutoff=40.0 * units.Bohr,
    **kwargs,
) -> MACECalculator:
    """
    NOTE: This code is copied from the original MACE repo (https://github.com/ACEsuit/mace/blob/main/mace/calculators/mace.py)
    and modified slightly to fit our application.

    Constructs a MACECalculator with a pretrained model based on the Materials Project (89 elements).
    The model is released under the MIT license. See https://github.com/ACEsuit/mace-mp for all models.
    Note:
        If you are using this function, please cite the relevant paper for the Materials Project,
        any paper associated with the MACE model, and also the following:
        - MACE-MP by Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena,
            Dávid P. Kovács, Janosh Riebesell, et al., 2023, arXiv:2401.00096
        - MACE-Universal by Yuan Chiang, 2023, Hugging Face, Revision e5ebd9b,
            DOI: 10.57967/hf/1202, URL: https://huggingface.co/cyrusyc/mace-universal
        - Matbench Discovery by Janosh Riebesell, Rhys EA Goodall, Philipp Benner, Yuan Chiang,
            Alpha A Lee, Anubhav Jain, Kristin A Persson, 2023, arXiv:2308.14920

    Args:
        model (str, optional): Path to the model. Defaults to None which first checks for
            a local model and then downloads the default model from figshare. Specify "small",
            "medium" or "large" to download a smaller or larger model from figshare.
        device (str, optional): Device to use for the model. Defaults to "cuda".
        default_dtype (str, optional): Default dtype for the model. Defaults to "float32".
        dispersion (bool, optional): Whether to use D3 dispersion corrections. Defaults to False.
        dispersion_xc (str, optional): Exchange-correlation functional for D3 dispersion corrections.
        dispersion_cutoff (float, optional): Cutoff radius in Bhor for D3 dispersion corrections.
        **kwargs: Passed to MACECalculator and TorchDFTD3Calculator.

    Returns:
        MACECalculator: trained on the MPtrj dataset (unless model otherwise specified).
    """
    if model in (None, "medium") and os.path.isfile(local_model_path):
        model = local_model_path
        print(
            f"Using local medium Materials Project MACE model for MACECalculator {model}"
        )
    elif model in (None, "small", "medium", "large") or str(model).startswith("https:"):
        try:
            urls = dict(
                # 2023-12-10-mace-128-L0_energy_epoch-249.model
                small="http://tinyurl.com/46jrkm3v",
                medium="http://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
                large="http://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
            )
            checkpoint_url = (
                urls.get(model, urls["medium"])
                if model in (None, "small", "medium", "large")
                else model
            )
            cache_dir = os.path.expanduser("~/.cache/mace")
            checkpoint_url_name = "".join(
                c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
            )
            cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
            if not os.path.isfile(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {checkpoint_url!r}")
                _, http_msg = urllib.request.urlretrieve(
                    checkpoint_url, cached_model_path
                )
                if "Content-Type: text/html" in http_msg:
                    raise RuntimeError(
                        f"Model download failed, please check the URL {checkpoint_url}"
                    )
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            msg = f"Using Materials Project MACE for MACECalculator with {model}"
            print(msg)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if default_dtype == "float64":
        print(
            "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
        )
    if default_dtype == "float32":
        print(
            "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
        )
    mace_calc = MACECalculator(
        model_path=model, device=device, default_dtype=default_dtype, **kwargs
    )
    if dispersion:
        gh_url = "https://github.com/pfnet-research/torch-dftd"
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError:
            raise RuntimeError(
                f"Please install torch-dftd to use dispersion corrections (see {gh_url})"
            )
        print(
            f"Using TorchDFTD3Calculator for D3 dispersion corrections (see {gh_url})"
        )
        dtype = torch.float32 if default_dtype == "float32" else torch.float64
        d3_calc = TorchDFTD3Calculator(
            device=device,
            damping="bj",
            dtype=dtype,
            xc=dispersion_xc,
            cutoff=dispersion_cutoff,
            **kwargs,
        )
    calc = mace_calc if not dispersion else SumCalculator([mace_calc, d3_calc])
    return calc


class MACEBiasedCalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="Qs",
        model_type="MACE",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type

        if model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]
        elif model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]
        else:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE], {model_type} not supported"
            )

        self.model = torch.load(f=model_path).to(device)
        self.r_max = float(self.model.r_max.cpu().item())

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.charges_key = charges_key
        model_dtype = get_model_dtype(self.model)
        if model_dtype != default_dtype:
            print(
                f"Changing default dtype to {model_dtype} to match model dtype, save a new version of the model to overload the type."
            )
            default_dtype = model_dtype
        torch_tools.set_default_dtype(default_dtype)
        for param in self.model.parameters():
            param.requires_grad = False

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        """
        Create tensors to store the results of the committee
        :param model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]
        :param num_models: int, number of models in the committee
        :return: tuple of torch tensors
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        return dict_of_tensors

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.model.atomic_energies_fn(batch["node_attrs"])
            compute_stress = True
        else:
            compute_stress = False

        batch_base = next(iter(data_loader)).to(self.device)
        ret_tensors = {}
        batch = batch_base.clone()
        out = self.model(batch.to_dict(), training=True, compute_stress=compute_stress)
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            ret_tensors["energy"] = out["energy"] * self.energy_units_to_eV
            ret_tensors["node_energy"] = (out["node_energy"].T - node_e0).T
            ret_tensors["forces"] = (
                out["forces"] * self.energy_units_to_eV / self.length_units_to_A
            )
            if out["stress"] is not None:
                ret_tensors["stress"] = (
                    out["stress"] * self.energy_units_to_eV / self.length_units_to_A**3
                )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            ret_tensors["dipole"] = out["dipole"]

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = ret_tensors["energy"]
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = torch.mean(
                ret_tensors["node_energy"] - node_e0.reshape(-1, 1), dim=-1
            )
            self.results["forces"] = ret_tensors["forces"]
            self.results["embedding"] = out["node_feats"]
            self.results["xyz"] = out["xyz"]
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=-1).detach().cpu()
                )

        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = torch.mean(ret_tensors["dipole"], dim=0)

        self.results["nbr_list"] = torch.LongTensor(atoms.nbr_list).to(self.device)

    def get_descriptors(self, atoms=None, shift_variance="all", num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param shift_variance: if all or invariant or equivariant descriptors are return
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if num_layers == -1:
            num_layers = int(self.model.num_interactions)
        assert shift_variance in [
            "all",
            "invariant",
            "equivariant",
        ], "shift_variance must be one of ['all', 'invariant', 'equivariant']"

        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        descriptors = self.model(batch.to_dict())["node_feats"]
        if shift_variance == "invariant":
            irreps_out = self.model.products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_invariant(
                    d,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for d in descriptors
            ]
        elif shift_variance == "equivariant":
            irreps_out = self.model.products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_equivariant(
                    d,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for d in descriptors
            ]

        descriptors = [d.detach().cpu().numpy() for d in descriptors]

        return descriptors


class BiasBase(MACEBiasedCalculator):
    """Basic Calculator class with neural force field

    Args:
        model: the deural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energy_unbiased",
        "forces_unbiased",
        "cv_vals",
        "ext_pos",
        "cv_invmass",
        "grad_length",
        "cv_grad_lengths",
        "cv_dot_PES",
        "const_vals",
    ]

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        equil_temp: float = 300.0,
        device="cpu",
        cvs: list[CV] = None,
        extra_constraints: list[dict] = None,
        energy_units_to_eV=0.0433641,
        length_units_to_A=1.0,
        default_dtype="float32",
        starting_atoms: Atoms = None,
        **kwargs,
    ):
        MACEBiasedCalculator.__init__(
            self,
            model_path=model_path,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key="Qs",
            model_type="MACE",
            **kwargs,
        )
        self.implemented_properties = DEFAULT_PROPERTIES
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        if cvs:
            self.the_cvs = cvs
        else:
            self.the_cvs = []
            for i, cv_def in enumerate(self.cv_defs):
                if cv_def["definition"]["type"] == "uncertainty":
                    self.cv_defs[i]["definition"]["model"] = self.model
                    the_cv = CV(cv_def["definition"])
                    cv, cv_grad = the_cv(atoms=starting_atoms)
                    self.cv_defs[i]["ext_pos"] = cv
                else:
                    the_cv = CV(cv_def["definition"])

                self.the_cvs.append(the_cv)

        self.equil_temp = equil_temp

        self.ext_coords = np.zeros(shape=(self.num_cv, 1))
        self.ext_masses = np.zeros(shape=(self.num_cv, 1))
        self.ext_forces = np.zeros(shape=(self.num_cv, 1))
        self.ext_vel = np.zeros(shape=(self.num_cv, 1))
        self.ext_binwidth = np.zeros(shape=(self.num_cv, 1))
        self.ext_k = np.zeros(shape=(self.num_cv,))
        self.ext_dt = 0.0

        self.ranges = np.zeros(shape=(self.num_cv, 2))
        self.margins = np.zeros(shape=(self.num_cv, 1))
        self.conf_k = np.zeros(shape=(self.num_cv, 1))

        for ii, cv in enumerate(self.cv_defs):
            if "range" in cv.keys():
                self.ext_coords[ii] = cv["range"][0]
                self.ranges[ii] = cv["range"]
            else:
                raise KeyError("range")

            if "margin" in cv.keys():
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv.keys():
                self.conf_k[ii] = cv["conf_k"]

            if "ext_k" in cv.keys():
                self.ext_k[ii] = cv["ext_k"]
            elif "ext_sigma" in cv.keys():
                self.ext_k[ii] = (units.kB * self.equil_temp) / (
                    cv["ext_sigma"] * cv["ext_sigma"]
                )
            else:
                raise KeyError("ext_k/ext_sigma")

            if "type" not in cv.keys():
                self.cv_defs[ii]["type"] = "not_angle"
            else:
                self.cv_defs[ii]["type"] = cv["type"]

        self.constraints = None
        self.num_const = 0
        if extra_constraints is not None:
            self.constraints = []
            for cv in extra_constraints:
                self.constraints.append({})

                self.constraints[-1]["func"] = CV(cv["definition"])

                self.constraints[-1]["pos"] = cv["pos"]
                if "k" in cv.keys():
                    self.constraints[-1]["k"] = cv["k"]
                elif "sigma" in cv.keys():
                    self.constraints[-1]["k"] = (units.kB * self.equil_temp) / (
                        cv["sigma"] * cv["sigma"]
                    )
                else:
                    raise KeyError("k/sigma")

                if "type" not in cv.keys():
                    self.constraints[-1]["type"] = "not_angle"
                else:
                    self.constraints[-1]["type"] = cv["type"]

            self.num_const = len(self.constraints)

    def _update_bias(self, xi: np.ndarray):
        pass

    def _propagate_ext(self):
        pass

    def _up_extvel(self):
        pass

    def _check_boundaries(self, xi: np.ndarray):
        in_bounds = (xi <= self.ranges[:, 1]).all() and (xi >= self.ranges[:, 0]).all()
        return in_bounds

    def diff(
        self, a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str
    ) -> Union[np.ndarray, float]:
        """get difference of elements of numbers or arrays
        in range(-inf, inf) if is_angle is False or in range(-pi, pi) if is_angle is True
        Args:
            a: number or array
            b: number or array
        Returns:
            diff: element-wise difference (a-b)
        """
        diff = a - b

        # wrap to range(-pi,pi) for angle
        if cv_type in ["angle", "dihedral"]:
            if hasattr(diff, "__iter__"):
                while (diff >= np.pi).any() or (diff < -np.pi).any():
                    diff[diff < -np.pi] += 2 * np.pi
                    diff[diff >= np.pi] -= 2 * np.pi

            else:
                while diff >= np.pi or diff < -np.pi:
                    if diff < -np.pi:
                        diff += 2 * np.pi
                    else:
                        diff -= 2 * np.pi

        return diff

    def step_bias(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of bias

        Args:
            curr_cv: current value of the cv
            cv_index: for multidimensional FES

        Returns:
            bias_ener: bias energy
            bias_grad: gradiant of the bias in CV space, needs to be dotted with the cv_gradient
        """

        self._propagate_ext()
        bias_ener, bias_grad = self._extended_dynamics(xi, grad_xi)

        self._update_bias(xi)
        self._up_extvel()

        return bias_ener, bias_grad

    def _extended_dynamics(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bias_grad = np.zeros_like(grad_xi[0])
        bias_ener = 0.0

        for i in range(self.num_cv):
            # harmonic coupling of extended coordinate to reaction coordinate
            dxi = self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_grad += self.ext_k[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ext_k[i] * dxi**2

            # harmonic walls for confinement to range of interest
            if self.ext_coords[i] > (self.ranges[i][1] + self.margins[i]):
                r = self.diff(
                    self.ranges[i][1] + self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r

            elif self.ext_coords[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(
                    self.ranges[i][0] - self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r

        return bias_ener, bias_grad

    def harmonic_constraint(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of additional harmonic constraint

        Args:
            xi: current value of constraint "CV"
            grad_xi: Cartesian gradient of these CVs

        Returns:
            constr_ener: constraint energy
            constr_grad: gradient of the constraint energy

        """

        constr_grad = np.zeros_like(grad_xi[0])
        constr_ener = 0.0

        for i in range(self.num_const):
            dxi = self.diff(
                xi[i], self.constraints[i]["pos"], self.constraints[i]["type"]
            )
            constr_grad += self.constraints[i]["k"] * dxi * grad_xi[i]
            constr_ener += 0.5 * self.constraints[i]["k"] * dxi**2

        return constr_ener, constr_grad

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given Atoms.

        Args:
        atoms (Atoms): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the List[Atoms]
            to calculate using the models created.
        properties: list of keywords that can be present in self.results. Note
            that the units of energy and energy_grad should be in kcal/mol and
            kcal/mol/A.
        system_changes (default from ase)
        """
        if properties is None:
            properties = DEFAULT_PROPERTIES

        # for backwards compatability
        if getattr(self, "implemented_properties", None) is None:
            self.implemented_properties = properties

        MACEBiasedCalculator.calculate(self, atoms)

        self.results["num_atoms"] = [len(atoms)]
        self.results["energy_grad"] = -self.results["forces"]

        # get model prediction
        model_energy = self.results["energy"].detach().cpu().numpy()
        model_grad = self.results["energy_grad"].detach().cpu().numpy()
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

        for ii, the_cv in enumerate(self.the_cvs):
            xi, xi_grad = the_cv(atoms, pred=self.results)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.matmul(
                xi_grad.flatten(), np.matmul(M_inv, xi_grad.flatten())
            )
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        self.results = {
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
            "stress": self.results["stress"].squeeze(),
            "nbr_list": self.results["nbr_list"],
        }

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(
                    self.num_const,
                    atoms.get_positions().shape[0],
                    atoms.get_positions().shape[1],
                )
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)

            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            grad += const_grad

        self.results.update(
            {
                "energy": energy.reshape(-1),
                "forces": -grad.reshape(-1, 3),
                "ext_pos": self.ext_coords,
            }
        )

        if self.constraints:
            self.results["const_vals"] = consts


class eABF(BiasBase):
    """extended-system Adaptive Biasing Force Calculator
       class with neural force field

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Lagevin dyn of extended system (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
    """

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        starting_atoms: Atoms,
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        device="cpu",
        energy_units_to_eV: float = 0.0433641,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float32",
        **kwargs,
    ):
        BiasBase.__init__(
            self,
            model_path=model_path,
            cv_defs=cv_defs,
            starting_atoms=starting_atoms,
            equil_temp=equil_temp,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            **kwargs,
        )

        self.ext_dt = dt * units.fs
        self.nfull = nfull

        for ii, cv in enumerate(self.cv_defs):
            if "bin_width" in cv.keys():
                self.ext_binwidth[ii] = cv["bin_width"]
            elif "ext_sigma" in cv.keys():
                self.ext_binwidth[ii] = cv["ext_sigma"]
            else:
                raise KeyError("bin_width")

            if "ext_pos" in cv.keys():
                # set initial position
                self.ext_coords[ii] = cv["ext_pos"]
            else:
                raise KeyError("ext_pos")

            if "ext_mass" in cv.keys():
                self.ext_masses[ii] = cv["ext_mass"]
            else:
                raise KeyError("ext_mass")

        # initialize extended system at target temp of MD simulation
        for i in range(self.num_cv):
            self.ext_vel[i] = np.random.randn() * np.sqrt(
                self.equil_temp * units.kB / self.ext_masses[i]
            )

        self.friction = friction_per_ps * 1.0e-3 / units.fs
        self.rand_push = np.sqrt(
            self.equil_temp
            * self.friction
            * self.ext_dt
            * units.kB
            / (2.0e0 * self.ext_masses)
        )
        self.prefac1 = 2.0 / (2.0 + self.friction * self.ext_dt)
        self.prefac2 = (2.0e0 - self.friction * self.ext_dt) / (
            2.0e0 + self.friction * self.ext_dt
        )

        # set up all grid accumulators for ABF
        self.nbins_per_dim = np.array([1 for i in range(self.num_cv)])
        self.grid = []
        for i in range(self.num_cv):
            self.nbins_per_dim[i] = int(
                np.ceil(
                    np.abs(self.ranges[i, 1] - self.ranges[i, 0]) / self.ext_binwidth[i]
                )
            )
            self.grid.append(
                np.arange(
                    self.ranges[i, 0] + self.ext_binwidth[i] / 2,
                    self.ranges[i, 1],
                    self.ext_binwidth[i],
                )
            )
        self.nbins = np.prod(self.nbins_per_dim)

        # accumulators and conditional averages
        self.bias = np.zeros((self.num_cv, *self.nbins_per_dim), dtype=float)
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)

        self.cv_crit = np.copy(self.bias)

        self.histogram = np.zeros(self.nbins_per_dim, dtype=float)
        self.ext_hist = np.zeros_like(self.histogram)

    def get_index(self, xi: np.ndarray) -> tuple:
        """get list of bin indices for current position of CVs or extended variables
        Args:
            xi (np.ndarray): Current value of collective variable
        Returns:
            bin_x (list):
        """
        bin_x = np.zeros(shape=xi.shape, dtype=np.int64)
        for i in range(self.num_cv):
            bin_x[i] = int(
                np.floor(np.abs(xi[i] - self.ranges[i, 0]) / self.ext_binwidth[i])
            )
        return tuple(bin_x.reshape(1, -1)[0])

    def _update_bias(self, xi: np.ndarray):
        if self._check_boundaries(self.ext_coords):
            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1

            # linear ramp function
            ramp = (
                1.0
                if self.ext_hist[bink] > self.nfull
                else self.ext_hist[bink] / self.nfull
            )

            for i in range(self.num_cv):
                # apply bias force on extended system
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i]
                    * self.diff(
                        xi[i], self.ext_coords[i], self.cv_defs[i]["type"]
                    ).item(),
                )
                self.ext_forces[i] -= ramp * self.bias[i][bink]

        """
        Not sure how this can be dumped/printed to work with the rest
        # xi-conditioned accumulators for CZAR
        if (xi <= self.ranges[:,1]).all() and
               (xi >= self.ranges[:,0]).all():

            bink = self.get_index(xi)
            self.histogram[bink] += 1

            for i in range(self.num_cv):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]],
                          self.cv_defs[i]['type'])
                self.correction_czar[i][bink] += self.ext_k[i] * dx
        """

    def _propagate_ext(self):
        self.ext_rand_gauss = np.random.randn(len(self.ext_vel), 1)

        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses
        self.ext_coords += self.prefac1 * self.ext_dt * self.ext_vel

        # wrap to range(-pi,pi) for angle
        for ii in range(self.num_cv):
            if self.cv_defs[ii]["type"] == "angle":
                if self.ext_coords[ii] > np.pi:
                    self.ext_coords[ii] -= 2 * np.pi
                elif self.ext_coords[ii] < -np.pi:
                    self.ext_coords[ii] += 2 * np.pi

    def _up_extvel(self):
        self.ext_vel *= self.prefac2
        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses


class aMDeABF(eABF):
    """Accelerated extended-system Adaptive Biasing Force Calculator
       class with neural force field

       Accelerated Molecular Dynamics

        see:
            aMD: Hamelberg et. al., J. Chem. Phys. 120, 11919 (2004); https://doi.org/10.1063/1.1755656
            GaMD: Miao et. al., J. Chem. Theory Comput. (2015); https://doi.org/10.1021/acs.jctc.5b00436
            SaMD: Zhao et. al., J. Phys. Chem. Lett. 14, 4, 1103 - 1112 (2023); https://doi.org/10.1021/acs.jpclett.2c03688

        Apply global boost potential to potential energy, that is independent of Collective Variables.

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        amd_parameter: acceleration parameter; SaMD, GaMD == sigma0; aMD == alpha
        init_step: initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps: equilibration steps, min, max and var of potential energy is still updated
                          force constant of coupling is calculated from previous steps
        amd_method: "aMD": apply accelerated MD
                    "GaMD_lower": use lower bound for GaMD boost
                    "GaMD_upper: use upper bound for GaMD boost
                    "SaMD: apply Sigmoid accelerated MD
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Lagevin dyn of extended system (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
    """

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        starting_atoms: Atoms,
        dt: float,
        friction_per_ps: float,
        amd_parameter: float,
        collect_pot_samples: bool,
        estimate_k: bool,
        apply_amd: bool,
        amd_method: str = "gamd_lower",
        samd_c0: float = 0.0001,
        equil_temp: float = 300.0,
        nfull: int = 100,
        device="cpu",
        energy_units_to_eV: float = 0.0433641,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float32",
        **kwargs,
    ):
        eABF.__init__(
            self,
            model_path=model_path,
            cv_defs=cv_defs,
            starting_atoms=starting_atoms,
            dt=dt,
            friction_per_ps=friction_per_ps,
            equil_temp=equil_temp,
            nfull=nfull,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            **kwargs,
        )

        self.amd_parameter = amd_parameter
        self.collect_pot_samples = collect_pot_samples
        self.estimate_k = estimate_k
        self.apply_amd = apply_amd

        self.amd_method = amd_method.lower()

        if self.amd_method == "amd":
            print(
                " >>> Warning: Please use GaMD or SaMD to obtain accurate free energy estimates!\n"
            )

        self.pot_count = 0
        self.pot_var = 1e-8
        self.pot_std = 1e-8
        self.pot_m2 = 1e-8
        self.pot_avg = 1e-8
        self.pot_min = +np.inf
        self.pot_max = -np.inf
        self.k0 = 1e-8
        self.k1 = 1e-8
        self.k = 1e-8
        self.E = 1e-8
        self.c0 = samd_c0
        self.c = 1 / self.c0 - 1

        self.amd_pot = 1e-8
        self.amd_pot_traj = []

        self.amd_c1 = np.zeros_like(self.histogram)
        self.amd_c2 = np.zeros_like(self.histogram)
        self.amd_m2 = np.zeros_like(self.histogram)
        self.amd_corr = np.zeros_like(self.histogram)

    def step_bias(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of bias

        Args:
            curr_cv: current value of the cv
            cv_index: for multidimensional FES

        Returns:
            bias_ener: bias energy
            bias_grad: gradiant of the bias in CV space, needs to be dotted with the cv_gradient
        """

        epot = self.results["energy_unbiased"].item()
        self.amd_forces = np.copy(self.results["forces_unbiased"])

        self._propagate_ext()
        bias_ener, bias_grad = self._extended_dynamics(xi, grad_xi)

        if self.collect_pot_samples is True:
            self._update_pot_distribution(epot)

        if self.estimate_k is True:
            self._calc_E_k0()

        self._update_bias(xi)

        if self.apply_amd is True:
            # apply amd boost potential only if U0 below bound
            if epot < self.E:
                boost_ener, boost_grad = self._apply_boost(epot)
            else:
                boost_ener, boost_grad = 0.0, 0.0 * self.amd_forces
            bias_ener += boost_ener
            bias_grad += boost_grad

            is_in_bounds = self._check_boundaries(xi)
            if is_in_bounds:
                bink = self.get_index(xi)
                (
                    self.amd_c1[bink],
                    self.amd_m2[bink],
                    self.amd_c2[bink],
                ) = welford_var(
                    self.histogram[bink],
                    self.amd_c1[bink],
                    self.amd_m2[bink],
                    boost_ener,
                )

        self._up_extvel()

        return bias_ener, bias_grad

    def _apply_boost(self, epot):
        """Apply boost potential to forces"""
        if self.amd_method == "amd":
            amd_pot = np.square(self.E - epot) / (self.parameter + (self.E - epot))
            boost_grad = (
                ((epot - self.E) * (epot - 2.0 * self.parameter - self.E))
                / np.square(epot - self.parameter - self.E)
            ) * self.amd_forces

        elif self.amd_method == "samd":
            amd_pot = self.amd_pot = (
                self.pot_max
                - epot
                - 1
                / self.k
                * np.log(
                    (self.c + np.exp(self.k * (self.pot_max - self.pot_min)))
                    / (self.c + np.exp(self.k * (epot - self.pot_min)))
                )
            )
            boost_grad = (
                -(
                    1.0
                    / (
                        np.exp(
                            -self.k * (epot - self.pot_min) + np.log((1 / self.c0) - 1)
                        )
                        + 1
                    )
                    - 1
                )
                * self.amd_forces
            )

        else:
            prefac = self.k0 / (self.pot_max - self.pot_min)
            amd_pot = 0.5 * prefac * np.power(self.E - epot, 2)
            boost_grad = prefac * (self.E - epot) * self.amd_forces

        return amd_pot, boost_grad

    def _update_pot_distribution(self, epot: float):
        """update min, max, avg, var and std of epot

        Args:
            epot: potential energy
        """
        self.pot_min = np.min([epot, self.pot_min])
        self.pot_max = np.max([epot, self.pot_max])
        self.pot_count += 1
        self.pot_avg, self.pot_m2, self.pot_var = welford_var(
            self.pot_count, self.pot_avg, self.pot_m2, epot
        )
        self.pot_std = np.sqrt(self.pot_var)

    def _calc_E_k0(self):
        """compute force constant for amd boost potential

        Args:
            epot: potential energy
        """
        if self.amd_method == "gamd_lower":
            self.E = self.pot_max
            ko = (self.amd_parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )

            self.k0 = np.min([1.0, ko])

        elif self.amd_method == "gamd_upper":
            ko = (1.0 - self.amd_parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_avg - self.pot_min)
            )
            if 0.0 < ko <= 1.0:
                self.k0 = ko
            else:
                self.k0 = 1.0
            self.E = self.pot_min + (self.pot_max - self.pot_min) / self.k0

        elif self.amd_method == "samd":
            ko = (self.amd_parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg)
            )

            self.k0 = np.min([1.0, ko])
            if (self.pot_std / self.amd_parameter) <= 1.0:
                self.k = self.k0
            else:
                self.k1 = np.max(
                    [
                        0,
                        (
                            np.log(self.c)
                            + np.log((self.pot_std) / (self.amd_parameter) - 1)
                        )
                        / (self.pot_avg - self.pot_min),
                    ]
                )
                self.k = np.max([self.k0, self.k1])

        elif self.amd_method == "amd":
            self.E = self.pot_max

        else:
            raise ValueError(f" >>> Error: unknown aMD method {self.amd_method}!")


class WTMeABF(eABF):
    """Well tempered MetaD extended-system Adaptive Biasing Force Calculator
       based on eABF class

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Lagevin dyn of extended system (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
        hill_height: unscaled height of the MetaD Gaussian hills in eV
        hill_drop_freq: #steps between depositing Gaussians
        well_tempered_temp: ficticious temperature for the well-tempered scaling
    """

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        starting_atoms: Atoms,
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        hill_height: float = 0.0,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 4000.0,
        device="cpu",
        energy_units_to_eV: float = 0.0433641,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float32",
        **kwargs,
    ):
        eABF.__init__(
            self,
            model_path=model_path,
            cv_defs=cv_defs,
            starting_atoms=starting_atoms,
            equil_temp=equil_temp,
            dt=dt,
            friction_per_ps=friction_per_ps,
            nfull=nfull,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            **kwargs,
        )

        self.hill_height = hill_height
        self.hill_drop_freq = hill_drop_freq
        self.hill_std = np.zeros(shape=(self.num_cv))
        self.hill_var = np.zeros(shape=(self.num_cv))
        self.well_tempered_temp = well_tempered_temp
        self.call_count = 0
        self.center = []

        for ii, cv in enumerate(self.cv_defs):
            if "hill_std" in cv.keys():
                self.hill_std[ii] = cv["hill_std"]
                self.hill_var[ii] = cv["hill_std"] * cv["hill_std"]
            else:
                raise KeyError("hill_std")

        # set up all grid for MetaD potential
        self.metapot = np.zeros_like(self.histogram)
        self.metaforce = np.zeros_like(self.bias)

    def _update_bias(self, xi: np.ndarray):
        mtd_forces = self.get_wtm_force(self.ext_coords)
        self.call_count += 1

        if self._check_boundaries(self.ext_coords):
            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1

            # linear ramp function
            ramp = (
                1.0
                if self.ext_hist[bink] > self.nfull
                else self.ext_hist[bink] / self.nfull
            )

            for i in range(self.num_cv):
                # apply bias force on extended system
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i]
                    * self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"]),
                )
                self.ext_forces[i] -= ramp * self.bias[i][bink] + mtd_forces[i]

    def get_wtm_force(self, xi: np.ndarray) -> np.ndarray:
        """compute well-tempered metadynamics bias force from superposition of gaussian hills
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        is_in_bounds = self._check_boundaries(xi)

        if (self.call_count % self.hill_drop_freq == 0) and is_in_bounds:
            self.center.append(np.copy(xi.reshape(-1)))

        if is_in_bounds and self.num_cv == 1:
            bias_force, _ = self._accumulate_wtm_force(xi)
        else:
            bias_force, _ = self._analytic_wtm_force(xi)

        return bias_force

    def _accumulate_wtm_force(self, xi: np.ndarray) -> Tuple[list, float]:
        """compute numerical WTM bias force from a grid
        Right now this works only for 1D CVs
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        bink = self.get_index(xi)
        if self.call_count % self.hill_drop_freq == 0:
            w = self.hill_height * np.exp(
                -self.metapot[bink] / (units.kB * self.well_tempered_temp)
            )

            dx = self.diff(self.grid[0], xi[0], self.cv_defs[0]["type"]).reshape(
                -1,
            )
            epot = w * np.exp(-(dx * dx) / (2.0 * self.hill_var[0]))
            self.metapot += epot
            self.metaforce[0] -= epot * dx / self.hill_var[0]

        return self.metaforce[:, bink], self.metapot[bink]

    def _analytic_wtm_force(self, xi: np.ndarray) -> Tuple[list, float]:
        """compute analytic WTM bias force from sum of gaussians hills
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        local_pot = 0.0
        bias_force = np.zeros(shape=(self.num_cv))

        # this should never be the case!
        if len(self.center) == 0:
            print(" >>> Warning: no metadynamics hills stored")
            return bias_force

        ind = np.ma.indices((len(self.center),))[0]
        ind = np.ma.masked_array(ind)

        dist_to_centers = []
        for ii in range(self.num_cv):
            dist_to_centers.append(
                self.diff(
                    xi[ii], np.asarray(self.center)[:, ii], self.cv_defs[ii]["type"]
                )
            )

        dist_to_centers = np.asarray(dist_to_centers)

        if self.num_cv > 1:
            ind[
                (abs(dist_to_centers) > 3 * self.hill_std.reshape(-1, 1)).all(axis=0)
            ] = np.ma.masked
        else:
            ind[
                (abs(dist_to_centers) > 3 * self.hill_std.reshape(-1, 1)).all(axis=0)
            ] = np.ma.masked

        # can get slow in long run, so only iterate over significant elements
        for i in np.nditer(ind.compressed(), flags=["zerosize_ok"]):
            w = self.hill_height * np.exp(
                -local_pot / (units.kB * self.well_tempered_temp)
            )

            epot = w * np.exp(
                -np.power(dist_to_centers[:, i] / self.hill_std, 2).sum() / 2.0
            )
            local_pot += epot
            bias_force -= epot * dist_to_centers[:, i] / self.hill_var

        return bias_force.reshape(-1, 1), local_pot


class AttractiveBias(MACEBiasedCalculator):
    """Biased Calculator that introduces an attractive term
       Designed to be used with UQ as CV

    Args:
        model: the deural force field model
        cv_def: list of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        gamma: coupling strength, regulates strength of attraction
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energy_unbiased",
        "forces_unbiased",
        "cv_vals",
        "ext_pos",
        "cv_invmass",
        "grad_length",
        "cv_grad_lengths",
        "cv_dot_PES",
        "const_vals",
    ]

    def __init__(
        self,
        model_path: str,
        cv_defs: list[dict],
        gamma: float = 1.0,
        device: str = "cpu",
        en_key: str = "energy",
        extra_constraints: Union[List[dict], None] = None,
        energy_units_to_eV: float = 0.0433641,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float32",
        starting_atoms: Union[Atoms, None] = None,
        **kwargs,
    ):
        MACEBiasedCalculator.__init__(
            self,
            model_path=model_path,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key="Qs",
            model_type="MACE",
            **kwargs,
        )

        self.implemented_properties = DEFAULT_PROPERTIES
        self.gamma = gamma
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cvs = []
        for i, cv_def in enumerate(self.cv_defs):
            if cv_def["definition"]["type"] in ["uncertainty", "gmm_bias"]:
                self.cv_defs[i]["definition"]["model"] = self.model
                the_cv = CV(cv_def["definition"])
                cv, cv_grad = the_cv(atoms=starting_atoms)
                self.cv_defs[i]["ext_pos"] = cv
            else:
                the_cv = CV(cv_def["definition"])

            self.the_cvs.append(the_cv)

        self.ext_coords = np.zeros(shape=(self.num_cv, 1))
        self.ranges = np.zeros(shape=(self.num_cv, 2))
        self.margins = np.zeros(shape=(self.num_cv, 1))
        self.conf_k = np.zeros(shape=(self.num_cv, 1))

        for ii, cv in enumerate(self.cv_defs):
            if "range" in cv.keys():
                self.ext_coords[ii] = cv["range"][0]
                self.ranges[ii] = cv["range"]
            else:
                raise KeyError("range")

            if "margin" in cv.keys():
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv.keys():
                self.conf_k[ii] = cv["conf_k"]

            if "type" not in cv.keys():
                self.cv_defs[ii]["type"] = "not_angle"
            else:
                self.cv_defs[ii]["type"] = cv["type"]

        self.constraints = None
        self.num_const = 0
        if extra_constraints is not None:
            self.constraints = []
            for cv in extra_constraints:
                self.constraints.append({})

                self.constraints[-1]["func"] = CV(cv["definition"])

                self.constraints[-1]["pos"] = cv["pos"]
                if "k" in cv.keys():
                    self.constraints[-1]["k"] = cv["k"]
                elif "sigma" in cv.keys():
                    self.constraints[-1]["k"] = (units.kB * self.equil_temp) / (
                        cv["sigma"] * cv["sigma"]
                    )
                else:
                    raise KeyError("k/sigma")

                if "type" not in cv.keys():
                    self.constraints[-1]["type"] = "not_angle"
                else:
                    self.constraints[-1]["type"] = cv["type"]

            self.num_const = len(self.constraints)

    def diff(
        self, a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str
    ) -> Union[np.ndarray, float]:
        """get difference of elements of numbers or arrays
        in range(-inf, inf) if is_angle is False or in range(-pi, pi) if is_angle is True
        Args:
            a: number or array
            b: number or array
        Returns:
            diff: element-wise difference (a-b)
        """
        diff = a - b

        # wrap to range(-pi,pi) for angle
        if cv_type in ["angle", "dihedral"]:
            if hasattr(diff, "__iter__"):
                while (diff >= np.pi).any() or (diff < -np.pi).any():
                    diff[diff < -np.pi] += 2 * np.pi
                    diff[diff >= np.pi] -= 2 * np.pi

            else:
                while diff >= np.pi or diff < -np.pi:
                    if diff < -np.pi:
                        diff += 2 * np.pi
                    else:
                        diff -= 2 * np.pi

        return diff

    def step_bias(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of bias

        Args:
            curr_cv: current value of the cv
            cv_index: for multidimensional FES

        Returns:
            bias_ener: bias energy
            bias_grad: gradiant of the bias in CV space, needs to be dotted with the cv_gradient
        """

        bias_grad = -(self.gamma * grad_xi).sum(axis=0)
        bias_ener = -(self.gamma * xi).sum()

        # harmonic walls for confinement to range of interest
        for i in range(self.num_cv):
            if xi[i] > (self.ranges[i][1] + self.margins[i]):
                r = self.diff(
                    self.ranges[i][1] + self.margins[i],
                    xi[i],
                    self.cv_defs[i]["type"],
                )
                bias_grad += self.conf_k[i] * r
                bias_ener += 0.5 * self.conf_k[i] * r**2

            elif xi[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(
                    self.ranges[i][0] - self.margins[i],
                    xi[i],
                    self.cv_defs[i]["type"],
                )
                bias_grad += self.conf_k[i] * r
                bias_ener += 0.5 * self.conf_k[i] * r**2

        return bias_ener, bias_grad

    def harmonic_constraint(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of additional harmonic constraint

        Args:
            xi: current value of constraint "CV"
            grad_xi: Cartesian gradient of these CVs

        Returns:
            constr_ener: constraint energy
            constr_grad: gradient of the constraint energy

        """

        constr_grad = np.zeros_like(grad_xi[0])
        constr_ener = 0.0

        for i in range(self.num_const):
            dxi = self.diff(
                xi[i], self.constraints[i]["pos"], self.constraints[i]["type"]
            )
            constr_grad += self.constraints[i]["k"] * dxi * grad_xi[i]
            constr_ener += 0.5 * self.constraints[i]["k"] * dxi**2

        return constr_ener, constr_grad

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given Atoms.

        Args:
        atoms (Atoms): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the List[Atoms]
            to calculate using the models created.
        properties: list of keywords that can be present in self.results
        system_changes (default from ase)
        """

        if properties is None:
            properties = DEFAULT_PROPERTIES

        # for backwards compatability
        if getattr(self, "implemented_properties", None) is None:
            self.implemented_properties = properties

        MACEBiasedCalculator.calculate(self, atoms)

        self.results["num_atoms"] = [len(atoms)]
        self.results["energy_grad"] = -self.results["forces"]

        # get model prediction
        model_energy = self.results["energy"].detach().cpu().numpy()
        model_grad = self.results["energy_grad"].detach().cpu().numpy()
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
        for ii, the_cv in enumerate(self.the_cvs):
            xi, xi_grad = the_cv(atoms, pred=self.results)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.einsum(
                "i,ii,i", xi_grad.flatten(), M_inv, xi_grad.flatten()
            )
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        self.results = {
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
        }

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(
                    self.num_const,
                    atoms.get_positions().shape[0],
                    atoms.get_positions().shape[1],
                )
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)

            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            grad += const_grad

        self.results.update(
            {
                "energy": energy.reshape(-1),
                "forces": -grad.reshape(-1, 3),
                "ext_pos": self.ext_coords,
            }
        )

        if self.constraints:
            self.results["const_vals"] = consts


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm
    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample
    Returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var
