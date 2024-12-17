import os
from typing import List, Union
import pandas as pd
import numpy as np
from .misc import load_from_xyz


__all__ = [
    "Paths",
    "Energy",
    "get_amber_topology_data_from_pdb",
    "get_opls_topology_data_from_pdb",
    "AmberTopologyInfo",
    "OplsTopologyInfo",
    "TopoInfo",
    "OpenMMTrajectory",
]


class Paths:
    if os.getenv("DATA") is None:
        raise ValueError("$DATA does not exist")
    else:
        base_dir = f"{os.getenv('DATA')}/projects/unc_eabf/ala"

    params_dir = os.path.join(base_dir, "params")
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    img_dir = os.path.join(base_dir, "images")
    processed_dir = os.path.join(base_dir, "processed_results")

    inbox_params_dir = os.path.join(params_dir, "inbox")
    completed_params_dir = os.path.join(params_dir, "completed")

    opls_dir = os.path.join(data_dir, "opls")
    amber_dir = os.path.join(data_dir, "amber")

    opls_ff_path = os.path.join(opls_dir, "ala_opls-aa.xml")
    amber_prmtop_path = os.path.join(amber_dir, "ala_amber.parm7")
    amber_inpcrd_path = os.path.join(amber_dir, "ala_amber.rst7")

    opls_pdb_path = os.path.join(opls_dir, "ala.pdb")
    amber_pdb_path = os.path.join(amber_dir, "ala.pdb")

    opls_gen0_path = os.path.join(data_dir, "opls", "gen0_80kcalmol.pth.tar")
    # deleted amber gen0 path since I want to choose from many

    @classmethod
    def gen0_path(cls, forcefield):
        if forcefield == "opls":
            return cls.opls_gen0_path
        elif forcefield == "amber":
            return cls.amber_gen0_path
        else:
            raise ValueError(f"Forcefield {forcefield} is not supported")

    @classmethod
    def pdb_path(cls, forcefield):
        if forcefield == "opls":
            return cls.opls_pdb_path
        elif forcefield == "amber":
            return cls.amber_pdb_path
        else:
            raise ValueError(f"Forcefield {forcefield} is not supported")


class Energy:
    min_opls_energy = -69.46824645996094  # kcal/mol
    min_amber_energy = 0.00000000000000  # kcal/mol

    @classmethod
    def scale_opls_energy(cls, energy):
        return energy - cls.min_opls_energy

    @classmethod
    def scale_amber_energy(cls, energy):
        return energy - cls.min_amber_energy

    @classmethod
    def scale_energy(cls, energy, forcefield):
        if forcefield == "opls":
            return cls.scale_opls_energy(energy)
        elif forcefield == "amber":
            return cls.scale_amber_energy(energy)
        else:
            raise ValueError(f"Forcefield {forcefield} is not supported")


def get_amber_topology_data_from_pdb() -> pd.DataFrame:
    """
    This is hardcoded to read from the same path everytime
    """
    data = pd.read_csv(
        "/mnt/data0/atan14/projects/unc_eabf/ala/data/amber/ala.pdb",
        sep="\s+",
        names=[
            "group",
            "id",
            "atom_label",
            "comp_id",
            "seq_id",
            "cartn_x",
            "cartn_y",
            "cartn_z",
            "occupancy",
            "formal_charge",
        ],
        skipfooter=2,
        dtype={
            "group": str,
            "id": int,
            "atom_label": str,
            "comp_id": str,
            "seq_id": int,
            "cartn_x": float,
            "cartn_y": float,
            "cartn_z": float,
            "occupancy": float,
            "formal_charge": float,
        },
        engine="python",
    )
    return data


def get_opls_topology_data_from_pdb() -> pd.DataFrame:
    """
    This is hardcoded to read from the same path everytime
    """
    data = pd.read_csv(
        "/mnt/data0/atan14/projects/unc_eabf/ala/data/opls/ala.pdb",
        sep="\s+",
        skiprows=2,
        names=[
            "group",
            "id",
            "atom_label",
            "comp_id",
            "seq_id",
            "cartn_x",
            "cartn_y",
            "cartn_z",
        ],
        skipfooter=23,
        engine="python",
        dtype={
            "group": str,
            "id": int,
            "atom_label": str,
            "comp_id": str,
            "seq_id": int,
            "cartn_x": float,
            "cartn_y": float,
            "cartn_z": float,
        },
    )
    return data


class AmberTopologyInfo:
    @classmethod
    def get_topology_info(cls) -> pd.DataFrame:
        rec = np.rec.array(
            [
                ("ATOM", 1, "H1", "ACE", 1, 2.0, 1.0, -0.0, 1.0, 0.0),
                ("ATOM", 2, "CH3", "ACE", 1, 2.0, 2.09, 0.0, 1.0, 0.0),
                ("ATOM", 3, "H2", "ACE", 1, 1.486, 2.454, 0.89, 1.0, 0.0),
                ("ATOM", 4, "H3", "ACE", 1, 1.486, 2.454, -0.89, 1.0, 0.0),
                ("ATOM", 5, "C", "ACE", 1, 3.427, 2.641, -0.0, 1.0, 0.0),
                ("ATOM", 6, "O", "ACE", 1, 4.391, 1.877, -0.0, 1.0, 0.0),
                ("ATOM", 7, "N", "ALA", 2, 3.555, 3.97, -0.0, 1.0, 0.0),
                ("ATOM", 8, "H", "ALA", 2, 2.733, 4.556, -0.0, 1.0, 0.0),
                ("ATOM", 9, "CA", "ALA", 2, 4.853, 4.614, -0.0, 1.0, 0.0),
                ("ATOM", 10, "HA", "ALA", 2, 5.408, 4.316, 0.89, 1.0, 0.0),
                ("ATOM", 11, "CB", "ALA", 2, 5.661, 4.221, -1.232, 1.0, 0.0),
                ("ATOM", 12, "HB1", "ALA", 2, 5.123, 4.521, -2.131, 1.0, 0.0),
                ("ATOM", 13, "HB2", "ALA", 2, 6.63, 4.719, -1.206, 1.0, 0.0),
                ("ATOM", 14, "HB3", "ALA", 2, 5.809, 3.141, -1.241, 1.0, 0.0),
                ("ATOM", 15, "C", "ALA", 2, 4.713, 6.129, 0.0, 1.0, 0.0),
                ("ATOM", 16, "O", "ALA", 2, 3.601, 6.653, 0.0, 1.0, 0.0),
                ("ATOM", 17, "N", "NME", 3, 5.846, 6.835, 0.0, 1.0, 0.0),
                ("ATOM", 18, "H", "NME", 3, 6.737, 6.359, -0.0, 1.0, 0.0),
                ("ATOM", 19, "C", "NME", 3, 5.846, 8.284, 0.0, 1.0, 0.0),
                ("ATOM", 20, "H1", "NME", 3, 4.819, 8.648, 0.0, 1.0, 0.0),
                ("ATOM", 21, "H2", "NME", 3, 6.36, 8.648, 0.89, 1.0, 0.0),
                ("ATOM", 22, "H3", "NME", 3, 6.36, 8.648, -0.89, 1.0, 0.0),
            ],
            dtype=[
                ("group", "O"),
                ("id", "<i8"),
                ("atom_label", "O"),
                ("comp_id", "O"),
                ("seq_id", "<i8"),
                ("cartn_x", "<f8"),
                ("cartn_y", "<f8"),
                ("cartn_z", "<f8"),
                ("occupancy", "<f8"),
                ("formal_charge", "<f8"),
            ],
        )

        topo_info = pd.DataFrame.from_records(rec)
        return topo_info

    @classmethod
    def get_atom_names(cls, indices: List[int] = range(22)) -> List[str]:
        atom_names = [
            "H1",
            "CH3",
            "H2",
            "H3",
            "C",
            "O",
            "N",
            "H",
            "CA",
            "HA",
            "CB",
            "HB1",
            "HB2",
            "HB3",
            "C",
            "O",
            "N",
            "H",
            "C",
            "H1",
            "H2",
            "H3",
        ]
        return [atom_names[i] for i in indices]

    @classmethod
    def get_atom_types(cls, indices: List[int] = range(22)) -> List[str]:
        atom_names = cls.get_atom_names(indices)
        return [atom_name[0] for atom_name in atom_names]

    @classmethod
    def get_nbr_list(cls) -> np.ndarray:
        nbr_list = np.array(
            [
                [0, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [4, 5],
                [4, 6],
                [6, 7],
                [6, 8],
                [8, 9],
                [8, 10],
                [8, 14],
                [10, 11],
                [10, 12],
                [10, 13],
                [14, 15],
                [14, 16],
                [16, 17],
                [16, 18],
                [18, 19],
                [18, 20],
                [18, 21],
            ]
        )
        return nbr_list

    @classmethod
    def get_angle_inds(cls) -> np.ndarray:
        angle_inds = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [1, 2, 3],
                [1, 2, 4],
                [1, 3, 4],
                [1, 4, 5],
                [1, 4, 6],
                [4, 5, 6],
                [4, 6, 7],
                [4, 6, 8],
                [6, 7, 8],
                [6, 8, 9],
                [6, 8, 10],
                [6, 8, 14],
                [8, 9, 10],
                [8, 9, 14],
                [8, 10, 14],
                [8, 10, 11],
                [8, 10, 12],
                [8, 10, 13],
                [8, 14, 15],
                [8, 14, 16],
                [10, 11, 12],
                [10, 11, 13],
                [10, 12, 13],
                [14, 15, 16],
                [14, 16, 17],
                [14, 16, 18],
                [16, 17, 18],
                [16, 18, 19],
                [16, 18, 20],
                [16, 18, 21],
                [18, 19, 20],
                [18, 19, 21],
                [18, 20, 21],
            ]
        )
        return angle_inds

    @classmethod
    def get_dihedral_info(cls) -> dict:
        dihedral_info = {
            "phi": [4, 6, 8, 14],  # phi [C(=O), Ca, N, C(=O)],
            "psi": [6, 8, 14, 16],  # psi [N, Ca, C(=O), N]
            "omega_1": [3, 4, 6, 8],  # omega1 [Ca, N, C(=O), Cw]
            "omega_2": [8, 14, 16, 18],  # omega2 [Ca, C(=O), N, Cw]
            "chi_1": [14, 8, 10, 11],  # chi1 [C(=O), Ca, Cb, H] #my own reference
            "chi_2": [6, 4, 3, 0],  # chi2 [N, C(=O), Cw, H] #my own reference
            "chi_3": [14, 16, 18, 19],  # chi3 [C(=O), N, Cw, H] #my own reference
        }

        return dihedral_info

    @classmethod
    def get_dihedral_inds(
        cls,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        dihedral_info = cls.get_dihedral_info()
        inds = np.array([dihedral_info[h] for h in dihedrals])
        return inds

    @classmethod
    def get_dihedral_names(cls) -> List:
        dihedral_info = cls.get_dihedral_info()
        return list(dihedral_info.keys())

    @classmethod
    def get_dihedral_names_for_matplotlib(cls) -> List:
        dihedral_info = cls.get_dihedral_info()
        return [f"$\\{k}$" for k in dihedral_info.keys()]

    @classmethod
    def get_dihedral_branch_info(cls) -> dict:
        branch_info = {
            "phi": [0, 1, 2, 3, 5, 7],
            "psi": [15, 17, 18, 19, 20, 21],
            "omega1": [0, 1, 2, 3, 5],
            "omega2": [17, 18, 19, 20, 21],
            "chi1": [12, 13],
            "chi2": [1, 2],
            "chi3": [20, 21],
        }
        return branch_info

    @classmethod
    def get_branch_inds(
        cls,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        branch_info = cls.get_dihedral_branch_info()
        inds = np.array([branch_info[h] for h in dihedrals])
        return inds


class OplsTopologyInfo:
    @classmethod
    def get_topology_info(cls) -> pd.DataFrame:
        rec = np.rec.array(
            [
                (0, "ATOM", 1, "C00", "UNK", 1, 1.0, 1.0, 0.0),
                (1, "ATOM", 2, "C01", "UNK", 1, -0.527, 1.0, 0.0),
                (2, "ATOM", 3, "C02", "UNK", 1, -1.016, 1.0, 1.463),
                (3, "ATOM", 4, "O03", "UNK", 1, -1.145, -0.033, 2.113),
                (4, "ATOM", 5, "N04", "UNK", 1, -1.161, 2.262, 2.01),
                (5, "ATOM", 6, "C05", "UNK", 1, -1.554, 2.475, 3.381),
                (6, "ATOM", 7, "N06", "UNK", 1, -1.076, 2.167, -0.706),
                (7, "ATOM", 8, "C07", "UNK", 1, -0.914, 2.31, -2.068),
                (8, "ATOM", 9, "O08", "UNK", 1, -0.267, 1.539, -2.765),
                (9, "ATOM", 10, "C09", "UNK", 1, -1.608, 3.516, -2.64),
                (10, "ATOM", 11, "H0A", "UNK", 1, 1.386, 0.124, 0.532),
                (11, "ATOM", 12, "H0B", "UNK", 1, 1.397, 1.892, 0.496),
                (12, "ATOM", 13, "H0C", "UNK", 1, 1.408, 0.976, -1.015),
                (13, "ATOM", 14, "H0D", "UNK", 1, -0.916, 0.102, -0.492),
                (14, "ATOM", 15, "H0E", "UNK", 1, -0.887, 3.059, 1.448),
                (15, "ATOM", 16, "H0F", "UNK", 1, -1.485, 1.557, 3.969),
                (16, "ATOM", 17, "H0G", "UNK", 1, -2.585, 2.837, 3.386),
                (17, "ATOM", 18, "H0H", "UNK", 1, -0.901, 3.239, 3.81),
                (18, "ATOM", 19, "H0I", "UNK", 1, -1.933, 2.55, -0.326),
                (19, "ATOM", 20, "H0J", "UNK", 1, -2.0, 4.168, -1.854),
                (20, "ATOM", 21, "H0K", "UNK", 1, -2.434, 3.187, -3.275),
                (21, "ATOM", 22, "H0M", "UNK", 1, -0.887, 4.088, -3.23),
            ],
            dtype=[
                ("index", "<i8"),
                ("group", "O"),
                ("id", "<i8"),
                ("atom_label", "O"),
                ("comp_id", "O"),
                ("seq_id", "<i8"),
                ("cartn_x", "<f8"),
                ("cartn_y", "<f8"),
                ("cartn_z", "<f8"),
            ],
        )
        topo_info = pd.DataFrame.from_records(rec)
        return topo_info

    @classmethod
    def get_atom_names(cls, indices: List[int] = range(22)) -> List[str]:
        atom_names = [
            "C00",
            "C01",
            "C02",
            "O03",
            "N04",
            "C05",
            "N06",
            "C07",
            "O08",
            "C09",
            "H0A",
            "H0B",
            "H0C",
            "H0D",
            "H0E",
            "H0F",
            "H0G",
            "H0H",
            "H0I",
            "H0J",
            "H0K",
            "H0M",
        ]
        return [atom_names[i] for i in indices]

    @classmethod
    def get_nbr_list(cls) -> np.ndarray:
        nbr_list = np.array(
            [
                [0, 1],
                [0, 10],
                [0, 11],
                [0, 12],
                [1, 2],
                [1, 6],
                [1, 13],
                [2, 3],
                [2, 4],
                [4, 5],
                [4, 14],
                [5, 15],
                [5, 16],
                [5, 17],
                [6, 7],
                [6, 18],
                [7, 8],
                [7, 9],
                [9, 19],
                [9, 20],
                [9, 21],
            ]
        )
        return nbr_list

    @classmethod
    def get_angle_inds(cls) -> np.ndarray:
        angle_inds = np.array(
            [
                [1, 0, 11],
                [6, 7, 9],
                [11, 0, 12],
                [2, 1, 6],
                [5, 4, 14],
                [1, 6, 18],
                [0, 1, 6],
                [3, 2, 4],
                [7, 9, 21],
                [19, 9, 21],
                [4, 5, 15],
                [10, 0, 11],
                [2, 4, 5],
                [2, 4, 14],
                [1, 0, 10],
                [6, 7, 8],
                [0, 1, 2],
                [15, 5, 17],
                [6, 1, 13],
                [1, 2, 4],
                [7, 6, 18],
                [7, 9, 20],
                [19, 9, 20],
                [4, 5, 17],
                [16, 5, 17],
                [1, 0, 12],
                [8, 7, 9],
                [1, 6, 7],
                [2, 1, 13],
                [15, 5, 16],
                [0, 1, 13],
                [1, 2, 3],
                [7, 9, 19],
                [4, 5, 16],
                [10, 0, 12],
                [20, 9, 21],
            ]
        )
        return angle_inds

    @classmethod
    def get_dihedral_info(cls) -> dict:
        # https://doi.org/10.1063/1.3574394
        # https://doi.org/10.1002/jcc.25589
        dihedral_info = {
            "phi": [2, 1, 6, 7],  # phi [C(=O), Ca, N, C(=O)],
            "psi": [6, 1, 2, 4],  # psi [N, Ca, C(=O), N]
            "omega_1": [1, 6, 7, 9],  # omega1 [Ca, N, C(=O), Cw]
            "omega_2": [1, 2, 4, 5],  # omega2 [Ca, C(=O), N, Cw]
            "chi_1": [2, 1, 0, 10],  # chi1 [C(=O), Ca, Cb, H] #my own reference
            "chi_2": [6, 7, 9, 19],  # chi2 [N, C(=O), Cw, H] #my own reference
            "chi_3": [2, 4, 5, 15],  # chi3 [C(=O), N, Cw, H] #my own reference
        }

        return dihedral_info

    @classmethod
    def get_dihedral_inds(
        cls,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        dihedral_info = cls.get_dihedral_info()
        inds = np.array([dihedral_info[h] for h in dihedrals])
        return inds

    @classmethod
    def get_dihedral_names(cls) -> List:
        dihedral_info = cls.get_dihedral_info()
        return list(dihedral_info.keys())

    @classmethod
    def get_dihedral_names_for_matplotlib(cls) -> List:
        dihedral_info = cls.get_dihedral_info()
        return [f"$\\{k}$" for k in dihedral_info.keys()]

    @classmethod
    def get_dihedral_branch_info(cls) -> dict:
        branch_info = {
            "phi": [7, 8, 9, 18, 19, 20, 21],
            "psi": [3, 4, 5, 14, 15, 16, 17],
            "omega1": [8, 9, 19, 20, 21],
            "omega2": [5, 14, 15, 16, 17],
            "chi1": [11, 12],
            "chi2": [20, 21],
            "chi3": [16, 17],
        }
        return branch_info

    @classmethod
    def get_branch_inds(
        cls,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        branch_info = cls.get_dihedral_branch_info()
        inds = np.array([branch_info[h] for h in dihedrals])
        return inds


class TopoInfo:
    @classmethod
    def get_forcefield(cls, ff: str) -> None:
        if ff == "opls":
            return OplsTopologyInfo
        elif ff == "amber":
            return AmberTopologyInfo
        else:
            raise ValueError(f"Forcefield {ff} not supported")

    @classmethod
    def get_topology_info(cls, ff: str) -> pd.DataFrame:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_topology_info()

    @classmethod
    def get_atom_names(cls, ff: str, indices: List[int] = range(22)) -> List[str]:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_atom_names(indices)

    @classmethod
    def get_nbr_list(cls, ff: Union[str, None]) -> Union[np.ndarray, None]:
        if ff is None:
            return None
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_nbr_list()

    @classmethod
    def get_angle_inds(cls, ff: str) -> np.ndarray:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_angle_inds()

    @classmethod
    def get_dihedral_info(cls, ff: str) -> dict:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_dihedral_info()

    @classmethod
    def get_dihedral_inds(
        cls,
        ff: str,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_dihedral_inds(dihedrals)

    @classmethod
    def get_dihedral_names(cls, ff: str) -> List:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_dihedral_names()

    @classmethod
    def get_dihedral_names_for_matplotlib(cls, ff: str) -> List:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_dihedral_names_for_matplotlib()

    @classmethod
    def get_dihedral_branch_info(cls, ff: str) -> dict:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_dihedral_branch_info()

    @classmethod
    def get_branch_inds(
        cls,
        ff: str,
        dihedrals: List = [
            "phi",
            "psi",
            "omega_1",
            "omega_2",
            "chi_1",
            "chi_2",
            "chi_3",
        ],
    ) -> np.ndarray:
        topoinfo = cls.get_forcefield(ff)
        return topoinfo.get_branch_inds(dihedrals)


class OpenMMTrajectory:
    def __init__(self, output_folder):
        """
        Args:
            pdbfile (str): path to pdb trajectory file
            datafile (str): path to state data file
            forcefile (str): path to force file

        Returns:
            trajectory (mdtraj.Trajectory): trajectory of loaded pdb file
            molecules (list of ase.Atoms): list of molecules from individual trajectories
            state_data (pandas.DataFrame): state data of production run
            forces (np.array): array containing forces of atoms
        """
        self.load_traj(os.path.join(output_folder, "trajectory.pdb"))
        self.load_state_data(os.path.join(output_folder, "statedata.txt"))
        self.load_force_data(os.path.join(output_folder, "forces.txt"))
        assert self.state_data.shape[0] == len(self.trajectory)

    def load_traj(self, pdbfile):
        """
        Helper function to __init__()
        Load trajectory from pdb file using mdtraj.

        Args:
            pdbfile (str): path to pdb trajectory file
        """
        self.trajectory = load_from_xyz(pdbfile)
        self.trajectory = self.trajectory[:-1]
        self.natoms = self.trajectory[0].get_global_number_of_atoms()

    def load_state_data(self, datafile):
        """
        Helper function to __init__()
        Load state data from production run.

        Args:
            datafile (str): path to state data file

        Returns:
            state_data (pandas.DataFrame): state data of production run
        """
        self.state_data = pd.read_csv(datafile, sep=",", header=0)

    def load_force_data(self, forcefile):
        """
        Helper function to __init__()
        Load forces of atoms from production run.

        Args:
            forcefile (str) : path to force data file
        """
        with open(forcefile, "r") as f:
            self.forces = f.read().split("\n")
            self.forces = [np.array(i.split()).astype(float) for i in self.forces][:-1]
            self.forces = np.stack(self.forces).reshape(-1, self.natoms, 3)
            f.close()

    def get_dihedrals(self, a0, a1, a2, a3, mic=True):
        """
        Get dihedral angle between four indexed atoms

        Args:
            a1, a2, a3, a4 (int) : indices of the four atoms
            mic (bool): whether minimum image convention is taken into account

        Returns:
            dihedrals (list): List of dihedral angles (degree) of the four atoms as values
        """
        dihedrals = [
            mol.get_dihedral(a0=a0, a1=a1, a2=a2, a3=a3, mic=mic)
            for mol in self.trajectory
        ]
        return dihedrals

    def get_angles(self, a1, a2, a3, mic=True):
        """
        Get angles between three indexes atoms

        Args:
            a1, a2, a3 (int) : indices of the three atoms
            mic (bool): whether minimum image convention is taken into account

        Returns:
            angles (list): List of angles (degree) of the three atoms as values
        """
        angles = [
            mol.get_angle(a1=a1, a2=a2, a3=a3, mic=mic) for mol in self.trajectory
        ]
        return angles
