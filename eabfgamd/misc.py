import os
import shutil
import time
import warnings
from typing import List, Tuple, Union

import numpy as np
import yaml
from ase import Atoms
from ase.io import read
from pymatgen.core import Structure

__all__ = [
    "get_logger",
    "load_config",
    "save_config",
    "make_dir",
    "make_and_backup_dir",
    "get_least_used_device",
    "get_main_model_outdir",
    "load_from_xyz",
    "ase_to_pmg",
    "is_energy_valid",
    "is_geom_valid",
    "convert_0_2pi_to_negpi_pi",
    "convert_negpi_pi_to_0_2pi",
    "get_neighbor_list",
    "find_angles_from_bonds",
    "get_bond_distances",
    "get_bond_angles",
    "get_bond_dihedrals",
    "get_dihedrals",
    "featurize_bond_info",
    "get_atoms",
    "attach_data_to_atoms",
    "get_atoms_with_data",
    "convert_dset_to_atoms_list",
    "write_atoms_list_to_xyzfile",
]


def get_logger(prefix):
    logger = lambda msg: print(f"{time.asctime()}|{prefix}|{msg}")
    return logger


def load_config(path: str) -> dict:
    """
    Load config file
    :param path: path to config file
    :return dict: config
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        f.close()
    return config


def save_config(config: dict, path: str, **kwargs) -> None:
    """
    Save config file
    :param config: config
    :param path: path to config file
    """
    with open(path, "w") as f:
        yaml.dump(config, f, **kwargs)
        f.close()


def make_dir(path: str) -> str:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def make_and_backup_dir(directory: str) -> None:
    """
    Make a directory and backup if it already exists
    :param directory: directory to make
    """
    if os.path.exists(directory):
        if os.path.exists(f"{directory}_backup"):
            shutil.rmtree(f"{directory}_backup")
        os.rename(directory, f"{directory}_backup")
    os.makedirs(directory)


def get_least_used_device() -> str:
    """
    Get least used cuda device
    :return: cuda device
    """
    import torch

    torch.cuda.init()
    free_memory = [
        torch.cuda.mem_get_info(device_id)
        for device_id in range(torch.cuda.device_count())
    ]
    free_memory = [free / total for free, total in free_memory]
    free_memory = [0.0 if i > 1.0 else i for i in free_memory]
    least_used_idx = np.argmax(free_memory).item()
    return f"cuda:{least_used_idx}"


def get_main_model_outdir(outdir: Union[str, list]) -> str:
    """
    Get the main model outdir since now the model outdir can be a list
    :param config: config
    :return: outdir
    """
    if isinstance(outdir, str):
        return outdir

    elif isinstance(outdir, list):
        if len(outdir) == 1:
            return outdir[0]
        else:
            common_prefix = os.path.commonprefix(outdir)
            if not common_prefix.endswith("/model/"):
                common_prefix = common_prefix[: common_prefix.rfind("/model/") + 6]
            return common_prefix
    else:
        raise ValueError("Outdir is not str or list")


def load_from_xyz(path: str) -> List[Atoms]:
    """
    Load ASE atoms object from xyz file
    :param path: path to xyz file
    :return: list of ASE atoms objects
    """
    atoms = read(path, index=":", format="extxyz")

    for i, at in enumerate(atoms):
        at.info["energy"] = at.get_potential_energy()
        at.arrays["forces"] = at.get_forces()

    return atoms


def ase_to_pmg(atoms: Atoms) -> Structure:
    """
    Convert ASE atoms object to pymatgen structure object
    :param atoms: ASE atoms object
    :return: pymatgen structure object
    """
    return Structure(
        lattice=atoms.get_cell(),
        species=atoms.get_atomic_numbers(),
        coords=atoms.get_positions(),
        coords_are_cartesian=True,
    )


def is_energy_valid(
    energy: float,
    min_energy: float,
    max_energy: float,
) -> bool:
    """
    Check if the energy is valid
    Args:
        energy: energy
        max_energy: maximum energy (kcal/mol)
        min_energy: minimum energy (kcal/mol)
    Returns:
        True if valid, False otherwise
    """
    return (energy >= min_energy) and (energy < max_energy) and (~np.isnan(energy))


def is_geom_valid(
    dist_mat: Union[np.ndarray, None] = None,
    rmin: float = 1.2,
    rmax: Union[float, None] = None,
    nbr_list: Union[List, None] = None,
    atoms: Union[Atoms, None] = None,
) -> bool:
    """
    Check if the atoms object has valid interatomic distances
    Args:
        atoms: ase.Atoms object
        nbr_list: neighbor list
        rmin: minimum distance between atoms (angstrom)
        rmax: maximum distance between atoms (angstrom)
    Returns:
        True if valid, False otherwise
    """
    assert not (atoms is None and dist_mat is None), "Either atoms or dist_mat must be provided"

    # get the distance matrix
    if dist_mat is None:
        print("No distance matrix provided, calculating the distance matrix...")
        dist_mat = atoms.get_all_distances(mic=True)

    # if any two pairs of atoms are too close, return False
    triu_inds = np.triu_indices(dist_mat.shape[0], k=1)
    triu_dist = dist_mat[triu_inds]
    if np.any(triu_dist < rmin):
        return False

    # get distances only for nbr_list (connected bonds)
    # both nbr_list and rmax must be provided since this is a bond check for
    # molecules, does not make sense to check for all atoms
    if rmax is not None and nbr_list is None:
        warnings.warn("nbr_list is not provided, cannot check for rmax")

    elif nbr_list is not None and rmax is not None:
        distances = dist_mat[nbr_list[:, 0], nbr_list[:, 1]]
        # return False if there are any bond distance greater than rmax
        if np.any(distances >= rmax) is True:
            return False

    return True


def convert_0_2pi_to_negpi_pi(x: float) -> float:
    """
    Convert angle from 0 to 2pi to -pi to pi
    :param x: angle in degrees
    :return: angle in degrees
    """
    if (x >= -180) and (x < 180):
        return x
    elif (x >= -360) and (x < -180):
        return x + 360
    elif (x >= 180) and (x < 360):
        return x - 360
    else:
        return convert_0_2pi_to_negpi_pi(x % (np.sign(x) * 360))


def convert_negpi_pi_to_0_2pi(x: float) -> float:
    if x >= 0:
        return x % 360
    elif (x < 0) & (x >= -360):
        return x + 360
    else:
        return convert_negpi_pi_to_0_2pi(x % -360)


def get_neighbor_list(starting_atoms):
    from ase.neighborlist import build_neighbor_list

    neighborlist = build_neighbor_list(
        atoms=starting_atoms,
        self_interaction=False,
    )
    conn_matrix = neighborlist.get_connectivity_matrix().todense()
    nbr_list = np.stack(conn_matrix.nonzero(), axis=-1)
    return nbr_list


def find_angles_from_bonds(bonds: np.ndarray) -> np.ndarray:
    """
    Find indices of atoms involved in all possible angles from a list of bonds/nbr_list
    :param bonds: list of bonds/nbr_list (num_bonds x 2)
    :return: list of angles (num_angles x 3)
    """
    angles = []
    for i in range(bonds.shape[0]):
        for j in range(bonds.shape[0]):
            if i != j:
                # Check if they share an atom
                shared_atom = np.intersect1d(bonds[i], bonds[j])
                if len(shared_atom) == 1:
                    # Sort to ensure the unique representation of an angle
                    angle = np.sort(
                        [
                            bonds[i][bonds[i] != shared_atom[0]][0],
                            shared_atom[0],
                            bonds[j][bonds[j] != shared_atom[0]][0],
                        ]
                    )
                    # Add the angle if it's not a bond and not already added
                    if angle[0] != angle[2] and not any(
                        np.array_equal(angle, existing_angle)
                        for existing_angle in angles
                    ):
                        angles.append(angle)
    return np.array(angles)


def get_bond_distances(
    atoms: Atoms,
    nbr_list: np.array = None,
    periodic: bool = False,
) -> dict:
    dist_mat = atoms.get_all_distances(mic=periodic)

    if nbr_list is None:
        bond_dist = {}
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if j >= i:
                    continue
                bond_dist[(i, j)] = dist_mat[i, j]
    else:
        dist = dist_mat[nbr_list[:, 0], nbr_list[:, 1]]
        bond_dist = {(i[0], i[1]): d for i, d in zip(nbr_list, dist)}

    return bond_dist


def get_bond_angles(atoms: Atoms, angle_inds: np.array) -> dict:
    angles = atoms.get_angles(angle_inds)
    angles_dict = {(i[0], i[1], i[2]): a for i, a in zip(angle_inds, angles)}

    return angles_dict


def get_bond_dihedrals(
    atoms: Atoms,
    dihedral_inds: np.ndarray,
) -> dict:
    diheds = atoms.get_dihedrals(dihedral_inds)
    dihed_dict = {(i[0], i[1], i[2], i[3]): d for i, d in zip(dihedral_inds, diheds)}

    return dihed_dict


def get_dihedrals(atoms: Atoms, dihedral_inds: np.ndarray) -> List:
    dihed = atoms.get_dihedrals(dihedral_inds)
    dihed = [convert_0_2pi_to_negpi_pi(i) for i in dihed]
    return dihed


def featurize_bond_info(
    atoms: Atoms,
    nbr_list: np.array = None,
    angle_inds: np.array = None,
    dihedral_inds: np.array = None,
    periodic: bool = False,
) -> dict:
    bond_info = {}
    # get all bond distances
    bond_info.update(
        get_bond_distances(
            atoms=atoms,
            nbr_list=nbr_list,
            periodic=periodic,
        )
    )

    # get all bond angles
    if angle_inds is not None:
        bond_info.update(
            get_bond_angles(
                atoms=atoms,
                angle_inds=angle_inds,
            )
        )

    # get all dihedral angles
    if dihedral_inds is not None:
        bond_info.update(
            get_bond_dihedrals(
                atoms=atoms,
                dihedral_inds=dihedral_inds,
            )
        )

    return bond_info


def get_atoms(data: dict) -> Atoms:
    lattice = data.get("lattice", None)
    if lattice is not None:
        atoms = Atoms(
            positions=data["nxyz"].numpy()[:, 1:],
            numbers=data["nxyz"].numpy()[:, 0],
            cell=lattice.numpy(),
            pbc=True,
        )
    else:
        atoms = Atoms(
            positions=data["nxyz"].numpy()[:, 1:],
            numbers=data["nxyz"].numpy()[:, 0],
        )

    return atoms


def attach_data_to_atoms(
    atoms: Atoms, energy: float, forces: List[Tuple[float, float, float]], **kwargs
) -> Atoms:
    _at = atoms.copy()
    _at.set_array("forces", forces)
    _at.info["energy"] = energy

    for key in kwargs:
        _at.info[key] = kwargs[key]

    return _at


def get_atoms_with_data(data: dict) -> Atoms:
    atoms = get_atoms(data)
    identifier = data.get("identifier", None)
    dihedrals = data.get("dihedrals", None)
    atoms = attach_data_to_atoms(
        atoms,
        energy=data["energy"].numpy(),
        forces=-data["energy_grad"].numpy(),
        identifier=identifier,
        dihedrals=dihedrals,
    )
    return atoms


def write_atoms_list_to_xyzfile(
    atoms: Union[Atoms, List[Atoms]], filename: str
) -> None:
    """Creates an extended XYZ file from an ase Atoms.
    If you want to create a dataset, simply pass a list of atoms
    as `atoms`.
    """
    from ase.io import write

    write(filename, atoms, format="extxyz")


def convert_dset_to_atoms_list(dset: List[dict]) -> List[Atoms]:
    xyz = []
    for d in list(dset):
        if "dihedrals" in d.keys():
            d["dihedrals"] = d["dihedrals"].squeeze().numpy()

        at = get_atoms_with_data(d)
        xyz.append(at)

    return xyz
