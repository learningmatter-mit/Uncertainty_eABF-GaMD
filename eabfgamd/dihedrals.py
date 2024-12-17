import torch
import numpy as np
from typing import List, Tuple
from ase import Atoms
from nff.data import Dataset


__all__ = [
    "set_dihedrals",
    "get_geom_with_new_dihedrals",
]


DEVICE = "cpu"


def get_axis_vector(u1: torch.tensor, u2: torch.tensor):
    """
    Get the axis vector of the rotation in unit vector form
    Args:
        u1, u2: torch.tensor of shape (3,)
    Returns:
        u: torch.tensor of shape (3,), unit vector of the rotation axis
    """
    u = u2 - u1
    return u / ((u**2).sum() ** 0.5)


def get_rotation_matrix(th: torch.tensor, u: torch.tensor, device=DEVICE):
    """
    Get the rotation matrix
    Args:
        th: float, rotation angle in degrees
        u: torch.tensor of shape (3,), unit vector of the rotation axis
    Returns:
        R: torch.tensor of shape (3, 3), rotation matrix
    """
    # convert to radians
    th = th.to(device)
    theta = th * np.pi / 180
    # get rotation matrix
    R = torch.zeros(3, 3).to(device)
    R[0, 0] = torch.cos(theta) + u[0] ** 2 * (1 - torch.cos(theta))
    R[0, 1] = (u[0] * u[1] * (1 - torch.cos(theta))) - \
        (u[2] * torch.sin(theta))
    R[0, 2] = (u[0] * u[2] * (1 - torch.cos(theta))) + \
        (u[1] * torch.sin(theta))
    R[1, 0] = (u[0] * u[1] * (1 - torch.cos(theta))) + \
        (u[2] * torch.sin(theta))
    R[1, 1] = torch.cos(theta) + u[1] ** 2 * (1 - torch.cos(theta))
    R[1, 2] = (u[1] * u[2] * (1 - torch.cos(theta))) - \
        (u[0] * torch.sin(theta))
    R[2, 0] = (u[2] * u[0] * (1 - torch.cos(theta))) - \
        (u[1] * torch.sin(theta))
    R[2, 1] = (u[2] * u[1] * (1 - torch.cos(theta))) + \
        (u[0] * torch.sin(theta))
    R[2, 2] = torch.cos(theta) + u[2] ** 2 * (1 - torch.cos(theta))
    return R


def get_mask(n: int, mask_indices: List[int], device: str = DEVICE):
    """
    Mask matrix for rotating indicated atoms while keeping the rest fixed
    Args:
        n: int, number of atoms
        mask_indices: list of int, indices of atoms to be rotated
    Returns:
        mask: torch.tensor of shape (n, n), mask matrix
    """
    mask = torch.zeros((n, n)).to(device)
    for i in mask_indices:
        mask[i, i] += 1
    return mask


def get_rotated_xyz(
    theta: torch.tensor,
    u: torch.tensor,
    center: torch.tensor,
    xyz: torch.tensor,
    mask_indices: List[int],
    device: str = DEVICE,
) -> torch.tensor:
    """
    Args:
        theta: float, rotation angle in degrees
        u: torch.tensor of shape (3,), unit vector of the rotation axis
        center: torch.tensor of shape (3,), center of rotation
        xyz: torch.tensor of shape (n, 3), coordinates to be rotated
        mask_indices: list of int, indices of atoms to be rotated
    Returns:
        new_xyz: torch.tensor of shape (n, 3), rotated coordinates
    """
    n = len(xyz)
    mask = get_mask(n, mask_indices, device=device)

    xyz = xyz - center
    mask_xyz = mask @ xyz

    R = get_rotation_matrix(theta, u, device=device)
    new_xyz = (R @ mask_xyz.T).T
    new_xyz += ((torch.eye(n).to(device) - mask) @ xyz) + center

    return new_xyz


def set_dihedrals(
    nxyz: torch.tensor,
    rotations: dict,
    dihed_inds: dict,
    atoms_rotated: dict,
    device: str = DEVICE,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Set dihedral angles of a molecule to the amount of rotations specified
    Note:
        The rotation orders have to be such a way that the subsequent
        rotations do not alter the earlier rotations (heuristic: in decreasing
        order of numbers of atoms_rotated)
    Args:
        nxyz: torch.tensor of shape (n, 3) coordinates of the molecule
        rotations (dict): amount of rotation angles in degrees to be applied
        dihed_inds (dict): indices of atoms of specified dihedral angles
        atoms_rotated (dict): indices of atoms that are going to be rotated
    Returns:
        new_nxyz: torch.tensor of shape (n, 3), new coordinates of the molecule
        rotated_dihedrals: amount of rotation angles in degrees
    """
    if nxyz.device != torch.device(device):
        nxyz = nxyz.to(device)

    new_xyz = nxyz[:, 1:].clone()

    for name in rotations.keys():
        xyz = new_xyz.clone()
        rotated_xyz = get_rotated_xyz(
            theta=rotations[name],
            u=get_axis_vector(xyz[dihed_inds[name][1]],
                              xyz[dihed_inds[name][2]]),
            center=xyz[dihed_inds[name][2]],
            xyz=xyz,
            mask_indices=atoms_rotated[name],
            device=device,
        )

        transl_xyz = rotated_xyz - xyz

        new_xyz = new_xyz + transl_xyz

    new_nxyz = torch.cat([nxyz[:, 0][:, None], new_xyz], dim=1).detach().cpu()
    rotation_angs = torch.tensor(list(rotations.values())).detach().cpu()

    return new_nxyz, rotation_angs


def get_atom_dihedrals(data: dict, dihed_inds: np.ndarray):
    atoms = Atoms(
        numbers=data["nxyz"][:, 0],
        positions=data["nxyz"][:, 1:],
        pbc=False,
    )
    diheds = atoms.get_dihedrals(dihed_inds)
    return diheds


def get_dihedral_counts_histogram(dihedrals: np.ndarray, num_bins: int = 361):
    # define the edges of the bins for each dimension
    edges = np.linspace(-180, 180, num_bins)
    # create a histogram of the explored dihedrals
    histograms = [
        np.histogram(dihedrals[:, i], bins=edges)[0] for i in range(dihedrals.shape[1])
    ]
    return histograms, edges


def get_rand_unexplored_diheds(explored_diheds: np.ndarray, num_bins: int = 361):
    histograms, edges = get_dihedral_counts_histogram(
        explored_diheds, num_bins=num_bins
    )
    # any bin with a count of 0 is unexplored
    unexplored = [hist == 0 for hist in histograms]
    # now unexplored is a list of boolean arrays, each of which is True for bins that are unexplored
    # we want to find the indices of the unexplored bins
    new_diheds = []
    for i, u in enumerate(unexplored):
        unexp = np.where(u)[0]
        if len(unexp) == 0:
            min_unexp = np.argmin(histograms[i])
            new_diheds.append(edges[min_unexp] + np.random.rand())
        else:
            new_diheds.append(
                edges[np.random.choice(unexp)] + np.random.rand())

    new_diheds = np.array(new_diheds)
    return new_diheds


def get_rotations_needed(
    seed_geom: Atoms, targ_diheds: np.ndarray, dihed_inds: np.ndarray, dihed_names: list
):
    seed_diheds = seed_geom.get_dihedrals(dihed_inds)
    rotations = {
        k: (targ_diheds[i] - seed_diheds[i]) % 360 for i, k in enumerate(dihed_names)
    }
    rotations = {k: torch.tensor(v) for k, v in rotations.items()}
    return rotations


def get_geom_with_new_dihedrals(
    seed_geom: Atoms,
    targ_diheds: np.ndarray,
    dihed_info: dict,
    branch_info: dict,
    device: str = "cpu",
) -> Tuple[Atoms, torch.tensor]:
    rotations = get_rotations_needed(
        seed_geom=seed_geom,
        targ_diheds=targ_diheds,
        dihed_inds=np.array(list(dihed_info.values())),
        dihed_names=list(dihed_info.keys()),
    )

    seed_nxyz = torch.tensor(
        np.concatenate(
            [
                seed_geom.get_atomic_numbers().reshape(-1, 1),
                seed_geom.get_positions(),
            ],
            axis=-1,
            dtype=np.float32,
        )
    )

    # rotate the atom
    new_nxyz, rotation_diheds = set_dihedrals(
        nxyz=seed_nxyz,
        rotations=rotations,
        device=device,
        dihed_inds=dihed_info,
        atoms_rotated=branch_info,
    )
    rotated_geom = Atoms(
        numbers=new_nxyz[:, 0].numpy(),
        positions=new_nxyz[:, 1:].numpy(),
        pbc=seed_geom.pbc,
    )

    dihed_inds = np.array(list(dihed_info.values()))

    return rotated_geom, rotation_diheds


def handle_dihedral_setting(
    config: dict,
    dihed_inds: np.ndarray,
    dihed_names: list,
    num_bins: int = 361,
    new_geom_path: str = "new_geom.xyz",
):
    dset = Dataset.from_file(config["dset"]["path"])
    dihedrals = np.stack([get_atom_dihedrals(d) for d in dset])

    # get a sample atom as the seed
    d = dset[np.random.choice(len(dset))]
    seed_geom = Atoms(
        numbers=d["nxyz"][:, 0],
        positions=d["nxyz"][:, 1:],
        pbc=False,
    )
    seed_diheds = seed_geom.get_dihedrals(dihed_inds)

    # get the dihedrals that have not been explored
    targ_diheds = get_rand_unexplored_diheds(dihedrals)

    # get geometry with the target dihedrals
    new_nxyz, rotation_angles = get_geom_with_new_dihedrals(
        seed_geom=seed_geom,
        targ_diheds=targ_diheds,
        dihed_inds=dihed_inds,
        dihed_names=dihed_names,
        device=DEVICE,
    )

    # save the new geometry
    new_geom = Atoms(
        numbers=new_nxyz[:, 0],
        positions=new_nxyz[:, 1:],
        pbc=False,
    )
    new_geom.write(new_geom_path)
