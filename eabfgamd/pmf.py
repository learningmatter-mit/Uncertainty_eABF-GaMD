import numpy as np


__all__ = [
    "pmf_unbiased_traj_phipsi_cv",
]


def pmf_unbiased_traj_phipsi_cv(
    ngrid: int,
    cv: np.ndarray,
    energy: np.ndarray,
    equil_temp: float = 300.0,
) -> tuple:
    """
    Get PMF for unbiased simulation of alanine dipeptide
    where CVs are the phi and psi dihedral angles. Energy
    values should be given in kcal/mol.
    :param ngrid: number of bins
    :param cv: array of size (N, 2) containing phi and psi
        dihedral angles of the configurations in trajectory.
        Dihedral angles should be from -180 to 180 degree.
    :param energy: array of size (N,) containing energy in
        kcal/mol unit of configurations in the trajectory
    :param equil_temp: equilibration temperature in kelvin
    :return tuple: Tuple of phi_centers of size (ngrid, 1),
        psi_centers of size (ngrid, 1), rho of size
        (ngrid, ngrid), and W of size (ngrid, ngrid). rho
        contains the probability $p_i(\\phi, \\psi)$ to
        find a configuration $i$ in a specific $\\phi$ and
        $\\psi$ grid. W is the PMF.
    """
    psi_edges = np.linspace(-180, 180, ngrid + 1)
    phi_edges = np.linspace(-180, 180, ngrid + 1)

    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    psi_centers = (psi_edges[:-1] + psi_edges[1:]) / 2

    # Digitize to find bin indices
    phi_indices = np.digitize(cv[:, 0], phi_edges) - 1
    psi_indices = np.digitize(cv[:, 1], psi_edges) - 1

    # Create an empty list to store indices for each grid point
    grid_indices = [[[] for _ in range(ngrid)] for _ in range(ngrid)]

    # Assign each coordinate index to the appropriate grid point
    for idx, (phi_idx, psi_idx) in enumerate(zip(phi_indices, psi_indices)):
        if phi_idx < ngrid and psi_idx < ngrid:  # Ensure index is within grid
            grid_indices[phi_idx][psi_idx].append(idx)
        else:
            raise Exception("Something is wrong")

    R_kcal = 1.98720425864083 / 1000.0
    RT = R_kcal * equil_temp
    beta = 1.0 / RT

    Z = np.exp(-beta * energy).sum()

    rho = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            inds = grid_indices[i][j]
            U0 = energy[inds]
            rho[i, j] = np.exp(-beta * U0).sum() / Z

    W = -RT * np.log(rho)

    return phi_centers, psi_centers, rho, W
