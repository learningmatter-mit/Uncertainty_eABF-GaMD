import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations
from openmm import unit
from nff.data import Dataset, concatenate_dict

sys.path.append(f"{os.getenv('PROJECTSDIR')}/uncertainty_eABF-GaMD")
from eabfgamd.calc import OpenMMCalculator
from eabfgamd.ala import (
    Paths,
    OpenMMTrajectory,
    AmberTopologyInfo,
)
from eabfgamd.misc import convert_0_2pi_to_negpi_pi


def simulate_using_amber_ff(params, total_time, output_dir):
    assert hasattr(total_time, "unit")

    total_steps = int(total_time / (params["timestep"] * unit.picoseconds))

    calculator = OpenMMCalculator(params, output_dir)
    calculator.simulate(total_steps)


def load_results_from_amber_simulation(output_dir):
    results = OpenMMTrajectory(output_dir)
    temperature = int(output_dir.split("/")[-1].split("K")[0])

    dihedral_inds = AmberTopologyInfo.get_dihedral_inds(
        ["phi", "psi", "omega_1", "omega_2"]
    )

    phi = results.get_dihedrals(*dihedral_inds[0])
    psi = results.get_dihedrals(*dihedral_inds[1])
    omega_1 = results.get_dihedrals(*dihedral_inds[2])
    omega_2 = results.get_dihedrals(*dihedral_inds[3])
    dihedrals = np.stack([phi, psi, omega_1, omega_2], axis=1)

    vfunc = np.vectorize(convert_0_2pi_to_negpi_pi)
    dihedrals = vfunc(dihedrals)
    nbr_list = torch.LongTensor(list(combinations(range(results.natoms), 2)))

    data = []
    for i, atoms in enumerate(results.trajectory):
        nxyz = np.concatenate(
            [atoms.get_atomic_numbers().reshape(-1, 1), atoms.get_positions()], axis=1
        )
        energy = results.state_data.loc[i, "Potential Energy (kJ/mole)"] * (
            unit.kilojoule / unit.mole
        )
        forces = results.forces[i] * \
            (unit.kilojoule / unit.mole / unit.angstrom)
        energy = energy.value_in_unit(unit.kilocalorie / unit.mole)
        forces = forces.value_in_unit(
            unit.kilocalorie / unit.mole / unit.angstrom)
        dihed = torch.FloatTensor(dihedrals[i]).unsqueeze(0)
        datum = {
            "nxyz": torch.FloatTensor(nxyz),
            "num_atoms": torch.LongTensor([len(nxyz)]).squeeze(),
            "energy": torch.FloatTensor([energy]).squeeze(),
            "energy_grad": -torch.FloatTensor(forces),
            "dihedrals": dihed,
            "nbr_list": nbr_list,
            "identifier": f"gen0_{temperature}K_{i}",
        }
        data.append(datum)

    dset = Dataset(concatenate_dict(*data))

    return dset


def create_train_set_from_amber_simulation():
    params = {
        "pdb": "ala.pdb",
        "forcefield": "amber",
        "prmtop": Paths.amber_prmtop_path,
        "inpcrd": Paths.amber_inpcrd_path,
        "cutoff": 1.0,  # nm
        "temperature": 500.0,  # K
        "timestep": 0.0002,  # ps
        "hbond_constraint": False,
        "constraint_tol": 0.00001,
        "record_interval": 1000,  # 0.2 ps
    }

    total_time = 1.0 * unit.nanoseconds
    output_dir = (
        f"{Paths.data_dir}/{params['forcefield']}/{int(params['temperature'])}K"
    )

    simulate_using_amber_ff(params, total_time, output_dir)
    dset = load_results_from_amber_simulation(output_dir)

    dset.save(
        f"{Paths.data_dir}/amber/alad{int(params['temperature'])}K_kcalmol.pth.tar"
    )
    print("Saved dataset")


def calc_from_opls_train_set():
    params = {
        "forcefield": "amber",
        "prmtop": Paths.amber_prmtop_path,
        "inpcrd": Paths.amber_inpcrd_path,
        "cutoff": 1.0,  # nm
        "temperature": 300.0,  # K
        "timestep": 0.0002,  # ps
        "constraint_tol": 1e-6,
        "record_interval": 1,
    }
    calculator = OpenMMCalculator(params)

    dset = Dataset.from_file(
        f"{os.getenv('DATA')}/projects/unc_eabf/ala/data/opls/alad1200K_kcalmol.pth.tar"
    )
    dset = list(dset)

    for i, geom in enumerate(tqdm(dset)):
        energy, forces = calculator.single_point_calc(geom)
        dset[i]["energy"] = torch.FloatTensor([energy]).squeeze()
        dset[i]["energy_grad"] = -torch.FloatTensor(forces)

    dset = Dataset(concatenate_dict(*dset))

    dset.save(
        f"{os.getenv('DATA')}/projects/unc_eabf/ala/data/amber/alad1200K_kcalmol.pth.tar"
    )
    print("Saved dataset")


if __name__ == "__main__":
    create_train_set_from_amber_simulation()
