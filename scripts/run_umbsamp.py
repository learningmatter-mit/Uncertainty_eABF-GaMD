import os
import argparse
import yaml
import torch
import numpy as np
import itertools
from typing import Union
from ase import Atoms
from nff.md.nvt import Langevin
from nff.md.utils import BiasedNeuralMDLogger
from nff.io.ase import AtomsBatch
from evi.data import Dataset
from ase.io import read

import sys

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.misc import get_atoms, TopoInfo
from eabfgamd.enhsamp import relax_atoms
from eabfgamd.umbrella_sampling import NFFUmbrellaSampling, MACEUmbrellaSampling


def check_completion(traj_dir):
    if not os.path.exists(f"{traj_dir}/status.log"):
        return False

    with open(f"{traj_dir}/status.log", "r") as f:
        status = f.read()

    return "Completed" in status


def run_dynamics(
    starting_atoms: Union[Atoms, AtomsBatch],
    cv_defs: list[dict],
    md_params: dict,
    model_type: str,
    model_path: str,
    device: str,
    directed: bool = True,
):
    traj_dir = md_params["traj_dir"]

    print("----------------------------------------------------------------")
    print(f"Running dynamics for {traj_dir}")

    with open(f"{traj_dir}/params.yaml", "w") as f:
        params = {"cv_def": cv_defs, "md_params": md_params}
        yaml.dump(params, f)

    # set up umbrella sampling calculator
    if model_type in ["painn", "schnet"]:
        model = torch.load(model_path, map_location=device)
        calc = NFFUmbrellaSampling(
            model=model,
            cv_defs=cv_defs,
            device=device,
            en_key="energy",
            directed=directed,
        )
    elif model_type == "mace":
        calc = MACEUmbrellaSampling(
            model_path=model_path,
            cv_defs=cv_defs,
            device=device,
            energy_units_to_eV=0.0433634,
            length_units_to_A=1.0,
        )

    # set calculator to atoms
    starting_atoms.set_calculator(calc)

    # relax atoms
    print("Relaxing atoms")
    relax_atoms(
        starting_atoms,
        algs="lbfgs",
    )

    # initial equilibrium
    print("Running initial equilibrium dynamics")
    dyn = Langevin(
        starting_atoms,
        timestep=md_params["dt"],
        temperature=md_params["temperature"],
        friction_per_ps=md_params["friction_per_ps"],
        # maxwell_temp=md_params["maxwell_temp"],
        logfile=f"{md_params['traj_dir']}/equil_md.log",
        trajectory=f"{md_params['traj_dir']}/equil.traj",
        loginterval=md_params["loginterval"],
        unphysical_rmin=md_params["unphysical_rmin"],
        unphysical_rmax=md_params["unphysical_rmax"],
    )
    dyn.attach(
        BiasedNeuralMDLogger(
            dyn,
            starting_atoms,
            f"{md_params['traj_dir']}/equil_umbsamp.log",
            header=True,
            mode="w",
        ),
        interval=md_params["loginterval"],
    )
    dyn.run(steps=md_params["equil_nsteps"])
    equil_atoms = dyn.atoms

    # production run
    print("Running production dynamics")

    equil_atoms.set_calculator(calc)

    dyn = Langevin(
        equil_atoms,
        timestep=md_params["dt"],
        temperature=md_params["temperature"],
        friction_per_ps=md_params["friction_per_ps"],
        # maxwell_temp=md_params["maxwell_temp"],
        logfile=f"{md_params['traj_dir']}/prod_md.log",
        trajectory=f"{md_params['traj_dir']}/prod.traj",
        loginterval=md_params["loginterval"],
        unphysical_rmin=md_params["unphysical_rmin"],
        unphysical_rmax=md_params["unphysical_rmax"],
    )

    dyn.attach(
        BiasedNeuralMDLogger(
            dyn,
            equil_atoms,
            f"{md_params['traj_dir']}/prod_umbsamp.log",
            header=True,
            mode="w",
        ),
        interval=md_params["loginterval"],
    )

    dyn.run(steps=md_params["prod_nsteps"])

    print("Done")
    print("----------------------------------------------------------------")


def get_params(
    phi_center: float,  # radians
    psi_center: float,  # radians
    k_phi: float,  # eV/rad^2
    k_psi: float,  # eV/rad^2
    forcefield: str = "amber",
    dt: float = 0.25,  # fs
    temperature: float = 298.15,  # K
    friction_per_ps: float = 1.0,  # 1/ps
    unphysical_rmin: float = 0.75,  # angstrom
    unphysical_rmax: float = 2.00,  # angstrom
    loginterval: int = 10,
    equil_nsteps: int = 5000,
    prod_nsteps: int = 10000,
    traj_dir: Union[str, None] = None,
):
    cv_defs = [
        {
            "definition": {
                "type": "dihedral",
                "index_list": TopoInfo.get_dihedral_inds(forcefield, ["phi"])
                .squeeze()
                .tolist(),
            },
            "name": "phi",
            "k": k_phi,
            "center": phi_center,
        },
        {
            "definition": {
                "type": "dihedral",
                "index_list": TopoInfo.get_dihedral_inds(forcefield, ["psi"])
                .squeeze()
                .tolist(),
            },
            "name": "psi",
            "k": k_psi,
            "center": psi_center,
        },
    ]

    md_params = {
        "dt": dt,  # fs
        "temperature": temperature,
        "friction_per_ps": friction_per_ps,
        "unphysical_rmin": unphysical_rmin,  # angstrom
        "unphysical_rmax": unphysical_rmax,  # angstrom
        "loginterval": loginterval,
        "equil_nsteps": equil_nsteps,
        "prod_nsteps": prod_nsteps,
        "traj_dir": traj_dir,
    }

    return cv_defs, md_params


def calculate_k(ngrids: int, T: float = 298.15, unit: str = "eV/rad^2"):
    kB = {
        "kJ/mol/rad^2": 8.314462618e-3,
        "kcal/mol/rad^2": 1.9872043e-3,
        "eV/rad^2": 8.617333262145e-5,
    }
    kT = kB[unit] * T
    grid_size = 2 * np.pi / ngrids
    return kT / (grid_size**2)


def run_umbrella_sampling(
    model_path: str,
    dataset_path: str,
    traj_dir: str,
    phi_center: float,  # radians
    psi_center: float,  # radians
    k_phi: float,  # eV/rad^2
    k_psi: float,  # eV/rad^2
    model_type: str = "mace",
    forcefield: str = "amber",
):

    logfile = f"{traj_dir}/status.log"

    # check completion
    completed = check_completion(traj_dir)
    if completed:
        print(f"Trajectory {traj_dir} already completed. Skipping.")
        return

    print("================================================================")
    print(f"Umbrella sampling for model {model_path} at center ({phi_center:.2f}, {psi_center:.2f})")

    # device = get_least_used_device()
    device = "cuda"

    if dataset_path.endswith(".pth.tar"):
        dset = Dataset.from_file(dataset_path)
        min_e_idx = np.argmin(dset.props['energy'])
        starting_atoms = AtomsBatch.from_atoms(
            get_atoms(dset[min_e_idx]), directed=True
        )
    elif dataset_path.endswith(".xyz"):
        dset = read(dataset_path, index=":")
        min_e_idx = np.argmin([at.info["energy"] for at in dset])
        starting_atoms = AtomsBatch.from_atoms(dset[min_e_idx], directed=True)
    else:
        raise ValueError(f"Dataset {dataset_path} not supported")

    cv_defs, md_params = get_params(
        phi_center=phi_center,
        psi_center=psi_center,
        k_phi=k_phi,
        k_psi=k_psi,
        forcefield=forcefield,
        traj_dir=traj_dir,
    )

    run_dynamics(
        starting_atoms=starting_atoms,
        cv_defs=cv_defs,
        md_params=md_params,
        model_path=model_path,
        model_type=model_type,
        device=device,
        directed=True,
    )

    with open(logfile, "w") as f:
        f.write("Completed")
    print(f"Completed for {traj_dir}")
    print("================================================================")


def sweep_umbrella_sampling(
    model_path: str, dataset_path: str, traj_dir:str, ngrids: int, cpu_count: int = 1, forcefield: str = "amber", model_type: str = "mace"
):
    import torch.multiprocessing as mp

    print("================================================================")
    print(f"Umbrella sampling sweep over {ngrids ** 2} grids for model {model_path} with {cpu_count} CPUs")
    phi_centers = np.linspace(-np.pi, np.pi, ngrids, endpoint=False)
    psi_centers = np.linspace(-np.pi, np.pi, ngrids, endpoint=False)

    phi_psi = np.meshgrid(phi_centers, psi_centers)
    phi_psi = np.stack(phi_psi, -1).reshape(-1, 2)

    k = calculate_k(ngrids)

    with mp.Pool(cpu_count) as pool:
        pool.starmap(
            run_umbrella_sampling,
            zip(
                itertools.repeat(model_path),
                itertools.repeat(dataset_path),
                itertools.repeat(traj_dir),
                phi_psi[:, 0],
                phi_psi[:, 1],
                itertools.repeat(k),
                itertools.repeat(k),
                itertools.repeat(forcefield),
                itertools.repeat(model_type),
            ),
        )
        pool.close()
        pool.join()

    print("================================================================")


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help="sweep over grid or single run")

    sweep_parser = subparsers.add_parser("sweep", help="sweep over grid")
    sweep_parser.add_argument("--model_path", "-m", type=str, required=True)
    sweep_parser.add_argument("--dataset_path", "-d", type=str, required=True)
    sweep_parser.add_argument("--traj_dir", "-t", type=str, required=True)
    sweep_parser.add_argument("--ngrids", "-n", type=int, required=True)
    sweep_parser.add_argument("--cpu_count", type=int, default=1)
    sweep_parser.add_argument("--forcefield", "-ff", type=str, default="amber")
    sweep_parser.add_argument("--model_type", type=str, default="mace")

    single_parser = subparsers.add_parser("single", help="single run")
    single_parser.add_argument("--model_path", "-m", type=str, required=True)
    single_parser.add_argument("--dataset_path", "-d", type=str, required=True)
    single_parser.add_argument("--traj_dir", "-t", type=str, required=True)
    single_parser.add_argument("--phi", "-phi", type=float, required=True)
    single_parser.add_argument("--psi", "-psi", type=float, required=True)
    single_parser.add_argument("--ngrids", "-n", type=int, required=True)
    single_parser.add_argument("--k_phi", "-k_phi", type=float, default=None)
    single_parser.add_argument("--k_psi", "-k_psi", type=float, default=None)
    single_parser.add_argument("--forcefield", "-ff", type=str, default="amber")
    single_parser.add_argument("--model_type", type=str, default="mace")

    args = parser.parse_args()

    if args.command == 'sweep':
        sweep_umbrella_sampling(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            traj_dir=args.traj_dir,
            ngrids=args.ngrids,
            cpu_count=args.cpu_count,
            forcefield=args.forcefield,
            model_type=args.model_type,
        )

    elif args.command == 'single':
        k = calculate_k(args.ngrids)
        k_phi, k_psi = k, k
        if args.k_phi is not None:
            k_phi = args.k_phi
        if args.k_psi is not None:
            k_psi = args.k_psi
        run_umbrella_sampling(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            traj_dir=args.traj_dir,
            phi_center=args.phi,
            psi_center=args.psi,
            k_phi=k_phi,
            k_psi=k_psi,
            forcefield=args.forcefield,
            model_type=args.model_type,
        )


if __name__ == "__main__":
    argument_parser()
