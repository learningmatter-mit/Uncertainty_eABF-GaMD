import os
import argparse
import time
import yaml
import torch
import numpy as np
from typing import Union
from ase import Atoms, units
from nff.md.nvt import Langevin
from nff.io.ase import NeuralFF, AtomsBatch
from nff.data import Dataset

import sys

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.misc import get_atoms, load_from_xyz
from eabfgamd.ensemble import get_ensemble_path
from eabfgamd.enhsamp import relax_atoms
from eabfgamd.mace_calculators import MACECalculator


def check_completion(traj_dir):
    if not os.path.exists(f"{traj_dir}/status.log"):
        return False

    with open(f"{traj_dir}/status.log", "r") as f:
        status = f.read()

    return "Completed" in status


def run_dynamics(
    starting_atoms: Union[Atoms, AtomsBatch],
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
        yaml.dump(md_params, f)

    # set up calculator
    if model_type in ["painn", "schnet"]:
        model = torch.load(model_path, map_location=device)
        calc = NeuralFF(
            model=model,
            device=device,
            en_key="energy",
            directed=directed,
        )
    elif model_type == "mace":
        calc = MACECalculator(
            model_path=get_ensemble_path(model_path),
            device=device,
            energy_units_to_eV=0.0433634,
            length_units_to_A=1.0,
            default_dtype="float64",
            charges_key="Qs",
            model_type="MACE",
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
        logfile=f"{md_params['traj_dir']}/equil.log",
        trajectory=f"{md_params['traj_dir']}/equil.traj",
        loginterval=md_params["loginterval"],
        unphysical_rmin=md_params["unphysical_rmin"],
        unphysical_rmax=md_params["unphysical_rmax"],
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
        friction_per_ps=md_params["friction_per_ps"] * 1.0e-3 / units.fs,
        # maxwell_temp=md_params["maxwell_temp"],
        logfile=f"{md_params['traj_dir']}/prod.log",
        trajectory=f"{md_params['traj_dir']}/prod.traj",
        loginterval=md_params["loginterval"],
        unphysical_rmin=md_params["unphysical_rmin"],
        unphysical_rmax=md_params["unphysical_rmax"],
    )
    start_time = time.time()
    dyn.run(steps=md_params["prod_nsteps"])
    end_time = time.time()

    print(f"Production run took {end_time - start_time} seconds")
    print("Done")
    print("----------------------------------------------------------------")


def get_params(
    dt: float,  # fs
    temperature: float,  # K
    friction_per_ps: float,  # 1/ps
    unphysical_rmin: float,  # angstrom
    unphysical_rmax: float,  # angstrom
    loginterval: int,
    equil_nsteps: int,
    prod_nsteps: int,
    traj_dir: str,
    forcefield: str = "amber",
):
    md_params = {
        "dt": dt,  # fs
        "temperature": temperature,  # K
        "friction_per_ps": friction_per_ps,  # 1/ps
        "unphysical_rmin": unphysical_rmin,  # angstrom
        "unphysical_rmax": unphysical_rmax,  # angstrom
        "forcefield": forcefield,
        "loginterval": loginterval,
        "equil_nsteps": equil_nsteps,
        "prod_nsteps": prod_nsteps,
        "traj_dir": traj_dir,
    }

    return md_params


def run_nvt(
    model_path: str,
    dataset_path: str,
    traj_dir: str,
    model_type: str = "mace",
    forcefield: str = "amber",
    dt: float = 0.25,  # fs
    temperature: float = 300.0,  # K
    friction_per_ps: float = 1.0,  # 1/ps
    unphysical_rmin: float = 0.75,  # angstrom
    unphysical_rmax: float = 2.00,  # angstrom
    loginterval: int = 10,
    equil_nsteps: int = 5000,
    prod_nsteps: int = 10000,
):

    logfile = f"{traj_dir}/status.log"

    # check completion
    completed = check_completion(traj_dir)
    if completed:
        print(f"Trajectory {traj_dir} already completed. Skipping.")
        return

    print("================================================================")
    print(f"NVT for model {model_path} at {temperature}K")

    # device = get_least_used_device()
    device = "cuda"

    if dataset_path.endswith(".pth.tar"):
        dset = Dataset.from_file(dataset_path)
        min_e_idx = np.argmin(dset.props['energy'])
        starting_atoms = AtomsBatch.from_atoms(
            get_atoms(dset[min_e_idx]), directed=True
        )
    elif dataset_path.endswith(".xyz"):
        dset = load_from_xyz(dataset_path)
        min_e_idx = np.argmin([at.info["energy"] for at in dset])
        starting_atoms = AtomsBatch.from_atoms(dset[min_e_idx], directed=True)
    else:
        raise ValueError(f"Dataset {dataset_path} not supported")

    md_params = get_params(
        forcefield=forcefield,
        traj_dir=traj_dir,
        dt=dt,
        temperature=temperature,
        friction_per_ps=friction_per_ps,
        unphysical_rmin=unphysical_rmin,
        unphysical_rmax=unphysical_rmax,
        loginterval=loginterval,
        equil_nsteps=equil_nsteps,
        prod_nsteps=prod_nsteps,
    )

    run_dynamics(
        starting_atoms=starting_atoms,
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


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--traj_dir", "-t", type=str, required=True)
    parser.add_argument("--forcefield", "-ff", type=str, default="amber")
    parser.add_argument("--model_type", type=str, default="mace")
    parser.add_argument("--dt", "-dt", type=float, default=0.25, help="timestep[fs]")
    parser.add_argument("--temperature", "-T", type=float, default=300.0, help="temperature[K]")
    parser.add_argument("--friction_per_ps", type=float, default=1.0, help="friction[1/ps]")
    parser.add_argument("--unphysical_rmin", "-rmin", type=float, default=0.75, help="unphysical_rmin[angstrom]")
    parser.add_argument("--unphysical_rmax", "-rmax", type=float, default=2.00, help="unphysical_rmax[angstrom]")
    parser.add_argument("--loginterval", type=int, default=10, help="loginterval")
    parser.add_argument("--equil_nsteps", type=int, default=1000, help="equil_nsteps")
    parser.add_argument("--prod_nsteps", type=int, default=10000, help="prod_nsteps")

    args = parser.parse_args()

    run_nvt(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        traj_dir=args.traj_dir,
        forcefield=args.forcefield,
        model_type=args.model_type,
        dt=args.dt,
        temperature=args.temperature,
        friction_per_ps=args.friction_per_ps,
        unphysical_rmin=args.unphysical_rmin,
        unphysical_rmax=args.unphysical_rmax,
        loginterval=args.loginterval,
        equil_nsteps=args.equil_nsteps,
        prod_nsteps=args.prod_nsteps,
    )


if __name__ == "__main__":
    argument_parser()
