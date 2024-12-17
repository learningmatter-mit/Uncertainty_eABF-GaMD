import gc
import os
import time
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from ase import Atoms, units

from nff.data import Dataset
from nff.io.ase import AtomsBatch
from nff.md.utils import BiasedNeuralMDLogger

from .ala import TopoInfo
from .ensemble import get_ensemble_path
from .misc import (
    get_least_used_device,
    get_logger,
    get_main_model_outdir,
    load_from_xyz,
    make_and_backup_dir,
    save_config,
)
from .mdlogger import MDLogger
from .npt import BerendsenNPT
from .nvt import Langevin

__all__ = [
    "relax_atoms",
    "make_config_for_each_simulation",
    "make_config_for_each_unbiased_simulation",
    "get_starting_atom",
    "get_CV_parameters",
    "get_uncertainty_params",
    "set_up_unbiased_calculator",
    "set_up_biased_calculator",
    "set_up_dynamics",
    "run_dynamics",
    "run",
    "run_unbiased_dynamics",
    "handle_multiproc_enhsamp",
]


logger = get_logger("BiasedMD")


def check_completion(config: dict) -> bool:
    if not os.path.exists(f"{config['enhsamp']['traj_dir']}/status.log"):
        return False
    else:
        with open(f"{config['enhsamp']['traj_dir']}/status.log", "r") as f:
            lines = f.read()
            if "BiasedMD: Done" in lines:
                return True
            else:
                return False


def make_config_for_each_simulation(config: dict) -> List[dict]:
    md_ensemble = config["enhsamp"].get("md_ensemble", "nvt")

    num_simulations = config["enhsamp"]["num_simulations"]
    temperatures = config["enhsamp"]["temperature"]
    pressures = config["enhsamp"].get("pressure", None)
    traj_dirs = config["enhsamp"]["traj_dir"]

    # check that the number of simulations is consistent
    assert num_simulations == len(temperatures) == len(traj_dirs)
    if md_ensemble == "npt":
        assert num_simulations == len(pressures)

    # get the ensemble path
    model_outdir = get_main_model_outdir(config["model"]["outdir"])

    # create a config for each simulation
    parameters = []
    for i in range(config["enhsamp"]["num_simulations"]):
        params = deepcopy(config)

        params["model"]["outdir"] = model_outdir

        params["enhsamp"] = {}
        params["enhsamp"]["traj_dir"] = traj_dirs[i]
        params["enhsamp"]["temperature"] = temperatures[i]

        if md_ensemble == "nvt":
            params["enhsamp"]["scaling_array"] = config["enhsamp"].get(
                "scaling_array", [1, 1, 1, 1, 1, 1]
            )
        elif md_ensemble == "npt":
            params["enhsamp"]["pressure"] = pressures[i]
            params["enhsamp"]["barostat_mask"] = config["enhsamp"].get(
                "barostat_mask", [1, 1, 1]
            )
        else:
            raise ValueError(f"Unknown statistical ensemble: {md_ensemble}")

        for key, val in config["enhsamp"].items():
            if key in [
                "temperature",
                "pressure",
                "traj_dir",
                "scaling_array",
                "barostat_mask",
            ]:
                continue  # already set above
            elif isinstance(val, list):
                params["enhsamp"][key] = val[i]
            else:
                params["enhsamp"][key] = val

        # set random starting geometry
        params["enhsamp"]["start_geom_idx"] = np.random.randint(0, 100000)

        parameters.append(params)

    return parameters


def get_uncertainty_params(config: dict) -> dict:
    # set up uncertainty parameters for CV
    if config["uncertainty"]["type"] == "ensemble":
        uncertainty_params = {
            "quantity": config["uncertainty"]["quantity"],
            "order": config["uncertainty"]["order"],
            "std_or_var": config["uncertainty"]["std_or_var"],
        }
    elif config["uncertainty"]["type"] == "mve":
        uncertainty_params = {
            "quantity": config["uncertainty"]["quantity"],
            "vkey": config["uncertainty"]["mve_vkey"],
            "order": config["uncertainty"]["order"],
        }
    elif config["uncertainty"]["type"] == "evidential":
        uncertainty_params = {
            "order": config["uncertainty"]["order"],
            "shared_v": config["uncertainty"]["evi_shared_v"],
            "source": config["uncertainty"]["evi_source"],
            "quantity": config["uncertainty"]["quantity"],
            "calibrate": config["uncertainty"].get("calibrate", False),
            "cp_alpha": config["uncertainty"].get("cp_alpha", None),
        }
    elif config["uncertainty"]["type"] == "gmm":
        uncertainty_params = {
            "train_embed_key": config["uncertainty"]["gmm_train_embed_key"],
            "test_embed_key": config["uncertainty"]["gmm_test_embed_key"],
            "quantity": config["uncertainty"]["quantity"],
            "order": config["uncertainty"]["order"],
            "set_min_uncertainty_at_level": config["uncertainty"].get(
                "set_min_uncertainty_at_level", "system"
            ),
            "n_clusters": config["uncertainty"]["gmm_n_clusters"],
            "calibrate": config["uncertainty"].get("calibrate", False),
            "cp_alpha": config["uncertainty"].get("cp_alpha", None),
            "gmm_path": config["uncertainty"].get("gmm_path", None),
        }
    else:
        raise ValueError(
            f"Uncertainty type {config['uncertainty']['type']} not recognized"
        )

    return uncertainty_params


def get_starting_atom(
    dset: Dataset,
    random_idx: Union[int, None] = None,
    directed: bool = True,
    device: str = "cuda",
    scaling_array: Union[None, List[float]] = None,
    cutoff: float = 5.0,
    cutoff_skin: float = 0.0,
) -> Tuple[AtomsBatch, int]:
    #     # get starting geometry
    #     if random_idx is None:
    #         random_idx = np.random.choice(len(dset))
    #     else:
    #         random_idx = random_idx % len(dset)

    energies = np.array([atoms.info["energy"] for atoms in dset])
    random_idx = np.argmin(energies)

    start_geom = dset[random_idx].copy()
    if isinstance(start_geom, dict):
        start_geom = Atoms(
            numbers=start_geom["nxyz"][:, 0],
            positions=start_geom["nxyz"][:, 1:],
            cell=start_geom.get("lattice", None),
            pbc=start_geom.get("lattice", False),
        )

    if scaling_array is not None:
        assert len(scaling_array) == 6

        cell = start_geom.get_cell_lengths_and_angles()
        new_cell = cell * np.array(scaling_array)
        start_geom.set_cell(new_cell, scale_atoms=True)

    # set up starting atoms
    starting_atoms = AtomsBatch.from_atoms(
        start_geom,
        directed=directed,
        device=device,
        cutoff=cutoff,
        cutoff_skin=cutoff_skin,
    )

    return starting_atoms, random_idx


def relax_atoms(
    starting_atoms: AtomsBatch,
    algs: str = "fire",
    nsteps: int = 100,
    fmax: float = 0.05,
) -> None:
    # relax the starting atoms
    if algs == "fire":
        from ase.optimize.fire import FIRE

        opt = FIRE(starting_atoms, logfile="-")
    elif algs == "mdmin":
        from ase.optimize import MDMin

        opt = MDMin(starting_atoms, logfile="-")
    elif algs == "bfgs":
        from ase.optimize import BFGS

        opt = BFGS(starting_atoms, logfile="-")
    elif algs == "lbfgs":
        from ase.optimize import LBFGS

        opt = LBFGS(starting_atoms, logfile="-")
    elif algs == "cg":
        from ase.optimize import CG

        opt = CG(starting_atoms, logfile="-")
    else:
        raise ValueError(f"Algorithm {algs} not recognized")

    if fmax is not None:
        opt.run(fmax=fmax, steps=nsteps)
    else:
        opt.run(steps=nsteps)


def get_CV_parameters(config: dict) -> dict:
    params = config["enhsamp"]

    uncertainty_params = get_uncertainty_params(config)
    cv_params = {
        "definition": {
            "type": params.get("cv_type", "uncertainty"),
            "model_path": get_ensemble_path(config["model"]["outdir"]),
            "device": config["train"]["device"],
            "directed": config["model"]["directed"],
            "model_type": config["model"]["model_type"],
            "batch_size": config["dset"]["batch_size"],
            "uncertainty_type": config["uncertainty"]["type"],
            "uncertainty_params": uncertainty_params,
            "cutoff": config["model"]["cutoff"],
            "cutoff_skin": 0.0,
        },
        "range": [params["cv_min"], params["cv_max"]],
        "bin_width": params["bin_width"],
        "margin": params["conf_margin"],  # there to keep in range
        "conf_k": params["conf_k"],  # harmonic wall k
    }

    if "eabf" in params["method"]:
        cv_params.update(
            {
                # 'ext_k': ['1e3'], # approx k = 10 eV/(cv-unit)^2
                "ext_sigma": params["ext_sigma"],
                "ext_mass": params["ext_mass"],
                "ext_pos": None,  # the starting position of the extended coordinate
            }
        )

    if params["method"] == "wtmeabf":
        cv_params.update(
            {
                "hill_height": params["hill_height"],
                "hill_std": params["hill_std"],  # std of Gaussians for WTMeABF
                # how often to drop hills
                "hill_drop_freq": params["hill_drop_freq"],
                "well_tempered_temp": params["well_tempered_temp"],
            }
        )
    elif params["method"] == "amdeabf":
        cv_params.update(
            {
                "amd_method": params["amd_method"],
                "amd_parameter": params["amd_parameter"],  # acceleration param
                "collect_pot_samples": params["collect_pot_samples"],
                "estimate_k": params["estimate_k"],
                "apply_amd": params["apply_amd"],
                "samd_c0": params["samd_c0"],
            }
        )

    if params["method"] == "attractivebias" and params["cv_type"] == "gmm_bias":
        cv_params["definition"].update(
            {
                "kB": params["gmm_bias_kB"],
                "T": params["gmm_bias_T"],
            }
        )

    return cv_params


def set_up_unbiased_calculator(
    config: dict,
    model=None,
):
    if config["model"]["model_type"] == "mace":
        from .mace_calculators import MACECalculator

        logger("Setting up unbiased MACE calculator")
        unbiased_calculator = MACECalculator(
            model_path=get_ensemble_path(config["model"]["outdir"]),
            device=config["train"]["device"],
            energy_units_to_eV=0.0433641,
            length_units_to_A=1.0,
            default_dtype="float64",
            charges_key="Qs",
            model_type="MACE",
        )
    elif config["model"]["model_type"] == "mace_mp":
        from .mace_calculators import MACEMPCalculator

        logger("Setting up unbiased MACE-MP calculator")
        unbiased_calculator = MACEMPCalculator(
            model="small",
            dispersion=False,
            default_dtype="float32",
            device=config["train"]["device"],
        )

    return unbiased_calculator


def set_up_biased_calculator(
    config: dict,
    starting_atoms: AtomsBatch,
    train_dset: Union[Dataset, List[Atoms], None],
    calib_dset: Union[Dataset, List[Atoms], None],
    cv_params: Union[dict, None] = None,
    model=None,
):
    cv_params["definition"]["train_dset"] = train_dset
    cv_params["definition"]["calib_dset"] = calib_dset

    if config["enhsamp"]["method"] == "eabf":
        from .mace_calculators import eABF

        logger("Setting up eABF calculator")
        calculator = eABF(
            model_path=cv_params["definition"]["model_path"],
            starting_atoms=starting_atoms,
            cv_defs=[cv_params],
            dt=config["enhsamp"]["dt"],
            friction_per_ps=config["enhsamp"]["friction_per_ps"],
            equil_temp=config["enhsamp"]["temperature"],
            nfull=config["enhsamp"]["nfull"],
            directed=config["model"]["directed"],
            device=config["train"]["device"],
        )
    elif config["enhsamp"]["method"] == "wtmeabf":
        from .mace_calculators import WTMeABF

        logger("Setting up WTMeABF calculator")
        calculator = WTMeABF(
            model_path=cv_params["definition"]["model_path"],
            starting_atoms=starting_atoms,
            cv_defs=[cv_params],
            dt=config["enhsamp"]["dt"],
            friction_per_ps=config["enhsamp"]["friction_per_ps"],
            equil_temp=config["enhsamp"]["temperature"],
            nfull=config["enhsamp"]["nfull"],
            directed=config["model"]["directed"],
            device=config["train"]["device"],
            hill_height=config["enhsamp"]["hill_height"],
            hill_drop_freq=config["enhsamp"]["hill_drop_freq"],
            well_tempered_temp=config["enhsamp"]["well_tempered_temp"],
        )
    elif config["enhsamp"]["method"] == "amdeabf":
        from .mace_calculators import aMDeABF

        logger("Setting up aMD-eABF calculator")
        calculator = aMDeABF(
            model_path=cv_params["definition"]["model_path"],
            starting_atoms=starting_atoms,
            cv_defs=[cv_params],
            dt=config["enhsamp"]["dt"],
            friction_per_ps=config["enhsamp"]["friction_per_ps"],
            amd_parameter=config["enhsamp"]["amd_parameter"],
            collect_pot_samples=config["enhsamp"]["collect_pot_samples"],
            estimate_k=config["enhsamp"]["estimate_k"],
            apply_amd=False,  # set to False for first few steps
            amd_method=config["enhsamp"]["amd_method"],
            samd_c0=config["enhsamp"]["samd_c0"],
            equil_temp=config["enhsamp"]["temperature"],
            nfull=config["enhsamp"]["nfull"],
            directed=config["model"]["directed"],
            device=config["train"]["device"],
        )
    elif config["enhsamp"]["method"] == "attractivebias":
        from .mace_calculators import AttractiveBias

        logger("Setting up AttractiveBias calculator")
        calculator = AttractiveBias(
            model_path=cv_params["definition"]["model_path"],
            cv_defs=[cv_params],
            gamma=config["enhsamp"]["gamma"],
            device=config["train"]["device"],
            directed=config["model"]["directed"],
            starting_atoms=starting_atoms,
        )
    elif config["enhsamp"]["method"] == "unbiased":
        from .mace_calculators import MACECalculator

        logger("Setting up unbiased MACE calculator")
        calculator = MACECalculator(
            model_path=get_ensemble_path(config["model"]["outdir"]),
            device=config["train"]["device"],
            energy_units_to_eV=0.0433641,
            length_units_to_A=1.0,
            default_dtype="float64",
            charges_key="Qs",
            model_type="MACE",
            cv_defs=[cv_params],
        )
    elif config["enhsamp"]["method"] == "unbiased_mp":
        from .mace_calculators import MACEMPCalculator

        logger("Setting up unbiased MACE-MP calculator")
        calculator = MACEMPCalculator(
            model="small",
            dispersion=False,
            default_dtype="float32",
            device=config["train"]["device"],
            # cv_defs=[cv_params],
        )
    else:
        raise ValueError(
            f"Unknown enhanced sampling method: {config['enhsamp']['method']}"
        )

    return calculator


def set_up_dynamics(
    config: dict, starting_atoms: AtomsBatch, nbr_list: Union[None, np.ndarray]
):
    params = config["enhsamp"]

    md_ensemble = params.get("md_ensemble", "nvt")

    if md_ensemble == "nvt":
        dyn = Langevin(
            starting_atoms,
            timestep=params["dt"] * units.fs,
            temperature_K=params["temperature"],  # K
            friction_per_ps=params["friction_per_ps"],  # 1/ps
            T_init=params["T_init"],
            logfile=f"{params['traj_dir']}/md.log",
            trajectory=f"{params['traj_dir']}/traj.traj",
            loginterval=params["loginterval"],
            unphysical_rmin=params.get("unphysical_rmin", None),
            unphysical_rmax=params.get("unphysical_rmax", None),
            hardcoded_nbr_list=nbr_list,
            append_trajectory=False,
        )

    elif md_ensemble == "npt":
        dyn = BerendsenNPT(
            atoms=starting_atoms,
            timestep=params["dt"] * units.fs,
            temperature_K=params["temperature"],  # K
            pressure_au=params["pressure"] * units.bar,
            taut=params["taut"] * units.fs,
            taup=params["taup"] * units.fs,
            T_init=params["T_init"],
            mask=tuple(params["barostat_mask"]),
            compressibility_au=params["compressibility"] / units.bar,
            fixcm=params["fixcm"],
            trajectory=f"{params['traj_dir']}/traj.traj",
            logfile=f"{params['traj_dir']}/md.log",
            loginterval=params["loginterval"],
            nbr_update_period=params.get("nbr_update_period", 20),
            unphysical_rmin=params.get("unphysical_rmin", None),
            unphysical_rmax=params.get("unphysical_rmax", None),
            hardcoded_nbr_list=nbr_list,
            append_trajectory=False,
        )

    else:
        raise ValueError(f"Unknown statistical ensemble: {md_ensemble}")

    if config["enhsamp"]["method"] not in ["unbiased", "unbiased_mp"]:
        dyn.attach(
            BiasedNeuralMDLogger(
                dyn,
                starting_atoms,
                f"{params['traj_dir']}/ext.log",
                header=True,
                mode="w",
            ),
            interval=params["loginterval"],
        )
    else:
        dyn.attach(
            MDLogger(
                dyn,
                starting_atoms,
                f"{params['traj_dir']}/ext.log",
                header=True,
                mode="w",
            ),
            interval=params["loginterval"],
        )

    return dyn


def run_dynamics(config: dict, force_restart: bool = False) -> None:
    if force_restart is False:
        completed = check_completion(config)
        if completed is True:
            logger(f"Simulation {config['enhsamp']['traj_dir']} already completed")
            return
    else:
        make_and_backup_dir(config["enhsamp"]["traj_dir"])
        save_config(config, f"{config['enhsamp']['traj_dir']}/config.yaml")

    # start dynamics
    with open(f"{config['enhsamp']['traj_dir']}/status.log", "w") as file:
        # get neighbor list
        if config["system"] == "ala":
            nbr_list = TopoInfo.get_nbr_list(config["openmm"]["forcefield"])
        else:
            nbr_list = None

        # load training dataset
        if config["dset"]["path"].endswith(".xyz"):
            train_dset = load_from_xyz(config["dset"]["path"])
        elif config["dset"]["path"].endswith(".pth.tar"):
            train_dset = Dataset.from_file(config["dset"]["path"])
        else:
            raise ValueError(f"Dataset type not recognized: {config['dset']['path']}")

        # load calibration dataset
        if config["dset"]["calib_path"].endswith(".xyz"):
            calib_dset = load_from_xyz(config["dset"]["calib_path"])
        elif config["dset"]["calib_path"].endswith(".pth.tar"):
            calib_dset = Dataset.from_file(
                config["dset"]["calib_path"].replace(".xyz", ".pth.tar")
            )
        else:
            raise ValueError(f"Dataset type not recognized: {config['dset']['path']}")

        # load dset for starting geometry if different from training dset
        if config["enhsamp"].get("start_geom_dset", None) is not None:
            start_geom_dset = load_from_xyz(config["enhsamp"]["start_geom_dset"])

            # load starting geometry
            starting_atoms, _ = get_starting_atom(
                dset=start_geom_dset,
                random_idx=config["enhsamp"]["start_geom_idx"],
                directed=config["model"]["directed"],
                device=config["train"]["device"],
                scaling_array=config["enhsamp"].get("scaling_array", None),
                cutoff=config["model"]["cutoff"],
                cutoff_skin=0.0,
            )
        else:
            # load starting geometry
            starting_atoms, _ = get_starting_atom(
                dset=train_dset,
                random_idx=config["enhsamp"]["start_geom_idx"],
                directed=config["model"]["directed"],
                device=config["train"]["device"],
                scaling_array=config["enhsamp"].get("scaling_array", None),
                cutoff=config["model"]["cutoff"],
                cutoff_skin=0.0,
            )

        # set up CV definition
        cv_params = get_CV_parameters(config)

        # set up calculator for relaxation
        unbiased_calculator = set_up_unbiased_calculator(config=config, model=None)

        # attach calculator to atoms object
        starting_atoms.set_calculator(unbiased_calculator)

        # relax the structure
        logger("Relaxing structure")
        relax_atoms(
            starting_atoms,
            algs=config["enhsamp"]["relax_algs"],
            nsteps=config["enhsamp"]["relax_nsteps"],
        )

        # set up calculator for enhanced sampling
        biased_calculator = set_up_biased_calculator(
            config=config,
            starting_atoms=starting_atoms,
            train_dset=train_dset,
            calib_dset=calib_dset,
            cv_params=cv_params,
            model=None,
        )

        # attach calculator to atoms object
        starting_atoms.set_calculator(biased_calculator)

        # set up dynamics
        dyn = set_up_dynamics(config, starting_atoms, nbr_list)

        logger("Running dynamics")
        start_time = time.time()

        if config["enhsamp"].get("apply_amd", False) is False:
            dyn.run(steps=config["enhsamp"]["nsteps"])
        else:
            apply_amd_after_steps = config["enhsamp"].get("apply_amd_after_steps", 10)
            if apply_amd_after_steps > config["enhsamp"]["nsteps"]:
                raise ValueError(
                    "apply_amd_after_steps should be less than nsteps in config"
                )
            dyn.run(
                steps=apply_amd_after_steps
            )  # run for 10 steps without AMD to avoid zero division error
            logger(f"Applying AMD after {apply_amd_after_steps} steps")
            dyn.atoms.calc.apply_amd = True
            dyn.run(steps=config["enhsamp"]["nsteps"] - apply_amd_after_steps)

        end_time = time.time()
        time_taken = end_time - start_time

        file.write(f"BiasedMD: Time taken: {time_taken}s\n")
        file.write("BiasedMD: Done")
        logger(f"Time taken: {time_taken}s")
        logger(f"Done at {time.ctime()}")

        file.close()


def run(config: dict, force_restart: bool = False):
    try:
        run_dynamics(config=config, force_restart=force_restart)
    except RuntimeError as e:
        if "CUDA" in str(e) and "out of memory" in str(e):
            gc.collect()
            torch.cuda.empty_cache()
            least_used_device = get_least_used_device()
            config["train"]["device"] = least_used_device
            print(f"BiasedMD: Switching to device {least_used_device}")
            run_dynamics(config=config, force_restart=force_restart)
        else:
            raise e


def make_config_for_each_unbiased_simulation(config: dict) -> List[dict]:
    md_ensemble = config["md"].get("md_ensemble", "nvt")

    num_simulations = config["md"]["num_simulations"]
    temperatures = config["md"]["temperature"]
    pressures = config["md"].get("pressure", None)
    traj_dirs = config["md"]["traj_dir"]

    # check that the number of simulations is consistent
    assert num_simulations == len(temperatures) == len(traj_dirs)
    if md_ensemble == "npt":
        assert num_simulations == len(pressures)

    # get the ensemble path
    model_outdir = get_main_model_outdir(config["model"]["outdir"])

    # create a config for each simulation
    parameters = []
    for i in range(config["md"]["num_simulations"]):
        params = deepcopy(config)

        params["model"]["outdir"] = model_outdir

        params["md"] = {}
        params["md"]["traj_dir"] = traj_dirs[i]
        params["md"]["temperature"] = temperatures[i]

        if md_ensemble == "nvt":
            params["md"]["scaling_array"] = config["md"].get(
                "scaling_array", [1, 1, 1, 1, 1, 1]
            )
        elif md_ensemble == "npt":
            params["md"]["pressure"] = pressures[i]
            params["md"]["barostat_mask"] = config["md"].get("barostat_mask", [1, 1, 1])
        else:
            raise ValueError(f"Unknown statistical ensemble: {md_ensemble}")

        for key, val in config["md"].items():
            if key in [
                "temperature",
                "pressure",
                "traj_dir",
                "scaling_array",
                "barostat_mask",
            ]:
                continue  # already set above
            elif isinstance(val, list):
                params["md"][key] = val[i]
            else:
                params["md"][key] = val

        parameters.append(params)

    return parameters


def run_unbiased_dynamics(config: dict, force_restart: bool = False) -> None:
    from ase.io import read

    params = config["md"]

    if force_restart is False:
        completed = check_completion(config)
        if completed is True:
            logger(f"Simulation {params['traj_dir']} already completed")
            return
    else:
        make_and_backup_dir(params["traj_dir"])
        save_config(config, f"{params['traj_dir']}/config.yaml")

    # start dynamics
    with open(f"{params['traj_dir']}/status.log", "w") as file:
        # get neighbor list
        if config["system"] == "ala":
            nbr_list = TopoInfo.get_nbr_list(config["openmm"]["forcefield"])
        else:
            nbr_list = None

        # load starting atoms
        starting_atoms = read(params["start_geom"])
        starting_atoms = AtomsBatch.from_atoms(
            atoms=starting_atoms,
            directed=config["model"]["directed"],
            device=config["train"]["device"],
            cutoff=config["model"]["cutoff"],
            cutoff_skin=0.0,
        )

        # set up calculator for relaxation
        calculator = set_up_unbiased_calculator(config=config, model=None)

        # attach calculator to atoms object
        starting_atoms.set_calculator(calculator)

        # set up dynamics
        md_ensemble = params.get("md_ensemble", "nvt")

        if md_ensemble == "nvt":
            dyn = Langevin(
                starting_atoms,
                timestep=params["dt"] * units.fs,
                temperature_K=params["temperature"],  # K
                friction_per_ps=params["friction_per_ps"],  # 1/ps
                T_init=params["T_init"],
                logfile=f"{params['traj_dir']}/md.log",
                trajectory=f"{params['traj_dir']}/traj.traj",
                loginterval=params["loginterval"],
                unphysical_rmin=params.get("unphysical_rmin", None),
                unphysical_rmax=params.get("unphysical_rmax", None),
                hardcoded_nbr_list=nbr_list,
                append_trajectory=False,
            )

        elif md_ensemble == "npt":
            dyn = BerendsenNPT(
                atoms=starting_atoms,
                timestep=params["dt"] * units.fs,
                temperature_K=params["temperature"],  # K
                pressure_au=params["pressure"] * units.bar,
                taut=params["taut"] * units.fs,
                taup=params["taup"] * units.fs,
                T_init=params["T_init"],
                mask=tuple(params["barostat_mask"]),
                compressibility_au=params["compressibility"] / units.bar,
                fixcm=params["fixcm"],
                trajectory=f"{params['traj_dir']}/traj.traj",
                logfile=f"{params['traj_dir']}/md.log",
                loginterval=params["loginterval"],
                nbr_update_period=params.get("nbr_update_period", 20),
                unphysical_rmin=params.get("unphysical_rmin", None),
                unphysical_rmax=params.get("unphysical_rmax", None),
                hardcoded_nbr_list=nbr_list,
                append_trajectory=False,
            )

        else:
            raise ValueError(f"Unknown statistical ensemble: {md_ensemble}")

        dyn.attach(
            MDLogger(
                dyn,
                starting_atoms,
                f"{params['traj_dir']}/ext.log",
                header=True,
                mode="w",
            ),
            interval=params["loginterval"],
        )

        logger("Running equilibration dynamics")
        start_time = time.time()
        dyn.run(steps=config["md"]["equilibration_nsteps"])
        end_time = time.time()
        time_taken = end_time - start_time
        file.write(f"UnbiasedMD: Time taken for equilibration: {time_taken}s\n")

        logger("Running production dynamics")
        start_time = time.time()
        dyn.run(steps=config["md"]["production_nsteps"])
        end_time = time.time()
        time_taken = end_time - start_time
        file.write(f"UnbiasedMD: Time taken for production: {time_taken}s\n")

        file.write("UnbiasedMD: Done")
        logger(f"Done at {time.ctime()}")

        file.close()


def handle_multiproc_enhsamp(
    config: dict, cpu_count: int = 3, force_restart: bool = False
) -> None:
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn")

    print("================================================================")
    print("BiasedMD: Running BiasedMD dynamics using uncertainty")
    print(f"BiasedMD: {time.ctime()}")

    orig_config = deepcopy(config)

    if force_restart:
        print("BiasedMD: Force restart enabled")

    # get least used device
    least_used_device = get_least_used_device()
    orig_config["train"]["device"] = least_used_device

    # create a config for each simulation
    all_configs = make_config_for_each_simulation(orig_config)

    # write config file to output directory
    save_config(orig_config, f"{orig_config['model']['outdir']}/config.yaml")

    # if force restart is not enabled, check completion of simulations
    if force_restart is False:
        # check completion of the simulations
        all_completed = []
        for config in all_configs:
            completed = check_completion(config)
            all_completed.append(completed)

        # if none of the simulations are completed, recreate base directory
        if any(all_completed) is False:
            # create base directory for traj
            make_and_backup_dir(orig_config["enhsamp"]["traj_dir"])

        # if some of the simulations are completed, recreate individual directories
        else:
            # remove completed simulations from list
            run_idx = np.argwhere(np.array(all_completed) is False).flatten()
            all_configs = [all_configs[i] for i in run_idx]
    # if force restart is enabled, recreate base directory
    else:
        make_and_backup_dir(orig_config["enhsamp"]["traj_dir"])

    # recreate individual directories
    for config in all_configs:
        make_and_backup_dir(config["enhsamp"]["traj_dir"])

    # if all simulations are completed, exit
    if len(all_configs) == 0:
        print("BiasedMD: All simulations completed")
        print("================================================================")
        return

    # run dynamics
    # create a new pool of worker processes
    with mp.Pool(cpu_count) as pool:
        # Use 'imap_unordered' to run the function concurrently for each
        # element in 'starting_atoms' and 'nbr_list'
        pool.map(run, all_configs)
        pool.close()
        pool.join()

    print("================================================================")
