import argparse
import os
import subprocess
import sys
from copy import deepcopy
from glob import glob
from typing import List, Tuple, Union

import numpy as np

sys.path.append(f"{os.getenv('PROJECTS')}/uncertainty_eABF-GaMD")
from eabfgamd.enhsamp import (
    check_completion,
    make_config_for_each_simulation,
    make_config_for_each_unbiased_simulation,
)
from eabfgamd.misc import (
    get_main_model_outdir,
    load_config,
    make_and_backup_dir,
    make_dir,
    save_config,
)
from eabfgamd.silica import Paths
from eabfgamd.train import make_config_for_each_network


def write_mace_train_job(
    config: dict,
    use_entire_node: bool = False,
) -> None:
    script_path = f"{config['model']['outdir']}/train_job.sh"
    make_dir(script_path)
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        fwrite(f"#SBATCH --job-name=train_{config['id']}_{config['gen']}")
        fwrite(f"#SBATCH --output={script_path.replace('.sh', '.log')}")
        fwrite(f"#SBATCH --error={script_path.replace('.sh', '.err')}")
        if use_entire_node is True:
            fwrite("#SBATCH --nodes=1")
        fwrite("#SBATCH --ntasks-per-node=2")
        fwrite("#SBATCH --cpus-per-task=2")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("#SBATCH --partition=xeon-g6-volta")
        fwrite("")
        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")
        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")

        model_config = config["model"]
        train_config = config["train"]
        dset_config = config["dset"]
        loss_config = config["loss"]

        config_type_weights = "'{\"Default\":1.0}'"
        fwrite(f"python $HOME/projects/mace/scripts/run_train.py {chr(92)}")
        fwrite(f"     --name='mace' {chr(92)}")
        fwrite(f"     --seed={dset_config['random_state']} {chr(92)}")
        fwrite(f"     --log_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(f"     --model_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(
            f"     --checkpoints_dir='{model_config['outdir']}/checkpoints' {chr(92)}"
        )
        fwrite(f"     --results_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(f"     --downloads_dir='{model_config['outdir']}/downloads' {chr(92)}")
        fwrite(f"     --device='{train_config['device']}' {chr(92)}")
        fwrite(f"     --default_dtype='{model_config['default_dtype']}' {chr(92)}")
        fwrite(f"     --error_table='PerAtomRMSE' {chr(92)}")
        fwrite(f"     --model='{model_config['model_type'].upper()}' {chr(92)}")
        fwrite(f"     --r_max={model_config['cutoff']} {chr(92)}")
        fwrite(f"     --radial_type='{model_config['radial_type']}' {chr(92)}")
        fwrite(f"     --num_radial_basis={model_config['num_radial_basis']} {chr(92)}")
        fwrite(f"     --num_cutoff_basis={model_config['num_cutoff_basis']} {chr(92)}")
        fwrite(f"     --interaction='{model_config['interaction']}' {chr(92)}")
        fwrite(
            f"     --interaction_first='{model_config['interaction_first']}' {chr(92)}"
        )
        fwrite(f"     --max_ell={model_config['max_ell']} {chr(92)}")
        fwrite(f"     --correlation={model_config['correlation']} {chr(92)}")
        fwrite(f"     --num_interactions={model_config['num_interactions']} {chr(92)}")
        fwrite(f"     --MLP_irreps='{model_config['MLP_irreps']}' {chr(92)}")
        fwrite(f"     --radial_MLP='{model_config['radial_MLP']}' {chr(92)}")
        fwrite(f"     --hidden_irreps='{model_config['hidden_irreps']}' {chr(92)}")
        fwrite(f"     --gate='{model_config['gate']}' {chr(92)}")
        fwrite(f"     --scaling='{model_config['scaling']}' {chr(92)}")
        fwrite(
            f"     --compute_avg_num_neighbors='{model_config['compute_avg_num_neighbors']}' {chr(92)}"
        )
        fwrite(
            f"     --avg_num_neighbors={model_config['avg_num_neighbors']} {chr(92)}"
        )
        fwrite(f"     --compute_forces='{model_config['compute_forces']}' {chr(92)}")
        fwrite(f"     --train_file='{dset_config['path']}' {chr(92)}")
        fwrite(f"     --valid_fraction={dset_config['val_size']} {chr(92)}")
        fwrite(f"     --test_file='{dset_config['test_path']}' {chr(92)}")
        fwrite(f"     --E0s='average' {chr(92)}")
        fwrite(f"     --energy_key='energy' {chr(92)}")
        fwrite(f"     --forces_key='forces' {chr(92)}")
        fwrite(f"     --virials_key='virials' {chr(92)}")
        fwrite(f"     --stress_key='stress' {chr(92)}")
        fwrite(f"     --dipole_key='dipole' {chr(92)}")
        fwrite(f"     --charges_key='charges' {chr(92)}")
        fwrite(f"     --loss='{loss_config['type']}' {chr(92)}")
        fwrite(f"     --forces_weight={loss_config['forces_coef']} {chr(92)}")
        fwrite(f"     --energy_weight={loss_config['energy_coef']} {chr(92)}")
        fwrite(f"     --virials_weight=1.0 {chr(92)}")
        fwrite(f"     --stress_weight=1.0 {chr(92)}")
        fwrite(f"     --dipole_weight=1.0 {chr(92)}")
        fwrite(f"     --config_type_weights={config_type_weights} {chr(92)}")
        fwrite(f"     --huber_delta=0.01 {chr(92)}")
        fwrite(f"     --optimizer='{train_config['optimizer']}' {chr(92)}")
        fwrite(f"     --batch_size={dset_config['batch_size']} {chr(92)}")
        fwrite(f"     --valid_batch_size={dset_config['batch_size']} {chr(92)}")
        fwrite(f"     --lr={train_config['lr']} {chr(92)}")
        fwrite(f"     --weight_decay={train_config['min_lr']} {chr(92)}")
        fwrite(f"     --scheduler='{train_config['scheduler']}' {chr(92)}")
        fwrite(f"     --lr_factor={train_config['lr_factor']} {chr(92)}")
        fwrite(f"     --scheduler_patience={train_config['patience']} {chr(92)}")
        fwrite(f"     --lr_scheduler_gamma=0.9993 {chr(92)}")
        fwrite(f"     --max_num_epochs={train_config['n_epochs']} {chr(92)}")
        fwrite(f"     --patience=200 {chr(92)}")
        fwrite(f"     --eval_interval={train_config['checkpoint_interval']} {chr(92)}")
        fwrite(f"     --clip_grad={train_config['clip_grad']} {chr(92)}")
        if train_config["swa"] is True:
            fwrite(f"     --swa {chr(92)}")
            fwrite(f"     --start_swa={train_config['start_swa']} {chr(92)}")
            fwrite(f"     --swa_lr=0.001 {chr(92)}")
            fwrite(f"     --swa_forces_weight=100.0 {chr(92)}")
            fwrite(f"     --swa_energy_weight=1000.0 {chr(92)}")
            fwrite(f"     --swa_virials_weight=10.0 {chr(92)}")
            fwrite(f"     --swa_stress_weight=10.0 {chr(92)}")
            fwrite(f"     --swa_dipole_weight=1.0 {chr(92)}")
        if train_config["ema"] is True:
            fwrite(f"     --ema {chr(92)}")
            fwrite(f"     --ema_decay=0.99 {chr(92)}")
        if train_config["amsgrad"] is True:
            fwrite(f"     --amsgrad {chr(92)}")
        if train_config["save_cpu"] is True:
            fwrite(f"     --save_cpu {chr(92)}")
        if train_config["keep_checkpoints"] is True:
            fwrite(f"     --keep_checkpoints {chr(92)}")
        if train_config["restart_latest"] is True:
            fwrite(f"     --restart_latest {chr(92)}")

        fwrite("     --log_level='INFO'")
        f.close()

    return script_path


def write_mace_finetune_job(
    config: dict,
    use_entire_node: bool = False,
) -> None:
    script_path = f"{config['model']['outdir']}/finetune_job.sh"
    make_dir(script_path)
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        fwrite(f"#SBATCH --job-name=finetune_{config['id']}_{config['gen']}")
        fwrite(f"#SBATCH --output={script_path.replace('.sh', '.log')}")
        fwrite(f"#SBATCH --error={script_path.replace('.sh', '.err')}")
        if use_entire_node is True:
            fwrite("#SBATCH --nodes=1")
        fwrite("#SBATCH --ntasks-per-node=2")
        fwrite("#SBATCH --cpus-per-task=2")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("#SBATCH --partition=xeon-g6-volta")
        fwrite("")
        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")
        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")

        model_config = config["model"]
        train_config = config["train"]
        dset_config = config["dset"]
        loss_config = config["loss"]

        fwrite(f"python $HOME/projects/mace/scripts/run_train.py {chr(92)}")
        fwrite(f"   --name='mace' {chr(92)}")
        fwrite(f"   --foundation_model='small' {chr(92)}")
        fwrite(f"   --log_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(f"   --model_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(f"   --checkpoints_dir='{model_config['outdir']}/checkpoints' {chr(92)}")
        fwrite(f"   --results_dir='{model_config['outdir']}' {chr(92)}")
        fwrite(f"   --downloads_dir='{model_config['outdir']}/downloads' {chr(92)}")
        #        fwrite(f"   --multiheads_finetuning=False {chr(92)}")
        fwrite(f"   --train_file='{dset_config['path']}' {chr(92)}")
        fwrite(f"   --valid_fraction={dset_config['val_size']} {chr(92)}")
        fwrite(f"   --test_file='{dset_config['test_path']}' {chr(92)}")
        fwrite(f"   --energy_weight={loss_config['energy_coef']} {chr(92)}")
        fwrite(f"   --forces_weight={loss_config['forces_coef']} {chr(92)}")
        fwrite(f"   --loss='{loss_config['type']}' {chr(92)}")
        fwrite(f"   --E0s='average' {chr(92)}")
        fwrite(f"   --compute_forces='{model_config['compute_forces']}' {chr(92)}")
        fwrite(f"   --lr={train_config['lr']} {chr(92)}")
        fwrite(f"   --scaling='{model_config['scaling']}' {chr(92)}")
        fwrite(f"   --batch_size={dset_config['batch_size']} {chr(92)}")
        fwrite(f"   --max_num_epochs={train_config['n_epochs']} {chr(92)}")
        fwrite(f"   --weight_decay={train_config['min_lr']} {chr(92)}")
        fwrite(f"   --scheduler='{train_config['scheduler']}' {chr(92)}")
        fwrite(f"   --lr_factor={train_config['lr_factor']} {chr(92)}")
        fwrite(f"   --scheduler_patience={train_config['patience']} {chr(92)}")
        fwrite(f"   --lr_scheduler_gamma=0.9993 {chr(92)}")
        fwrite(f"   --patience=200 {chr(92)}")
        fwrite(f"   --eval_interval={train_config['checkpoint_interval']} {chr(92)}")
        fwrite(f"   --clip_grad={train_config['clip_grad']} {chr(92)}")
        fwrite(f"   --ema {chr(92)}")
        fwrite(f"   --ema_decay=0.99 {chr(92)}")
        fwrite(f"   --swa {chr(92)}")
        fwrite(f"   --start_swa={train_config['start_swa']} {chr(92)}")
        fwrite(f"   --swa_lr=0.001 {chr(92)}")
        fwrite(f"   --swa_forces_weight=100.0 {chr(92)}")
        fwrite(f"   --swa_energy_weight=1000.0 {chr(92)}")
        fwrite(f"   --swa_virials_weight=10.0 {chr(92)}")
        fwrite(f"   --swa_stress_weight=10.0 {chr(92)}")
        fwrite(f"   --swa_dipole_weight=1.0 {chr(92)}")
        fwrite(f"   --amsgrad {chr(92)}")
        fwrite(f"   --default_dtype='{model_config['default_dtype']}' {chr(92)}")
        fwrite(f"   --device='{train_config['device']}' {chr(92)}")
        fwrite(f"   --seed={dset_config['random_state']}")
        f.close()

    return script_path


def write_enhsamp_job(
    config: dict,
    config_path: str,
    force_restart: bool = False,
    use_entire_node: bool = False,
):
    script_path = f"{config['enhsamp']['traj_dir']}/enhsamp_job.sh"
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        if config["enhsamp"]["md_ensemble"] == "nvt":
            fwrite(
                f"#SBATCH --job-name={int(config['enhsamp']['temperature'])}K_{config['id']}_{config['gen']}"
            )
        elif config["enhsamp"]["md_ensemble"] == "npt":
            P = int(config["enhsamp"]["pressure"] * 0.0001)  # convert from bar to GPa
            fwrite(
                f"#SBATCH --job-name={int(config['enhsamp']['temperature'])}K_{P}GPa_{config['id']}_{config['gen']}"
            )
        else:
            raise ValueError(
                f"Unknown ensemble type {config['enhsamp']['md_ensemble']}"
            )
        fwrite(f"#SBATCH --output={config['enhsamp']['traj_dir']}/enhsamp_job.log")
        fwrite(f"#SBATCH --error={config['enhsamp']['traj_dir']}/enhsamp_job.err")
        if use_entire_node is True:
            fwrite("#SBATCH --nodes=1")
        fwrite("#SBATCH --ntasks=4")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("#SBATCH --partition=xeon-g6-volta")
        fwrite("")
        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")
        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")
        fwrite(f"config_path={config_path}")
        fwrite(
            f"script={os.getenv('PROJECTS')}/uncertainty_eABF-GaMD/scripts/run_enhsamp.py"
        )
        fwrite("")
        if force_restart is False:
            fwrite("python $script $config_path --individual")
        else:
            fwrite("python $script $config_path --individual --force_restart")

        f.close()

    return script_path


def write_unbiased_dyn_job(
    config: dict,
    config_path: str,
    force_restart: bool = False,
    use_entire_node: bool = False,
):
    script_path = f"{config['md']['traj_dir']}/md_job.sh"
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        if config["md"]["md_ensemble"] == "nvt":
            fwrite(
                f"#SBATCH --job-name={int(config['md']['temperature'])}K_{config['id']}_{config['gen']}"
            )
        elif config["md"]["md_ensemble"] == "npt":
            P = int(config["md"]["pressure"] * 0.0001)  # convert from bar to GPa
            fwrite(
                f"#SBATCH --job-name={int(config['md']['temperature'])}K_{P}GPa_{config['id']}_{config['gen']}"
            )
        else:
            raise ValueError(f"Unknown ensemble type {config['md']['md_ensemble']}")
        fwrite(f"#SBATCH --output={config['md']['traj_dir']}/md_job.log")
        fwrite(f"#SBATCH --error={config['md']['traj_dir']}/md_job.err")
        if use_entire_node is True:
            fwrite("#SBATCH --nodes=1")
        fwrite("#SBATCH --ntasks=4")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("#SBATCH --partition=xeon-g6-volta")
        fwrite("")
        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")
        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")
        fwrite(f"config_path={config_path}")
        fwrite(
            f"script={os.getenv('PROJECTS')}/uncertainty_eABF-GaMD/scripts/run_unbiased_dyn.py"
        )
        fwrite("")
        if force_restart is False:
            fwrite("python $script $config_path")
        else:
            fwrite("python $script $config_path --force_restart")

        f.close()

    return script_path


def write_calc_job(config: dict, config_path: str):
    main_outdir = get_main_model_outdir(config["model"]["outdir"])
    script_path = f"{os.path.dirname(main_outdir)}/calc_job.sh"
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        fwrite(f"#SBATCH --job-name=calc_{config['id']}_{config['gen']}")
        fwrite(f"#SBATCH --output={script_path.replace('.sh', '.log')}")
        fwrite(f"#SBATCH --error={script_path.replace('.sh', '.err')}")
        fwrite("#SBATCH --ntasks=4")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("")

        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")

        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")

        fwrite(
            f"PROJDIR={os.getenv('PROJECTS')}/uncertainty_eABF-GaMD/scripts"
        )
        fwrite(f"config_path={config_path}")
        fwrite("")

        fwrite(f"python $PROJDIR/sample_points.py $config_path && {chr(92)}")
        fwrite(f"python $PROJDIR/single_point_calc.py $config_path && {chr(92)}")
        fwrite(f"python $PROJDIR/save_new_data.py $config_path && {chr(92)}")
        fwrite(
            f"mkdir -pv {os.path.dirname(config_path.replace('inbox', 'completed'))} && {chr(92)}"
        )
        fwrite(f"mv {config_path} {config_path.replace('inbox', 'completed')}")

        f.close()

    return script_path


def write_aggregate_job(config: dict, config_path: str):
    main_outdir = get_main_model_outdir(config["model"]["outdir"])
    script_path = f"{os.path.dirname(main_outdir)}/agg_job.sh"
    make_dir(script_path)
    with open(script_path, "w") as f:

        def fwrite(x):
            f.write(x + "\n")

        fwrite("#!/bin/bash")
        fwrite("")
        fwrite(f"#SBATCH --job-name=agg_{config['id']}_{config['gen']}")
        fwrite(f"#SBATCH --output={script_path.replace('.sh', '.log')}")
        fwrite(f"#SBATCH --error={script_path.replace('.sh', '.err')}")
        fwrite("#SBATCH --ntasks=4")
        fwrite("#SBATCH --gres=gpu:volta:1")
        fwrite("")
        fwrite("source /etc/profile")
        fwrite("source $HOME/.bashrc")
        fwrite("")
        fwrite("module unload anaconda")
        fwrite("module load anaconda/2023a-pytorch")
        fwrite("")
        fwrite(f"config_path={config_path}")
        fwrite(
            f"script={os.getenv('PROJECTS')}/uncertainty_eABF-GaMD/scripts/get_ensemble.py"
        )
        fwrite("")

        fwrite("python $script $config_path")

        f.close()

    return script_path


def handle_train(
    config: dict,
    restore: bool = False,
    force_retrain: bool = False,
    use_entire_node: bool = False,
) -> Tuple[List, List]:
    orig_config = deepcopy(config)
    config_files = make_config_for_each_network(orig_config)
    script_paths = []
    for config_file in config_files:
        if (
            force_retrain is True
            or os.path.exists(config_file["model"]["outdir"]) is False
        ):
            make_and_backup_dir(config_file["model"]["outdir"])

        if config_file["model"]["model_type"] == "mace":
            script_path = write_mace_train_job(config_file, use_entire_node)
        elif config_file["model"]["model_type"] == "mace_mp":
            script_path = write_mace_finetune_job(config_file, use_entire_node)

        script_paths.append(script_path)

    return script_paths


def handle_enhsamp(
    config: dict, force_restart: bool = False, use_entire_node: bool = False
) -> Tuple[List, List]:
    # create a config for each simulation
    orig_config = deepcopy(config)
    config_files = make_config_for_each_simulation(orig_config)
    script_paths = []
    for config_file in config_files:
        # if force restart is enabled, just recreate the directory without checking
        if force_restart is True:
            make_and_backup_dir(config_file["enhsamp"]["traj_dir"])
        # if force restart is not enabled, check completion of simulations
        else:
            # check completion of the simulations
            completed = check_completion(config_file)
            # if simulation is not done, just recreate a new directory
            if completed is False:
                make_and_backup_dir(config_file["enhsamp"]["traj_dir"])

        config_path = f"{config_file['enhsamp']['traj_dir']}/config.yaml"
        save_config(config_file, config_path)

        script_path = write_enhsamp_job(
            config_file, config_path, force_restart, use_entire_node
        )
        script_paths.append(script_path)

    return script_paths


def handle_unbiased_dynamics(
    config: dict, force_restart: bool = False, use_entire_node: bool = False
) -> Tuple[List, List]:
    # create a config for each simulation
    orig_config = deepcopy(config)
    config_files = make_config_for_each_unbiased_simulation(orig_config)
    script_paths = []
    for config_file in config_files:
        # if force restart is enabled, just recreate the directory without checking
        if force_restart is True:
            make_and_backup_dir(config_file["md"]["traj_dir"])
        # if force restart is not enabled, check completion of simulations
        else:
            # check completion of the simulations
            completed = check_completion(config_file)
            # if simulation is not done, just recreate a new directory
            if completed is False:
                make_and_backup_dir(config_file["md"]["traj_dir"])

        config_path = f"{config_file['md']['traj_dir']}/config.yaml"
        save_config(config_file, config_path)

        script_path = write_unbiased_dyn_job(
            config_file, config_path, force_restart, use_entire_node
        )
        script_paths.append(script_path)

    return script_paths


def submit_job(
    script_path: str, dependent_job_id: Union[int, List[int], None] = None
) -> int:
    """
    Submit a job to the Slurm scheduler and return the job ID.
    If dependent_job_id is given, then the job is submitted with a dependency
    on another job's completion.
    """
    if dependent_job_id is None:
        result = subprocess.run(
            ["sbatch", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    else:
        if isinstance(dependent_job_id, int):
            result = subprocess.run(
                ["sbatch", f"--dependency=afterok:{dependent_job_id}", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        elif isinstance(dependent_job_id, list):
            ids = ",".join(map(str, dependent_job_id))
            result = subprocess.run(
                ["sbatch", f"--dependency=afterok:{ids}", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        else:
            raise ValueError("Dependent_job_id is not type int or list")

    if result.returncode == 0:
        # Extract the job ID from the output and return it
        return int(result.stdout.strip().split()[-1])
    else:
        raise RuntimeError(f"Job submission failed: {result.stderr}")


def submit_individual_job(
    config_path: str,
    job_types: List[str],
    train_restore: bool,
    force_retrain: bool,
    md_force_restart: bool,
    use_entire_node: bool,
    dependent_job_id: Union[int, List[int], None] = None,
    dry_run: bool = False,
):
    # load config first as a dictionary
    config = load_config(config_path)

    for job_type in job_types:
        assert job_type in ["train", "agg", "enhsamp", "calc", "unbiased_dyn"]

        # write training job script
        if job_type == "train":
            script_paths = handle_train(
                config,
                restore=train_restore,
                force_retrain=force_retrain,
                use_entire_node=use_entire_node,
            )

            if dry_run is False:
                for script_path in script_paths:
                    job_id = submit_job(
                        script_path=script_path, dependent_job_id=dependent_job_id
                    )
                    print(f"Submitted train {script_path} with JOB ID {job_id}")
            else:
                for script_path in script_paths:
                    print(f"DRYRUN: Would have submitted train {script_path}")

        elif job_type == "agg":
            script_path = write_aggregate_job(
                config,
                config_path,
            )
            if dry_run is False:
                job_id = submit_job(
                    script_path=script_path, dependent_job_id=dependent_job_id
                )
                print(f"Submitted agg {script_path} with JOB ID {job_id}")
            else:
                print(f"DRYRUN: Would have submitted agg {script_path}")

        # run enhanced sampling trajectories
        elif job_type == "enhsamp":
            script_paths = handle_enhsamp(
                config,
                force_restart=md_force_restart,
                use_entire_node=use_entire_node,
            )
            if dry_run is False:
                job_ids = [
                    submit_job(script_path, dependent_job_id=dependent_job_id)
                    for script_path in script_paths
                ]
                for script_path, job_id in zip(script_paths, job_ids):
                    print(f"Submitted enhsamp {script_path} with JOB ID {job_id}")
            else:
                for script_path in script_paths:
                    print(f"DRYRUN: Would have submitted enhsamp {script_path}")

        # run calc job
        elif job_type == "calc":
            script_path = write_calc_job(
                config,
                config_path,
            )
            if dry_run is False:
                job_id = submit_job(
                    script_path=script_path, dependent_job_id=dependent_job_id
                )
                print(f"Submitted calc {script_path} with JOB ID {job_id}")
            else:
                print(f"DRYRUN: Would have submitted calc {script_path}")

        elif job_type == "unbiased_dyn":
            script_paths = handle_unbiased_dynamics(
                config,
                force_restart=md_force_restart,
                use_entire_node=use_entire_node,
            )
            if dry_run is False:
                job_ids = [
                    submit_job(script_path, dependent_job_id=dependent_job_id)
                    for script_path in script_paths
                ]
                for script_path, job_id in zip(script_paths, job_ids):
                    print(
                        f"Submitted unbiased dynamics {script_path} with JOB ID {job_id}"
                    )
            else:
                for script_path in script_paths:
                    print(f"DRYRUN: Would have submitted dynamics {script_path}")

        # get main directory of the run and save config
        model_outdir = get_main_model_outdir(config["model"]["outdir"])
        main_dir = os.path.dirname(model_outdir)
        if not os.path.exists(f"{main_dir}/config.yaml"):
            save_config(config, f"{main_dir}/config.yaml")
            print(f"Saved main config to {main_dir}/config.yaml")

    return


def submit_continuous_dependent_jobs(
    config_path: str,
    job_types: List[str],
    train_restore: bool,
    force_retrain: bool,
    md_force_restart: bool,
    use_entire_node: bool,
    dependent_job_id: Union[int, List[int], None] = None,
    dry_run: bool = False,
):
    # load config first as a dictionary
    config = load_config(config_path)

    possible_job_types = ["train", "agg", "enhsamp", "calc"]
    for i, job_type in enumerate(job_types):
        # ensure that the job type is valid
        assert job_type in possible_job_types
        # ensure that job_types are continuous
        if i > 0:
            assert (
                possible_job_types.index(job_type)
                - possible_job_types.index(job_types[i - 1])
                == 1
            )

    for job_type in job_types:
        # write training job script
        if job_type == "train":
            script_paths = handle_train(
                config,
                restore=train_restore,
                force_retrain=force_retrain,
                use_entire_node=use_entire_node,
            )

            if dry_run is False:
                for script_path in script_paths:
                    dependent_job_id = submit_job(
                        script_path=script_path, dependent_job_id=dependent_job_id
                    )
                    print(
                        f"Submitted train {script_path} with JOB ID {dependent_job_id}"
                    )
            else:
                for script_path in script_paths:
                    print(f"DRYRUN: Would have submitted train {script_path}")

        elif job_type == "agg":
            script_path = write_aggregate_job(
                config,
                config_path,
            )
            if dry_run is False:
                dependent_job_id = submit_job(
                    script_path=script_path, dependent_job_id=dependent_job_id
                )
                print(f"Submitted agg {script_path} with JOB ID {dependent_job_id}")
            else:
                print(f"DRYRUN: Would have submitted agg {script_path}")

        # run enhanced sampling trajectories
        elif job_type == "enhsamp":
            script_paths = handle_enhsamp(
                config,
                force_restart=md_force_restart,
                use_entire_node=use_entire_node,
            )
            if dry_run is False:
                dependent_job_id = [
                    submit_job(script_path, dependent_job_id=dependent_job_id)
                    for script_path in script_paths
                ]
                for script_path, job_id in zip(script_paths, dependent_job_id):
                    print(f"Submitted enhsamp {script_path} with JOB ID {job_id}")
            else:
                for script_path in script_paths:
                    print(f"DRYRUN: Would have submitted enhsamp {script_path}")

        # run calc job
        elif job_type == "calc":
            script_path = write_calc_job(
                config,
                config_path,
            )
            if dry_run is False:
                dependent_job_id = submit_job(
                    script_path=script_path, dependent_job_id=dependent_job_id
                )
                print(f"Submitted calc {script_path} with JOB ID {dependent_job_id}")
            else:
                print(f"DRYRUN: Would have submitted calc {script_path}")

        # get main directory of the run and save config
        model_outdir = get_main_model_outdir(config["model"]["outdir"])
        main_dir = os.path.dirname(model_outdir)
        if not os.path.exists(f"{main_dir}/config.yaml"):
            save_config(config, f"{main_dir}/config.yaml")
            print(f"Saved main config to {main_dir}/config.yaml")

    return


def pipeline(
    config_paths: List[str],
    train_restore: bool,
    force_retrain: bool,
    md_force_restart: bool,
    use_entire_node: bool,
    dry_run: bool = False,
) -> None:
    dependent_job_id = None
    for config_path in config_paths:
        # load config first as a dictionary
        config = load_config(config_path)

        # write training job script
        script_paths = handle_train(
            config,
            restore=train_restore,
            force_retrain=force_retrain,
        )
        if dry_run is False:
            dependent_job_id = [
                submit_job(script_path, dependent_job_id)
                for script_path in script_paths
            ]
            for script_path, job_id in zip(script_paths, dependent_job_id):
                print(f"Submitted {script_path} with JOB ID {job_id}")
        else:
            for script_path in script_paths:
                print(f"DRYRUN: Would have submitted {script_path}")

        # write aggregating network into ensemble job
        script_path = write_aggregate_job(
            config,
            config_path,
        )
        if dry_run is False:
            dependent_job_id = submit_job(
                script_path=script_path,
                dependent_job_id=dependent_job_id,
            )
            print(f"Submitted {script_path} with JOB ID {dependent_job_id}")
        else:
            print(f"DRYRUN: Would have submitted {script_path}")

        # run enhanced sampling trajectories
        script_paths = handle_enhsamp(
            config,
            force_restart=md_force_restart,
            use_entire_node=use_entire_node,
        )
        if dry_run is False:
            dependent_job_id = [
                submit_job(script_path, dependent_job_id)
                for script_path in script_paths
            ]
            for script_path, job_id in zip(script_paths, dependent_job_id):
                print(f"Submitted {script_path} with JOB ID {job_id}")
        else:
            for script_path in script_paths:
                print(f"DRYRUN: Would have submitted {script_path}")

        # run calc job
        script_path = write_calc_job(
            config,
            config_path,
        )
        if dry_run is False:
            dependent_job_id = submit_job(
                script_path=script_path,
                dependent_job_id=dependent_job_id,
            )
            print(f"Submitted {script_path} with JOB ID {dependent_job_id}")
        else:
            print(f"DRYRUN: Would have submitted {script_path}")

        # get main directory of the run and save config
        model_outdir = get_main_model_outdir(config["model"]["outdir"])
        main_dir = os.path.dirname(model_outdir)
        if not os.path.exists(f"{main_dir}/config.yaml"):
            save_config(config, f"{main_dir}/config.yaml")
            print(f"Saved main config to {main_dir}/config.yaml")

    if dry_run is False:
        print("All jobs submitted successfully.")
    else:
        print("DRYRUN: All jobs would have been submitted.")


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Subparser to do pipeline
    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument("model_ids", nargs="+", type=str)
    pipeline_parser.add_argument(
        "--use_entire_node", action="store_true", default=False
    )
    pipeline_parser.add_argument("--train_restore", action="store_true", default=False)
    pipeline_parser.add_argument("--force_retrain", action="store_true", default=False)
    pipeline_parser.add_argument(
        "--md_force_restart", action="store_true", default=False
    )
    pipeline_parser.add_argument("--only_generations", nargs="*", type=int, default=[])
    pipeline_parser.add_argument("--dry_run", action="store_true", default=False)

    # Subparser for individual job submission
    individual_parser = subparsers.add_parser("individual")
    individual_parser.add_argument("config_path", type=str)
    individual_parser.add_argument("--train", action="store_true", default=False)
    individual_parser.add_argument("--agg", action="store_true", default=False)
    individual_parser.add_argument("--enhsamp", action="store_true", default=False)
    individual_parser.add_argument("--calc", action="store_true", default=False)
    individual_parser.add_argument("--unbiased_dyn", action="store_true", default=False)
    individual_parser.add_argument("--all", action="store_true", default=False)
    individual_parser.add_argument(
        "--use_entire_node", action="store_true", default=False
    )
    individual_parser.add_argument(
        "--train_restore", action="store_true", default=False
    )
    individual_parser.add_argument(
        "--force_retrain", action="store_true", default=False
    )
    individual_parser.add_argument(
        "--md_force_restart", action="store_true", default=False
    )
    individual_parser.add_argument("--dependent_job_id", "-id", nargs="*", type=int)
    individual_parser.add_argument("--dry_run", action="store_true", default=False)

    # Subparser for continuous job submission dependent on previous job
    continuous_parser = subparsers.add_parser("continuous")
    continuous_parser.add_argument("config_path", type=str)
    continuous_parser.add_argument("--train", action="store_true", default=False)
    continuous_parser.add_argument("--agg", action="store_true", default=False)
    continuous_parser.add_argument("--enhsamp", action="store_true", default=False)
    continuous_parser.add_argument("--calc", action="store_true", default=False)
    continuous_parser.add_argument("--all", action="store_true", default=False)
    continuous_parser.add_argument(
        "--use_entire_node", action="store_true", default=False
    )
    continuous_parser.add_argument(
        "--train_restore", action="store_true", default=False
    )
    continuous_parser.add_argument(
        "--force_retrain", action="store_true", default=False
    )
    continuous_parser.add_argument(
        "--md_force_restart", action="store_true", default=False
    )
    continuous_parser.add_argument("--dependent_job_id", "-id", nargs="*", type=int)
    continuous_parser.add_argument("--dry_run", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "pipeline":
        submitted_model_ids = []
        for model_id in args.model_ids:
            config_paths = glob(f"{Paths.inbox_params_dir}/{model_id}_*")

            generations = [
                int(f.split("/")[-1].split("_")[-1].split(".")[0]) for f in config_paths
            ]
            inds = np.argsort(np.array(generations))

            if len(args.only_generations) > 0:
                inds = [i for i in inds if generations[i] in args.only_generations]

            config_paths = [config_paths[i] for i in inds]

            if len(config_paths) == 0:
                print("=================================================")
                print(f"No config files found for {model_id}")
                print("=================================================")
                continue

            print("=================================================")
            print(f"Submitting jobs for {model_id}")
            print("=================================================")
            pipeline(
                config_paths=config_paths,
                train_restore=args.train_restore,
                force_retrain=args.force_retrain,
                md_force_restart=args.md_force_restart,
                use_entire_node=args.use_entire_node,
                dry_run=args.dry_run,
            )
            submitted_model_ids.append(model_id)

        if len(submitted_model_ids) > 0 and args.dry_run is False:
            print("=================================================")
            print("Done submitting for all!")
            print("=================================================")

    elif args.command == "individual":
        job_types = []
        if args.train:
            job_types.append("train")
        if args.agg:
            job_types.append("agg")
        if args.enhsamp:
            job_types.append("enhsamp")
        if args.calc:
            job_types.append("calc")
        if args.unbiased_dyn:
            job_types.append("unbiased_dyn")
        if args.all:
            job_types = ["train", "agg", "enhsamp", "calc"]

        submit_individual_job(
            config_path=args.config_path,
            job_types=job_types,
            train_restore=args.train_restore,
            force_retrain=args.force_retrain,
            md_force_restart=args.md_force_restart,
            use_entire_node=args.use_entire_node,
            dependent_job_id=args.dependent_job_id,
            dry_run=args.dry_run,
        )

    elif args.command == "continuous":
        job_types = []
        if args.train:
            job_types.append("train")
        if args.agg:
            job_types.append("agg")
        if args.enhsamp:
            job_types.append("enhsamp")
        if args.calc:
            job_types.append("calc")
        if args.all:
            job_types = ["train", "agg", "enhsamp", "calc"]

        submit_continuous_dependent_jobs(
            config_path=args.config_path,
            job_types=job_types,
            train_restore=args.train_restore,
            force_retrain=args.force_retrain,
            md_force_restart=args.md_force_restart,
            use_entire_node=args.use_entire_node,
            dependent_job_id=args.dependent_job_id,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    argument_parser()
