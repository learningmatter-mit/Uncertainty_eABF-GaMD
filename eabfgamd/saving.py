import torch
from typing import List
from nff.data import Dataset, concatenate_dict
from .misc import (
    load_from_xyz,
    is_energy_valid,
    make_dir,
    get_dihedrals,
    get_atoms,
    attach_data_to_atoms,
    get_logger,
    write_atoms_list_to_xyzfile,
)
from .ala import TopoInfo


__all__ = [
    "combine_datasets",
    "handle_save",
]


logger = get_logger("SAVING")


def combine_datasets(
    all_dsets: List[Dataset],
    necessary_keys: List[str] = [
        "nxyz",
        "dihedrals",
        "nbr_list",
        "num_atoms",
        "identifier",
        "energy",
        "energy_grad",
    ],
) -> Dataset:
    for dset in all_dsets:
        assert all([key in dset.props.keys() for key in necessary_keys])

    # remove unnecessary keys
    for i, dset in enumerate(all_dsets):
        for key in set(dset.props.keys()) - set(necessary_keys):
            del all_dsets[i].props[key]

    # combine
    combined_dset = []
    for dset in all_dsets:
        combined_dset.extend(list(dset))

    combined_dset = Dataset(concatenate_dict(*combined_dset))

    return combined_dset


def get_next_dset_path(config: dict) -> str:
    if config["system"] == "ala":
        from .ala import Paths
    elif config["system"] == "silica":
        from .silica import Paths
    else:
        raise Exception("System not supported")

    next_dset_path = f"{Paths.results_dir}/{config['model']['model_type']}_{config['uncertainty']['type']}/{config['id']}/gen{int(config['gen']+1)}/dataset"

    return next_dset_path


def handle_save(config: dict) -> Dataset:
    logger("Reading datasets")
    next_dset = Dataset.from_file(config["enhsamp"]["sampled_path"])

    # remove geoms with unphysical energies
    logger("Removing unphysical energies")

    next_atoms_list, valid_inds = [], []
    for i, d in enumerate(next_dset):
        if (
            is_energy_valid(
                d["energy"].item(),
                min_energy=config["enhsamp"]["unphysical_emin"],
                max_energy=config["enhsamp"]["unphysical_emax"],
            )
            is False
        ):
            continue

        atoms = get_atoms(d)
        atoms = attach_data_to_atoms(
            atoms,
            energy=d["energy"].numpy(),
            forces=-d["energy_grad"].numpy(),
            identifier=d["identifier"],
        )
        next_atoms_list.append(atoms)
        valid_inds.append(i)

    next_dset = [next_dset[i] for i in valid_inds]

    assert len(next_dset) == len(
        next_atoms_list), "Mismatch in dataset lengths"

    # if no valid geometries are found, exit
    if len(next_dset) == 0:
        print("No valid geometries found. Exiting.")
        return

    if config["system"] == "ala":
        nbr_list = TopoInfo.get_nbr_list(
            config["openmm"].get("forcefield", None))

        logger("Standardizing datasets")
        if "nbr_list" not in next_dset.props.keys():
            next_dset.props["nbr_list"] = [
                torch.LongTensor(nbr_list)] * len(next_dset)

        if "dihedrals" not in next_dset.props.keys():
            dihedral_values = []
            dihedral_inds = TopoInfo.get_dihedral_inds(
                ff=config["openmm"]["forcefield"],
                dihedrals=["phi", "psi", "omega_1", "omega_2"],
            )

            for atoms in next_atoms_list:
                dihed = get_dihedrals(atoms, dihedral_inds)
                atoms.info["dihedrals"] = dihed

                dihed = torch.FloatTensor([dihed])
                dihedral_values.append(dihed)

            next_dset.props["dihedrals"] = dihedral_values

    logger("Combining datasets")
    next_dset_path = get_next_dset_path(config)

    if config["dset"]["path"].endswith(".xyz"):
        # get the save path to the next generation dataset
        next_dset_path = f"{next_dset_path}.xyz"
        make_dir(next_dset_path)

        prev_atoms_list = load_from_xyz(config["dset"]["path"])

        # remove attached calculator just in case values are overwritten
        for i, at in enumerate(prev_atoms_list):
            at.calc = None

        combined_dset = list(prev_atoms_list) + next_atoms_list

        # split and save dataset if calib_frac is provided
        if config["dset"].get("calib_frac", None) is not None:
            logger("Splitting dataset into calibration and training sets")
            calib_frac = config["dset"]["calib_frac"]
            calib_size = int(len(combined_dset) * calib_frac)
            calib_inds = torch.randperm(len(combined_dset))[:calib_size]
            train_dset = [
                combined_dset[i]
                for i in range(len(combined_dset))
                if i not in calib_inds
            ]
            calib_dset = [combined_dset[i] for i in calib_inds]

            logger(
                f"Train size: {len(train_dset)}; Calib size: {len(calib_dset)}")

            # save dataset
            write_atoms_list_to_xyzfile(
                train_dset, next_dset_path.replace(
                    "dataset.xyz", "train_dset.xyz")
            )
            write_atoms_list_to_xyzfile(
                calib_dset, next_dset_path.replace(
                    "dataset.xyz", "calib_dset.xyz")
            )
            logger(
                f"Saved train and calib datasets to {next_dset_path.replace('dataset.xyz', '')}"
            )

        # save dataset as a whole
        else:
            # save dataset
            write_atoms_list_to_xyzfile(combined_dset, next_dset_path)
            logger(f"Saved dataset to {next_dset_path}")

    elif config["dset"]["path"].endswith(".pth.tar"):
        # get the save path to the next generation dataset
        next_dset_path = f"{next_dset_path}.pth.tar"
        make_dir(next_dset_path)

        next_dset = Dataset(concatenate_dict(*next_dset))
        prev_dset = Dataset.from_file(config["dset"]["path"])
        combined_dset = combine_datasets([prev_dset, next_dset])

        # split and save dataset if calib_frac is provided
        if config["dset"].get("calib_frac", None) is not None:
            logger("Splitting dataset into calibration and training sets")
            calib_frac = config["dset"]["calib_frac"]
            calib_size = int(len(combined_dset) * calib_frac)
            calib_inds = torch.randperm(len(combined_dset))[:calib_size]
            train_dset = [
                combined_dset[i]
                for i in range(len(combined_dset))
                if i not in calib_inds
            ]
            calib_dset = [combined_dset[i] for i in calib_inds]

            train_dset = Dataset(concatenate_dict(*train_dset))
            calib_dset = Dataset(concatenate_dict(*calib_dset))

            # save dataset
            train_dset.save(
                next_dset_path.replace("dataset.pth.tar", "train_dset.pth.tar")
            )
            calib_dset.save(
                next_dset_path.replace("dataset.pth.tar", "calib_dset.pth.tar")
            )
            logger(
                f"Saved train and calib datasets to {next_dset_path.replace('dataset.pth.tar', '')}"
            )

        else:
            # save dataset as a whole
            combined_dset.save(next_dset_path)
            logger(f"Saved dataset to {next_dset_path}")

    else:
        raise Exception("Dataset format not supported")

    return combined_dset
