from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import Trajectory, write
from nff.data import Dataset, concatenate_dict

from .ala import TopoInfo
from .enhsamp import make_config_for_each_simulation
from .ensemble import Ensemble, get_ensemble_path, load_ensemble
from .misc import ase_to_pmg, get_logger, is_geom_valid, load_from_xyz, make_dir
from .prediction import get_prediction

__all__ = [
    "greedy_sampling",
    "sparse_time_sampling",
    "rmsd_hierarchical_sampling",
    "latent_space_sampling",
    "compute_rmsd",
    "deduplicate_geometries",
    "cosine_dist",
    "create_dset_from_traj",
    "remove_invalid_points",
    "write_to_pdb_opls",
    "run",
    "handle_sampling",
]


logger = get_logger("SAMPLING")


def calculate_uncertainty_llim(uncertainty: np.ndarray, quantile: float) -> float:
    """
    Calculate the uncertainty threshold based on the quantile of the uncertainty
    """
    return np.quantile(uncertainty, quantile)


def greedy_sampling(
    uncertainty_llim: float,
    extlog: pd.DataFrame,
    traj: Trajectory,
    uncertainty_key: str = "CV",
) -> Tuple[List[Atoms], List[float], List[int]]:
    """
    Get all geometries in traj that corresponds to high uncertainty
    in the extlog file (uncertainty > uncertainty_llim)
    return a list of Atoms objects
    """
    uncertainty = extlog[uncertainty_key].to_numpy()
    indices = np.where(uncertainty > uncertainty_llim)[0]

    sampled_points = [traj[i] for i in indices]
    times = extlog["Time[ps]"][indices].to_list()
    indices = indices.tolist()

    logger(
        f"Greedy sampling: {len(sampled_points)} points with {uncertainty_llim} uncertainty threshold"
    )

    return sampled_points, times, indices


def sparse_time_sampling(
    uncertainty_llim: float,
    extlog: pd.DataFrame,
    traj: Trajectory,
    time_width: float,
    uncertainty_key: str = "CV",
) -> Tuple[List[Atoms], List[float], List[int]]:
    """
    Get geometries with the highest uncertainty within each
    time window of the trajectory
    Return a list of Atoms objects
    """
    time = extlog["Time[ps]"].to_numpy()
    bins = np.arange(time[0], time[-1] + time_width, time_width)

    extlog["bin"] = pd.cut(extlog["Time[ps]"], bins)
    idmax_u = extlog.groupby(["bin"])[uncertainty_key].idxmax()
    max_u = extlog.iloc[idmax_u][uncertainty_key]

    # filter out idmax_u and max_u that are below the uncertainty threshold
    max_u = max_u.loc[max_u > uncertainty_llim]
    idmax_u = max_u.index

    # get high uncertainty atoms and the corresponding time
    sampled_points, times, indices = [], [], []
    for i in idmax_u:
        indices.append(i)
        times.append(extlog["Time[ps]"][i])
        sampled_points.append(traj[i])
    return sampled_points, times, indices


def compute_rmsd(atoms_list: List[Atoms]) -> np.ndarray:
    """
    Compute RMSD distance between all geometries in the list
    Note that this function does not do any rotation or translation
    I simply assume that the geometries are already aligned
    """
    allxyz = np.array([at.positions for at in atoms_list])

    # compute the distance matrix
    rmsd = (
        np.power(allxyz[:, None, ...] - allxyz[None, ...], 2)
        .reshape(len(atoms_list), len(atoms_list), -1)
        .mean(-1)
    )

    return rmsd


def deduplicate_geometries(
    atoms_list: List[Atoms],
    threshold: float = 0.005,
    linkage_method: str = "complete",
) -> Tuple[List[Atoms], List[int]]:
    """
    Remove geometries that are too similar to each other in terms of RMSD distance
    params:
        atoms_list: list of ase.Atoms objects
        threshold: threshold for RMSD distance for deduplication
        linkage_method: method for hierarchical clustering (see scipy.cluster.hierarchy.linkage)
            can choose from 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    Returns:
        atoms_list: a list of 'deduplicated' Atoms object
        representative: indices to the Atoms object
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    # compute rmsd for atoms list
    rmsd = compute_rmsd(atoms_list)
    # convert distance matrix to condensed distance matrix
    condensed_rmsd = rmsd[np.triu_indices(len(rmsd), k=1)]
    # perform hierarchical clustering using the "linkage" method
    Z = linkage(condensed_rmsd, method=linkage_method)
    # use distance threshold to cut the dendogram
    clusters = fcluster(Z, threshold, criterion="distance")
    # get a representative from each cluster
    representative = []
    for i in range(1, clusters.max() + 1):
        representative.append(np.where(clusters == i)[0][0])
    # return the representative atoms
    return [atoms_list[i] for i in representative], representative


def rmsd_hierarchical_sampling(
    uncertainty_llim: float,
    extlog: pd.DataFrame,
    traj: Trajectory,
    uncertainty_key: str = "CV",
    rmsd_threshold: float = 0.005,
    linkage_method: str = "complete",
) -> Tuple[List[Atoms], List[float], List[int]]:
    greedy_sampled_points, times, indices = greedy_sampling(
        uncertainty_llim=uncertainty_llim,
        extlog=extlog,
        traj=traj,
        uncertainty_key=uncertainty_key,
    )
    deduplicated_points, representative = deduplicate_geometries(
        atoms_list=greedy_sampled_points,
        threshold=rmsd_threshold,
        linkage_method=linkage_method,
    )
    times = [times[i] for i in representative]
    indices = [indices[i] for i in representative]
    return deduplicated_points, times, indices


def cosine_dist(
    X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1000
) -> torch.Tensor:
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]

    similarity = torch.empty((X.size(0), Y.size(0)), device=X.device, dtype=X.dtype)

    # normalize values
    X_normalized = X / torch.linalg.norm(X, dim=-1).reshape(-1, 1)
    Y_normalized = Y / torch.linalg.norm(Y, dim=-1).reshape(-1, 1)

    # Iterate through X and Y in batches
    for start_idx in range(0, X_normalized.size(0), batch_size):
        end_idx = min(start_idx + batch_size, X_normalized.size(0))
        X_batch = X_normalized[start_idx:end_idx]

        # Compute the dot product between the batch from X and all of Y
        # The resulting shape is (batch_size, Y.size(0))
        sim_batch = torch.mm(X_batch, Y_normalized.t())

        # Assign the computed batch to the corresponding part of the result tensor
        similarity[start_idx:end_idx, :] = sim_batch

    return 1 - similarity


def create_dset_from_traj(traj: Trajectory, forcefield: Union[str, None]) -> Dataset:
    dset = []
    nbr_list = torch.LongTensor(TopoInfo.get_nbr_list(forcefield))
    for atoms in traj:
        nxyz = np.concatenate(
            [atoms.get_atomic_numbers().reshape(-1, 1), atoms.get_positions()], axis=1
        )
        dset.append(
            {
                "nxyz": torch.FloatTensor(nxyz),
                "nbr_list": nbr_list,
            }
        )
    dset = Dataset(concatenate_dict(*dset))
    return dset


def iterate_upper_triangular(matrix: np.ndarray, k: int):
    n = len(matrix)
    for i in range(n):
        for j in range(i + k, n):
            yield matrix[i, j]


def latent_space_sampling(
    extlog: pd.DataFrame,
    traj: Trajectory,
    model: Ensemble,
    uncertainty_llim: float = 0.1,
    cosdist_threshold: float = 0.01,
    linkage_method: str = "complete",
    batch_size: int = 64,
    device: str = "cuda",
    forcefield: Union[str, None] = None,
    uncertainty_key: str = "CV",
) -> Tuple[List[Atoms], List[float], List[int]]:
    """
    Sample geometries based on the cosine distance between the latent space
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    # get indices of points with uncertainty above the threshold
    sampled_points, times, indices = greedy_sampling(
        uncertainty_llim=uncertainty_llim,
        extlog=extlog,
        traj=traj,
        uncertainty_key=uncertainty_key,
    )

    if len(sampled_points) == 0:
        return [], [], []

    # get embedding space of the sampled points
    if "PaiNN" in model.__repr__() or "SchNet" in model.__repr__():
        sampled_points = create_dset_from_traj(sampled_points, forcefield)

    _, sampled_pred = get_prediction(
        model, sampled_points, batch_size, device, requires_grad=False
    )

    # compute the cosine distance between the latent space of the training set and the sampled set
    # only choose the embedding from the first model
    sample_embedding = sampled_pred["embedding"][0].reshape(len(sampled_points), -1)
    del model, sampled_pred

    # calculate cosine distances between picked embeddings
    cos_dist = cosine_dist(sample_embedding, sample_embedding).detach().cpu().numpy()
    del sample_embedding

    # convert distance matrix to condensed distance matrix
    condensed_cosdist = iterate_upper_triangular(cos_dist, k=1)

    del cos_dist
    condensed_cosdist = np.fromiter(condensed_cosdist, dtype=np.float32)

    if len(condensed_cosdist) == 0:
        return [], [], []

    # perform hierarchical clustering using the "linkage" method
    Z = linkage(condensed_cosdist, method=linkage_method)
    del condensed_cosdist
    # use distance threshold to cut the dendogram
    clusters = fcluster(Z, cosdist_threshold, criterion="distance")
    del Z
    # get a representative from each cluster
    final_sampled_points, final_times, final_indices = [], [], []
    for i in range(1, clusters.max() + 1):
        idx = np.where(clusters == i)[0]
        idx = indices[np.random.choice(idx)]

        final_indices.append(idx)
        final_times.append(extlog.iloc[idx]["Time[ps]"])
        final_sampled_points.append(traj[idx])

    return final_sampled_points, final_times, final_indices


def furthest_point_sampling(
    extlog: pd.DataFrame,
    traj: Trajectory,
    model: Ensemble,
    train_dset: Union[Dataset, List[Atoms]],
    score_threshold: float = 20.0,
    score_threshold_type: str = "absolute",
    min_sampled_points: int = 20,
    batch_size: int = 64,
    device: str = "cuda",
    forcefield: Union[str, None] = None,
) -> Tuple[List[Atoms], List[float], List[int], float]:
    from skmatter.sample_selection import FPS

    # get embedding space of the sampled points
    if "PaiNN" in model.__repr__() or "SchNet" in model.__repr__():
        traj = create_dset_from_traj(traj, forcefield)

    _, train_pred = get_prediction(
        model, train_dset, batch_size, device, requires_grad=False
    )

    _, sampled_pred = get_prediction(
        model, traj, batch_size, device, requires_grad=False
    )

    # get the embedding space of the training set and the explored traj and
    # reshape into (n_samples * n_atoms, n_features)
    train_embedding = train_pred["embedding"][0].detach().cpu().numpy()
    train_embedding = train_embedding.reshape(len(train_dset), -1)

    sample_embedding = sampled_pred["embedding"][0].detach().cpu().numpy()
    sample_embedding = sample_embedding.reshape(len(traj), -1)

    del model, train_pred, sampled_pred

    # concatenate the embeddings
    embeddings = np.concatenate([train_embedding, sample_embedding], axis=0)
    del train_embedding, sample_embedding

    initialize = np.arange(len(train_dset)).tolist()

    while True:
        fps = FPS(
            initialize=initialize,
            n_to_select=None,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=False,
            full=False,
            random_state=0,
        )
        fps.fit(embeddings)

        selected_inds = fps.selected_idx_
        selected_inds = np.delete(
            selected_inds, np.where(np.isin(selected_inds, np.array(initialize)))
        )

        # if the number of selected indices is more than 10, break out of the loop
        if len(selected_inds) > min_sampled_points:
            break

        # if the score_threshold is less than 1e-6, break out of the loop
        if score_threshold < 1e-6:
            break

        score_threshold = score_threshold * 0.9
        min_sampled_points = int(min_sampled_points * 0.95)
        logger(
            f"IMPORTANT!! {len(selected_inds)} selected, less than the minimum {min_sampled_points} required. Lowering the score threshold to {score_threshold}"
        )

    # correct the indices
    selected_inds = [i - len(initialize) for i in selected_inds]

    final_sampled_points = [traj[i] for i in selected_inds]
    final_times = extlog.iloc[selected_inds]["Time[ps]"].to_list()

    return final_sampled_points, final_times, selected_inds, score_threshold


def remove_invalid_points(
    traj: List[Atoms],
    extlog: pd.DataFrame,
    nbr_list: np.ndarray,
    rmin: float = 0.75,
    rmax: float = 2.0,
    pbc: bool = True,
) -> Tuple[List, pd.DataFrame]:
    """
    Remove atoms that have exploded in the MD
    Args:
        traj: ase.Trajectory object
        extlog: extxyz log file
        nbr_list: neighbor list
        rmin: minimum distance between atoms (angstrom)
        rmax: maximum distance between atoms (angstrom)
        pbc: whether the system has periodic boundary conditions
    Returns:
        valid_atoms: list of valid ase.Atoms objects
        extlog: updated extxyz log file
    """
    assert len(traj) == len(extlog)

    # remove points where the CV, AbsGradPot, inv_m_cv, AbsGradCV, GradCV_GradU
    # are all stagnant (meaning the atoms have exploded)
    cols = [
        i
        for i in ["CV", "AbsGradPot", "inv_m_cv", "AbsGradCV", "GradCV_GradU"]
        if i in extlog.columns
    ]
    # compare the rows with the row two steps before
    diff = extlog.diff(periods=2)
    fulfilled_conds = (diff[cols] != 0.0).all(1)
    indices = np.where(fulfilled_conds)[0]

    extlog = extlog[fulfilled_conds].reset_index(drop=False)
    traj = [traj[i] for i in indices]
    assert len(traj) == len(
        extlog
    ), "Number of geometries and extlog entries do not match after removing null points"

    # remove points where the distance between atoms are too large or too small
    if pbc is True:
        valid_atoms, valid_index = [], []
        for i, at in enumerate(traj):
            dist_mat = ase_to_pmg(at).distance_matrix
            if (
                is_geom_valid(
                    dist_mat=dist_mat, rmin=rmin, rmax=rmax, nbr_list=nbr_list
                )
                is True
            ):
                valid_atoms.append(at)
                valid_index.append(i)

    else:
        atom_pos = np.stack([atoms.get_positions() for atoms in traj], axis=0)
        diff = atom_pos[:, :, np.newaxis, :] - atom_pos[:, np.newaxis, :, :]
        dist_matrices = np.sqrt(np.sum(diff**2, axis=-1))
        del diff, atom_pos
        valid_atoms, valid_index = [], []
        for i, (at, dist_mat) in enumerate(zip(traj, dist_matrices)):
            if (
                is_geom_valid(
                    nbr_list=nbr_list, rmin=rmin, rmax=rmax, dist_mat=dist_mat
                )
                is True
            ):
                valid_atoms.append(at)
                valid_index.append(i)

    extlog = extlog.iloc[valid_index].reset_index(drop=True)

    return valid_atoms, extlog


def write_to_pdb_opls(atoms: Atoms, filename: str) -> None:
    """
    Write ase.Atoms object to a pdb file with correct connections
    """
    write(filename=filename, images=atoms, format="proteindatabank")

    # rewrite the pdb
    output = open(filename, "r")
    data = [i.strip().split() for i in output.readlines()][1:-1]
    correct_form = [
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
    for i, d in enumerate(data):
        data[i][1] = str(i + 1)
        data[i][2] = correct_form[i]
        data[i][3] = "UNK"

    with open(filename, "w") as f:

        def fwrite(x):
            return f.write(x + "\n")

        fwrite(
            "CRYST1   {:>.3f}   {:>.3f}   {:>.3f}  90.00  90.00  90.00 P 1           1".format(
                20.0, 20.0, 20.0
            )
        )
        for d in data:
            x = "{:>.3f}".format(float(d[5]))
            y = "{:>.3f}".format(float(d[6]))
            z = "{:>.3f}".format(float(d[7]))
            fwrite(
                """{:<6s}{:>5s}{:>5s}{:>4s} {:>5s}      {:>6s}  {:>6s}  {:>6s}  {:>4s}  {:>4s}           {}""".format(
                    d[0], d[1], d[2], d[3], d[4], x, y, z, d[8], d[9], d[10]
                )
            )

        fwrite(
            """TER
CONECT    1    2
CONECT    2    3
CONECT    3    4
CONECT    3    5
CONECT    5    6
CONECT    2    7
CONECT    7    8
CONECT    8    9
CONECT    8   10
CONECT    1   11
CONECT    1   12
CONECT    1   13
CONECT    2   14
CONECT    5   15
CONECT    6   16
CONECT    6   17
CONECT    6   18
CONECT    7   19
CONECT   10   20
CONECT   10   21
CONECT   10   22
END"""
        )
        f.close()
    logger(f"Created {filename}")


def run(all_configs: List[dict]) -> List[dict]:
    all_extlogs, all_trajs = [], []
    for config in all_configs:
        logger(f"Reading from {config['enhsamp']['traj_dir']}")

        # read the log files
        extlog = pd.read_csv(
            f"{config['enhsamp']['traj_dir']}/ext.log",
            sep="\s+",
            header=0,
            on_bad_lines="warn",
        )
        extlog["temperature"] = config["enhsamp"]["temperature"]

        traj = Trajectory(f"{config['enhsamp']['traj_dir']}/traj.traj")

        logger(f"{len(traj)} geometries in trajectory")

        # get neighbor list
        if config["system"] == "ala":
            nbr_list = TopoInfo.get_nbr_list(config["openmm"].get("forcefield", None))
            forcefield = config["openmm"].get("forcefield", None)
        else:
            nbr_list = None
            forcefield = None

        # make sure all sampled points are valid
        logger("Removing invalid points")
        traj, extlog = remove_invalid_points(
            traj=traj,
            extlog=extlog,
            nbr_list=nbr_list,
            rmin=config["enhsamp"].get("unphysical_rmin"),
            rmax=config["enhsamp"].get("unphysical_rmax"),
            pbc=any(traj[0].pbc),
        )

        extlog["identifier"] = extlog.apply(
            lambda x: f"{int(x['temperature'])}K_{int(x['index'])}_{x['Time[ps]']:.3f}ps",
            axis=1,
        )

        all_extlogs.append(extlog)
        all_trajs.extend(traj)

        del traj, extlog

    all_extlogs = pd.concat(all_extlogs, ignore_index=True)
    logger(f"Total number of explored points: {len(all_trajs)}")

    # shuffle the traj and extlog for better sampling
    permutated_inds = np.random.permutation(len(all_trajs))
    all_trajs = [all_trajs[i] for i in permutated_inds]
    all_extlogs = all_extlogs.iloc[permutated_inds].reset_index(drop=True)

    # just use the first config for now
    config = all_configs[0]

    # if sampling_llim is None, calculate it from the uncertainty
    if config["enhsamp"].get("sampling_llim", None) is None:
        config["enhsamp"]["sampling_llim"] = calculate_uncertainty_llim(
            uncertainty=all_extlogs[config["enhsamp"]["uncertainty_key"]].to_numpy(),
            quantile=config["enhsamp"].get("sampling_llim_quantile", 0.70),
        )

    # sample the trajectory
    logger("Sampling trajectory")
    if config["enhsamp"]["sampling_method"] == "greedy":
        sampled_points, times, indices = greedy_sampling(
            uncertainty_llim=config["enhsamp"]["sampling_llim"],
            extlog=all_extlogs,
            traj=all_trajs,
            uncertainty_key=config["enhsamp"]["uncertainty_key"],
        )
    elif config["enhsamp"]["sampling_method"] == "rmsd_hierarchical":
        sampled_points, times, indices = rmsd_hierarchical_sampling(
            uncertainty_llim=config["enhsamp"]["sampling_llim"],
            extlog=all_extlogs,
            traj=all_trajs,
            uncertainty_key=config["enhsamp"]["uncertainty_key"],
            linkage_method=config["enhsamp"]["linkage_method"],
            rmsd_threshold=config["enhsamp"]["rmsd_threshold"],
        )
    elif config["enhsamp"]["sampling_method"] == "sparse_time":
        sampled_points, times, indices = sparse_time_sampling(
            uncertainty_llim=config["enhsamp"]["sampling_llim"],
            extlog=all_extlogs,
            traj=all_trajs,
            uncertainty_key=config["enhsamp"]["uncertainty_key"],
            time_width=config["enhsamp"]["time_width"],
        )
    elif config["enhsamp"]["sampling_method"] == "latent_embedding":
        model_path = get_ensemble_path(config["model"]["outdir"])
        model = load_ensemble(
            model_path=model_path,
            model_type=config["model"]["model_type"],
            device=config["train"]["device"],
        )
        sampled_points, times, indices = latent_space_sampling(
            extlog=all_extlogs,
            traj=all_trajs,
            model=model,
            uncertainty_llim=config["enhsamp"]["sampling_llim"],
            cosdist_threshold=config["enhsamp"]["cosdist_threshold"],
            linkage_method=config["enhsamp"]["linkage_method"],
            batch_size=config["dset"]["batch_size"],
            device=config["train"]["device"],
            forcefield=forcefield,
        )
        del model
    elif config["enhsamp"]["sampling_method"] == "furthest_point":
        model_path = get_ensemble_path(config["model"]["outdir"])
        model = load_ensemble(
            model_path=model_path,
            model_type=config["model"]["model_type"],
            device=config["train"]["device"],
        )

        train_path = all_configs[0]["dset"]["path"]
        if train_path.endswith(".pth.tar"):
            train_dset = Dataset.from_file(train_path)
        elif train_path.endswith(".xyz"):
            train_dset = load_from_xyz(train_path)
        else:
            raise ValueError(f"Unknown file format for {train_path}")

        sampled_points, times, indices, score_threshold = furthest_point_sampling(
            extlog=all_extlogs,
            traj=all_trajs,
            model=model,
            train_dset=train_dset,
            score_threshold=config["enhsamp"]["score_threshold"],
            score_threshold_type=config["enhsamp"]["score_threshold_type"],
            min_sampled_points=config["enhsamp"]["min_sampled_points"],
            batch_size=config["dset"]["batch_size"],
            device=config["train"]["device"],
            forcefield=config["openmm"].get("forcefield", None),
        )

        # rewrite the score_threshold to the config
        config["enhsamp"]["score_threshold"] = score_threshold
    else:
        raise ValueError(
            f"SAMPLING: {config['enhsamp']['sampling_method']} is not a valid sampling method"
        )

    # write the details of high uncertainty points to a csv file
    sampled_df = []
    for i, time in zip(indices, times):
        geom = all_trajs[i]
        nxyz = np.concatenate(
            [geom.get_atomic_numbers().reshape(-1, 1), geom.get_positions()], axis=-1
        )
        log = all_extlogs.iloc[i]
        sampled_df.append(
            {
                "nxyz": torch.FloatTensor(nxyz),
                "lattice": torch.FloatTensor(geom.get_cell().array),
                "simulation_type": config["enhsamp"]["method"],
                "identifier": f"gen{config['gen']}_{log['identifier']}",
                "temperature(K)": log["temperature"],
                "index": i,
                "time(ps)": time,
                "cv": log[config["enhsamp"]["uncertainty_key"]],
                "U0+bias(eV)": log.get("U0+bias[eV]", np.nan),
                "U0(eV)": log.get("U0[eV]", np.nan),
                "AbsGradPot": log.get("AbsGradPot", np.nan),
                "Lambda": log.get("Lambda", np.nan),
                "inv_m_cv": log.get("inv_m_cv", np.nan),
                "AbsGradCV": log.get("AbsGradCV", np.nan),
                "GradCV_GradU": log.get("GradCV_GradU", np.nan),
            }
        )

    logger(
        f"Sampled {len(sampled_df)} points in total using {config['enhsamp']['sampling_method']} sampling method"
    )

    return sampled_df


def handle_sampling(config: dict) -> Dataset:
    logger(
        f"Running {config['enhsamp']['sampling_method']} sampling on {config['enhsamp']['traj_dir']}"
    )

    orig_config = config.copy()

    # create a config for each simulation
    all_configs = make_config_for_each_simulation(orig_config)

    # run sampling for each simulation
    sampled_df = run(all_configs)

    if len(sampled_df) == 0:
        logger("No points sampled. Exiting.")
        return

    # create output dir if does not exist or replace if exist
    make_dir(orig_config["enhsamp"]["sampled_path"])
    # save the sampled points
    sampled_dset = Dataset(concatenate_dict(*sampled_df))
    sampled_dset.save(config["enhsamp"]["sampled_path"])

    logger("Done.")

    return sampled_dset
