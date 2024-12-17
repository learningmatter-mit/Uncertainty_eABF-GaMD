import os


__all__ = [
    "Paths",
    "Energy",
]


class Paths:
    if os.getenv("DATA") is None:
        raise ValueError("$DATA does not exist")
    else:
        base_dir = f"{os.getenv('DATA')}/projects/unc_eabf/silica"

    params_dir = os.path.join(base_dir, "params")
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    img_dir = os.path.join(base_dir, "images")
    processed_dir = os.path.join(base_dir, "processed_results")

    inbox_params_dir = os.path.join(params_dir, "inbox")
    completed_params_dir = os.path.join(params_dir, "completed")

    # train_path = os.path.join(data_dir, "dft", "train_num1218.xyz")
    # calib_path = os.path.join(data_dir, "dft", "calib_num100.xyz")
    # test_path = os.path.join(data_dir, "dft", "test_num373.xyz")
    train_path = os.path.join(data_dir, "lammps", "amorphous_train_num45.xyz")
    calib_path = os.path.join(data_dir, "lammps", "amorphous_calib_num5.xyz")
    test_path = os.path.join(data_dir, "lammps", "crystalline_num50.xyz")


class Energy:
    min_energy = -125667.96875  # kcal/mol

    @classmethod
    def scale_energy(cls, energy):
        return energy - cls.min_energy
