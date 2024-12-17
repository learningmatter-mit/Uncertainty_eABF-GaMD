import math
import os
import pickle
from typing import Union

import numpy as np
from ase import Atoms, units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize.optimize import Dynamics


class Langevin(MolecularDynamics):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float = 1.0 * units.fs,  # fs
        temperature_K: float = 300.0,  # K
        friction_per_ps: float = 1.0,  # 1/ps
        T_init: float = None,  # this is T_init for atoms
        random_seed: Union[int, None] = None,
        trajectory: Union[str, None] = None,
        logfile: Union[str, None] = None,
        loginterval: int = 1,
        max_steps: int = None,
        nbr_update_period: int = 20,
        append_trajectory: bool = True,
        unphysical_rmin: Union[float, None] = None,
        unphysical_rmax: Union[float, None] = None,
        hardcoded_nbr_list: Union[np.ndarray, None] = None,
        **kwargs,
    ):
        if "temperature" in kwargs:
            raise ValueError(
                "temperature is not a valid keyword argument. Please use temperature_K instead."
            )

        # Random Number Generator
        if random_seed is None:
            random_seed = np.random.randint(2147483647)
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
            print("THE RANDOM NUMBER SEED WAS: %i" % (random_seed))
        else:
            try:
                np.random.set_state(random_seed)
            except:
                raise ValueError(
                    "\tThe provided seed was neither an int nor a state of numpy random"
                )

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        with open(f"{os.path.dirname(logfile)}/random_state.pickle", "wb") as f:
            pickle.dump(np.random.get_state(), f)

        MolecularDynamics.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

        # Initialize simulation parameters
        # convert units
        self.dt = timestep
        self.T = temperature_K

        self.friction = friction_per_ps / (1e3 * units.fs)  # 1/ps -> 1/fs
        self.rand_push = np.sqrt(
            self.T * self.friction * self.dt * units.kB / 2.0e0
        ) / np.sqrt(self.atoms.get_masses().reshape(-1, 1))
        self.prefac1 = 2.0 / (2.0 + self.friction * self.dt)
        self.prefac2 = (2.0e0 - self.friction * self.dt) / (
            2.0e0 + self.friction * self.dt
        )

        self.num_steps = max_steps

        self.nbr_update_period = nbr_update_period

        self.unphysical_rmin = unphysical_rmin
        self.unphysical_rmax = unphysical_rmax
        self.hardcoded_nbr_list = hardcoded_nbr_list

        # initial Maxwell-Boltmann temperature for atoms
        if T_init is None:
            T_init = self.T

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=T_init)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.remove_constrained_vel(atoms)

    def remove_constrained_vel(self, atoms):
        """
        Set the initial velocity to zero for any constrained or fixed atoms
        """

        constraints = atoms.constraints
        fixed_idx = []
        for constraint in constraints:
            has_keys = False
            keys = ["idx", "indices", "index"]
            for key in keys:
                if hasattr(constraint, key):
                    val = np.array(getattr(constraint, key)).reshape(-1).tolist()
                    fixed_idx += val
                    has_keys = True
            if not has_keys:
                print(
                    (
                        "WARNING: velocity not set to zero for any atoms in constraint "
                        "%s; do not know how to find its fixed indices." % constraint
                    )
                )

        if not fixed_idx:
            return

        fixed_idx = np.array(list(set(fixed_idx)))
        vel = self.atoms.get_velocities()
        vel[fixed_idx] = 0
        self.atoms.set_velocities(vel)

    def converged(self):
        step_criterion = self.nsteps >= self.max_steps

        Ekin = self.atoms.get_kinetic_energy()
        # if Ekin == 0.0 or Ekin > 1e6:
        #     print("Ekin is zero or too high. Stopping simulation.")
        #     return True

        if self.unphysical_rmin is not None or self.unphysical_rmax is not None:
            dist_mat = self.atoms.get_all_distances(mic=True)
            if self.hardcoded_nbr_list is not None:
                dist_mat = dist_mat[
                    self.hardcoded_nbr_list[:, 0], self.hardcoded_nbr_list[:, 1]
                ]

            # check that distances btwn all atoms is not less than specified
            # threshold (e.g. 0.7 A)
            if self.unphysical_rmin is not None:
                if ((dist_mat < self.unphysical_rmin) & (dist_mat != 0.0)).any():
                    print("Unphysical distance found. Stopping simulation.")
                    return True
                else:
                    dist_criterion = False

            # check that distances btwn all atoms is not greater than specified
            # need nbr_list because we only care about distances between
            # atoms that are bonded
            if self.unphysical_rmax is not None:
                assert (
                    self.hardcoded_nbr_list is not None
                ), "Must provide hardcoded_nbr_list if unphysical_rmax is not None"
                mask = dist_mat > self.unphysical_rmax
                dist_criterion = mask.any()

            if dist_criterion:
                print("Unphysical distance found. Stopping simulation.")

            return step_criterion or dist_criterion

        return step_criterion

    def step(self):
        vel = self.atoms.get_velocities()
        masses = self.atoms.get_masses().reshape(-1, 1)

        self.rand_gauss = np.random.randn(
            self.atoms.get_positions().shape[0], self.atoms.get_positions().shape[1]
        )

        vel += self.rand_push * self.rand_gauss
        vel += 0.5e0 * self.dt * self.atoms.get_forces() / masses

        self.atoms.set_velocities(vel)
        self.remove_constrained_vel(self.atoms)

        vel = self.atoms.get_velocities()
        x = self.atoms.get_positions() + self.prefac1 * self.dt * vel

        # update positions
        self.atoms.set_positions(x)

        vel *= self.prefac2
        vel += self.rand_push * self.rand_gauss
        vel += 0.5e0 * self.dt * self.atoms.get_forces() / masses

        self.atoms.set_velocities(vel)
        self.remove_constrained_vel(self.atoms)

    def run(self, steps=None):
        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self, steps=steps_per_epoch)

            x = self.atoms.get_positions(wrap=True)
            self.atoms.set_positions(x)

            self.atoms.update_nbr_list()
            Stationary(self.atoms)
            ZeroRotation(self.atoms)

        # with open('random_state.pickle', 'wb') as f:
        #    pickle.dump(np.random.get_state(), f)

        # load random state for restart as
        # with open('random_state.pickle', 'rb') as f:
        #     state = pickle.load(f)
