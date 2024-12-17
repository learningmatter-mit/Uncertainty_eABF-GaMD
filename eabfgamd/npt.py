import math
import os
from typing import Union

import numpy as np
from ase import Atoms, units
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize.optimize import Dynamics


class BerendsenNPT(Inhomogeneous_NPTBerendsen):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float = 1 * units.fs,
        temperature_K: float = 300.0,
        taut: float = 0.5e3 * units.fs,
        taup: float = 1e3 * units.fs,
        pressure_au: float = 1 * units.bar,
        compressibility_au: float = 4.57e-5 / units.bar,
        T_init: Union[float, None] = None,
        mask: Union[list, tuple] = (1, 1, 1),
        fixcm: bool = True,
        trajectory: Union[str, None] = None,
        logfile: Union[str, None] = None,
        loginterval: int = 1,
        nbr_update_period: int = 20,
        append_trajectory: bool = False,
        max_steps: Union[int, None] = None,
        unphysical_rmin: Union[float, None] = None,
        unphysical_rmax: Union[float, None] = None,
        hardcoded_nbr_list: Union[np.ndarray, None] = None,
        **kwargs,
    ):
        if "temperature" in kwargs:
            raise ValueError(
                "temperature is not a valid keyword argument. Please use temperature_K instead."
            )

        for key in ["pressure", "compressibility"]:
            if key in kwargs:
                raise ValueError(
                    f"{key} is not a valid keyword argument. Please use {key}_au instead."
                )

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        Inhomogeneous_NPTBerendsen.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            taut=taut,
            taup=taup,
            pressure_au=pressure_au,
            compressibility_au=compressibility_au,
            fixcm=fixcm,
            mask=mask,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )

        # Initialize simulation parameters
        # convert units
        self.nbr_update_period = nbr_update_period
        self.num_steps = max_steps
        self.max_steps = 0

        self.T = temperature_K * units.kB

        self.unphysical_rmin = unphysical_rmin
        self.unphysical_rmax = unphysical_rmax
        self.hardcoded_nbr_list = hardcoded_nbr_list

        # initial Maxwell-Boltmann temperature for atoms
        if T_init is not None:
            # convert units
            T_init = T_init * units.kB
        else:
            T_init = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, T_init)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)

    def converged(self):
        step_criterion = self.nsteps >= self.max_steps

#         Ekin = self.atoms.get_kinetic_energy()
#         if Ekin == 0.0 or Ekin > 1e6:
#             print("Ekin is zero or too high. Stopping simulation.")
#             return True

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
                    print("Distance below min found. Stopping simulation.")
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
                print("Distance above max found. Stopping simulation.")

            return step_criterion or dist_criterion

        return step_criterion

    def run(self, steps=None):
        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        # self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()


class StochasticCellRescaling(Inhomogeneous_NPTBerendsen):
    def __init__(
        self,
        atoms,
        timestep: float,
        temperature: float,
        pressure: float,
        tau_t: float,
        tau_p: float,
        compressibility: float,
        maxwell_temp: float = None,
        trajectory: str = None,
        logfile: str = None,
        loginterval: int = 1,
        max_steps: int = None,
        random_seed: int = 354562342,
        nbr_update_period: int = 10,
        append_trajectory: bool = True,
        **kwargs,
    ):
        """
        This is the implementation of pressure coupling from the paper
        Mattia Bernetti, Giovanni Bussi; Pressure control using stochastic
        cell rescaling. J. Chem. Phys. 21 September 2020; 153 (11): 114107.
        https://doi.org/10.1063/5.0020514

        Args:
        """

        Inhomogeneous_NPTBerendsen.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            temperature=temperature,
            pressure=pressure,
            tau_t=tau_t,
            tau_p=tau_p,
            compressibility=compressibility,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

        self.dim = 3  # this code is for 3D simulations
        self.XX = 0
        self.YY = 1
        self.ZZ = 2

        self.dt = timestep * units.fs
        self.T = temperature * units.kB
        self.Natom = len(atoms)

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        print("THE RANDOM NUMBER SEED WAS: %i" % (self.random_seed))

        # no overall rotation or translation
        self.N_dof = self.d * self.Natom - 6
        self.targetEkin = 0.5 * self.N_dof * self.T

        # pressure coupling parameters
        self.pressure = pressure * units.GPa
        self.nstpcouple = 0  # number of steps between pressure coupling updates
        self.tau_p = tau_p * units.fs  # pressure coupling time constant
        self.tau_t = tau_t * units.fs  # temperature coupling time constant
        self.compressibility = np.zeros(
            (self.dim, self.dim)
        )  # compressibility matrix ((mol nm^3) / kJ)
        self.refcoord_scaling = np.zeros(
            (self.dim, self.dim)
        )  # how to scale absolute reference coordinates
        self.ref_p = np.zeros((self.dim, self.dim))  # reference pressure tensor

        self.num_steps = max_steps
        self.n_steps = 0
        self.max_steps = 0

        self.nbr_update_period = nbr_update_period

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is None:
            maxwell_temp = temperature

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=maxwell_temp)

    def calculate_scaling_matrix(self, pressure_coupling_type, pres, cell):
        """Calculate the scaling matrix for the stochastic cell rescaling
        algorithm. This is a symmetric matrix that is used to scale the
        cell vectors. The matrix is calculated from a random vector
        sampled from a normal distribution with zero mean and unit
        variance. The matrix is then calculated as the outer product of
        the random vector with itself. The matrix is then scaled by the
        square root of the time step and the square root of the
        temperature. The matrix is then added to the identity matrix to
        ensure that the cell vectors are not scaled in the absence of
        pressure coupling. The matrix is then returned.

        Returns:
            numpy.ndarray: The scaling matrix for the stochastic cell
            rescaling algorithm.
        """

        # call to the helper function to calculate the mu value
        # TODO: write _calculate_mu function
        # TODO: where does the pres come from?
        mu = self._calculate_mu(pressure_coupling_type, pres, cell)

        # adjusting mu for triclinic box constraints
        mu[1, 0] += mu[0, 1]
        mu[2, 0] += mu[0, 2]
        mu[2, 1] += mu[1, 2]
        mu[0, 1] = 0
        mu[0, 2] = 0
        mu[1, 2] = 0

        # Calculate the barostat integral
        baros_integral = 0
        for d in range(self.dim):  # Assuming DIM is 3
            for n in range(d + 1):
                # TODO: check out where the force_vir and constraint_vir are coming from
                baros_integral -= (
                    2
                    * (mu[d, n] - (1 if n == d else 0))
                    * (force_vir[d, n] + constraint_vir[d, n])
                )

        # Check for excessive pressure scaling
        if not (
            0.99 <= mu[0, 0] <= 1.01
            and 0.99 <= mu[1, 1] <= 1.01
            and 0.99 <= mu[2, 2] <= 1.01
        ):
            warning_msg = f"\nStep {step} Warning: pressure scaling more than 1%, mu: {mu[0, 0]} {mu[1, 1]} {mu[2, 2]}\n"
            print(
                warning_msg
            )  # Assuming print to stderr or logging to a file as needed

        return mu, baros_integral

    def compressibility_factor(self, i, j, coupling_time_period):
        return self.compressibility[i, j] * coupling_time_period / self.tau_p

    def _calculate_mu(
        self,
        pressure_coupling_type,
        pres,
        box,
        coupling_time_period,
        ensemble_temperature,
    ):
        scalar_pressure = 0
        xy_pressure = 0

        for d in range(self.dim):
            scalar_pressure += pres[d, d] / self.dim
            if d != self.ZZ:
                xy_pressure += pres[d, d] / (self.dim - 1)

        # Calculate the mu value
        mu = np.zeros((self.dim, self.dim))

        # set random number generator
        np.random.set_state(self.random_seed + self.n_steps)
        # TODO: is this really loc=0 and scale=1?
        gauss, gauss2 = np.random.normal(), np.random.normal()

        volume = np.linalg.det(box)
        # TODO: find out how the ensemble temperature is obtained
        kT = ensemble_temperature * kB
        if kT < 0.0:
            kT = 0.0
        # TODO: find out if coupling time period is the same as nstpcouple

        if pressure_coupling_type == "Isotropic":
            for d in range(self.dim):
                factor = self.compressibility_factor(
                    d, d, pressure_coupling_type, coupling_time_period
                )
                mu[d, d] = np.exp(
                    -factor * (self.ref_p[d, d] - scalar_pressure) / self.dim
                    + np.sqrt(2.0 * kT * factor * presfac / volume) * gauss / self.dim
                )
                # TODO: what is this presfac?

        elif pressure_coupling_type == "SemiIsotropic":
            for d in range(self.dim):
                factor = self.compressibility_factor(
                    d, d, pressure_coupling_type, coupling_time_period
                )
                if d != self.ZZ:
                    mu[d, d] = np.exp(
                        -factor * (self.ref_p[d, d] - xy_pressure) / self.dim
                        + np.sqrt(
                            (self.dim - 1)
                            * 2.0
                            * kT
                            * factor
                            * presfac
                            / volume
                            / self.dim
                        )
                        / (self.dim - 1)
                        * gauss
                    )
                else:
                    mu[d, d] = np.exp(
                        -factor * (self.ref_p[d, d] - pres[d, d]) / self.dim
                        + np.sqrt(2.0 * kT * factor * presfac / volume / self.dim)
                        * gauss2
                    )

        elif pressure_coupling_type == "SurfaceTension":
            zz = self.ZZ
            for d in range(self.dim):
                factor = self.compressibility_factor(
                    d, d, pressure_coupling_type, coupling_time_period
                )
                if d != self.ZZ:
                    mu[d, d] = np.exp(
                        -factor
                        * (
                            self.ref_p[zz, zz]
                            - self.ref_p[d, d] / box[zz, zz]
                            - xy_pressure
                        )
                        / self.dim
                        + np.sqrt(4.0 / 3.0 * kT * factor * presfac / volume)
                        / (self.dim - 1)
                        * gauss
                    )
                else:
                    mu[d, d] = np.exp(
                        -factor * (self.ref_p[d, d] - pres[d, d]) / self.dim
                        + np.sqrt(2.0 / 3.0 * kT * factor * presfac / volume) * gauss2
                    )

        else:
            raise ValueError(
                f"Pressure coupling type {pressure_coupling_type} not recognized"
            )

        return mu

    def invert_box_matrix(self, mu):
        # Ensure the matrix is a 3x3 matrix representing a box (with zeroes in the upper-right triangle)
        assert mu.shape == (3, 3) and np.allclose(
            mu[np.triu_indices(3, 1)], 0
        ), "Matrix is not a valid box matrix"

        # Ensure the matrix is invertible
        det = mu[0, 0] * mu[1, 1] * mu[2, 2]
        if abs(det) <= 100 * np.finfo(float).eps:
            raise ValueError("Cannot invert matrix, determinant is too close to zero")

        # Initialize the inverse_muination matrix with zeros
        inverse_mu = np.zeros_like(mu)

        # Invert the diagonal elements
        inverse_mu[0, 0] = 1 / mu[0, 0]
        inverse_mu[1, 1] = 1 / mu[1, 1]
        inverse_mu[2, 2] = 1 / mu[2, 2]

        # Compute the off-diagonal elements based on the given formula
        inverse_mu[2, 0] = (
            (mu[1, 0] * mu[2, 1] * inverse_mu[1, 1] - mu[2, 0])
            * inverse_mu[0, 0]
            * inverse_mu[2, 2]
        )
        inverse_mu[1, 0] = -mu[1, 0] * inverse_mu[0, 0] * inverse_mu[1, 1]
        inverse_mu[2, 1] = -mu[2, 1] * inverse_mu[1, 1] * inverse_mu[2, 2]

        # The upper-right triangle remains zero
        inverse_mu[0, 1] = 0.0
        inverse_mu[0, 2] = 0.0
        inverse_mu[1, 2] = 0.0

        return inverse_mu

    def scale_coordinates_and_velocities(self, mu):
        inverse_mu = self.invert_box_matrix(
            mu
        )  # TODO: check if this is the right way to calculate the inverse

        # Scale the positions
        new_pos = np.dot(self.atoms.get_positions(), mu)

        # Scale the velocities
        new_vel = np.dot(self.atoms.get_velocities(), inverse_mu)

        # Scale the box vectors
        new_box = np.dot(self.atoms.get_cell(), mu)
        # Preserve the shape of the box
        new_box = self.preserve_box_shape(new_box)
        # Ensure all positions are within the new box
        new_pos = self.pbc(positions=new_pos, cell=new_box)

        # set the new positions, velocities and box
        self.atoms.set_positions(new_pos)
        self.atoms.set_velocities(new_vel)
        self.atoms.set_cell(new_box)

    def preserve_box_shape(self, box):
        # TODO: check if this is the right way to preserve the box shape (this
        # is copilot code)
        # Ensure the box is a 3x3 matrix
        assert box.shape == (3, 3), "Box is not a valid 3x3 matrix"

        # Ensure the box is symmetric
        assert np.allclose(box, box.T), "Box is not symmetric"

        # Ensure the box is positive definite
        if not np.all(np.linalg.eigvals(box) > 0):
            raise ValueError("Box is not positive definite")

        return box

    def pbc(self, positions, cell):
        pass

    def step(self):
        # evolve thermostat for half a step
        # get forces
        # get scaling matrix
        # scale coordinates, velocities and box
        # act with forces on velocities for half a step
        # act with velocities on positions
        # update forces based on new positions
        # act with forces on velocities for half a step
        # evolve thermostat for half a step
        pass
