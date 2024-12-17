import os
import shutil
import subprocess
import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from ase import Atom, Atoms
from ase.io import read
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from nff.data import Dataset
from tqdm import tqdm

from .ala import Paths as AlaPaths
from .misc import get_logger

__all__ = [
    "ForceReporter",
    "OpenMMCalculator",
    "LAMMPSCHIKCalculator",
    "handle_calc",
]


logger = get_logger("CALC")


DEFAULT_PARAMS = {
    "pdb": "ala.pdb",
    "forcefield": "amber",
    "ff_path": AlaPaths.opls_ff_path,  # only used when forcefield == "opls"
    "prmtop": AlaPaths.amber_prmtop_path,  # only used when forcefield == "amber"
    "inpcrd": AlaPaths.amber_inpcrd_path,  # only used when forcefield == "amber"
    "cutoff": 1.0,  # nm
    "temperature": 300.0,  # K
    "timestep": 0.0002,  # ps
    #    "ewald_error_tol": 0.0005,
    "hbond_constraint": False,
    "constraint_tol": 0.00001,
    "record_interval": 1,
}


class ForceReporter:
    def __init__(self, file, reportInterval):
        self._out = open(file, "w")
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # return (steps, positions, velocities, forces, energies)
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        from openmm import unit

        forces = state.getForces().value_in_unit(
            unit.kilojoule / unit.mole / unit.angstrom
        )
        for f in forces:
            self._out.write("%g %g %g\n" % (f[0], f[1], f[2]))


class OpenMMCalculator:
    def __init__(
        self,
        params: dict,
        output_dir: str = "./",
    ):
        import openmm as omm
        from openmm import app, unit

        logger("Initializing OpenMM calculator")
        self.cutoff = params["cutoff"] * unit.nanometers
        # self.ewald_error_tol = params["ewald_error_tol"]
        self.timestep = params.get("timestep", 0.0002) * unit.picoseconds
        self.temperature = params.get("temperature", 3000) * unit.kelvin
        self.constraint_tol = params.get("constraint_tol", 0.00001)
        self.record_interval = params.get("record_interval", 1000)

        if params["forcefield"] == "opls":
            self.pdb = omm.PDBFile(params["pdb"])
            self.topology = self.pdb.topology
            self.forcefield = omm.ForceField(params["ff_path"])
            self.system = self.forcefield.createSystem(
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=self.cutoff,
                constraints=None,
            )

        elif params["forcefield"] == "amber":
            self.forcefield = app.AmberPrmtopFile(params["prmtop"])
            self.topology = self.forcefield.topology
            self.inpcrd = app.AmberInpcrdFile(params["inpcrd"])
            self.system = self.forcefield.createSystem(
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=self.cutoff,
                constraints=None,
            )

        else:
            raise ValueError(f"Forcefield {params['forcefield']} not supported")

        self.context = omm.Context(
            self.system,
            omm.VerletIntegrator(self.timestep),
        )
        self.output_dir = output_dir

    def single_point_calc(self, geom: dict) -> tuple:
        from openmm import unit
        from openmm.openmm import NmPerAngstrom, Vec3

        position = NmPerAngstrom * geom["nxyz"][:, 1:4].numpy()
        new_positions = ([Vec3(*xyz.tolist()) for xyz in position]) * unit.nanometers

        self.context.setPositions(new_positions)

        state = self.context.getState(
            getVelocities=True, getForces=True, getEnergy=True
        )

        # and saving in units of kcal/mol
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilocalories / unit.moles
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilocalories / unit.mole / unit.angstrom
        )

        return energy, forces

    def simulate(self, total_steps: int) -> None:
        import openmm as omm
        from openmm import app, unit

        integrator = omm.LangevinIntegrator(
            self.temperature, 1.0 / unit.picoseconds, self.timestep
        )
        integrator.setConstraintTolerance(self.constraint_tol)

        platform = omm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(self.topology, self.system, integrator, platform)
        simulation.context.setPositions(self.inpcrd.positions)
        simulation.context.setVelocitiesToTemperature(self.temperature)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        simulation.reporters.append(
            app.PDBReporter(
                file=os.path.join(self.output_dir, "trajectory.pdb"),
                reportInterval=self.record_interval,
            )
        )
        simulation.reporters.append(
            app.StateDataReporter(
                file=os.path.join(self.output_dir, "statedata.txt"),
                reportInterval=self.record_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=total_steps,
                separator=",",
            )
        )
        simulation.reporters.append(
            ForceReporter(
                file=os.path.join(self.output_dir, "forces.txt"),
                reportInterval=self.record_interval,
            )
        )

        total_time = self.timestep * total_steps
        record_interval = self.record_interval * self.timestep

        logger(
            f"Running production for {total_time} with timestep {self.timestep} ps and record interval of {record_interval}"
        )
        simulation.step(total_steps)

        logger(f"Simulation done. Saving output to {self.output_dir}")


class LAMMPSCHIKCalculator:
    def __init__(
        self,
        tmp_dir: str = f"{os.getcwd()}/tmp_{time.strftime('%d%b%y_%H.%M.%S', time.localtime())}",
        pot_dir: str = f"{os.getenv('PROJECTSDIR')}/NeuralForceField/unc_eabf/experiments/silica/scripts/supercloud/lammps",
        lmp_exe: str = f"{os.getenv('HOME')}/rgb_shared/pleon/lammps/src/lmp_serial",
        charges_dict: dict = {14: 1.910418, 8: -0.955209},
        specorder: List[str] = ["Si", "O"],
    ):
        self.temp_dir = tmp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.pot_dir = pot_dir
        self.lmp_exe = lmp_exe
        self.charges_dict = charges_dict
        self.specorder = specorder

    def close(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def write_lammps_script(
        self,
        input_struc: str,
        output_struc: str,
        script_path: str,
    ):
        lammps_script = """## Script to run single point calculation using CHIK potential in LAMMPS

variable		cutoff equal 6.5				# Cut-off distance for the Buckingham term (Angstrom in metal units)

## This part defines units and atomic information
# General
units			metal
atom_style		charge
timestep	    1e-3  # (ps in metal units)

# Read in the initial structure
read_data		INPUT_STRUC_FILE

# Atomic Information
mass			1 28.085500
mass			2 15.999400
set			    type 1 charge +1.910418
set			    type 2 charge -0.955209


## This part implements the CHIK pair potential with a cut-off distance for
## the Buckingham term. Long range Coulomb interactions are evaluated with the
## pppm method.

# Buckingham potential with Coulombic interactions
pair_style    	hybrid/overlay buck/coul/long ${cutoff} table linear 5000
# I | J | A (eV) | rho (Angstrom) | C (eV) | cutoff (Angstrom)
pair_coeff    	1 1 buck/coul/long 3150.4 0.35075 626.7  # Si-Si
pair_coeff   	1 2 buck/coul/long 27029.0 0.19385 148.0  # Si-O
pair_coeff    	2 2 buck/coul/long 659.5 0.38610 26.8    # O-O

# Define D/r^24 term using the interpolation table
pair_coeff      1 1 table ${pot_dir}/chik_si_si.table Si_Si # Si-Si
pair_coeff      1 2 table ${pot_dir}/chik_si_o.table Si_O # Si-O
pair_coeff      2 2 table ${pot_dir}/chik_o_o.table O_O   # O-O

kspace_style	pppm 1.0e-4

# Neighbor style
neighbor		3.0 bin
neigh_modify	check yes every 1 delay 0 page  100000 one 10000
group      		Si  type 1
group       	O   type 2

variable        varetotal equal etotal
variable        varke equal ke
variable        varpe equal pe
variable        varvol equal vol
variable        varpress equal press
variable        vardensity equal density
variable        varlx equal lx
variable        varly equal ly
variable        varlz equal lz

# Thermo settings
fix             thermo_print all print 1 "${varpress} ${varvol} ${vardensity} ${varetotal} ${varke} ${varpe} ${varlx} ${varly} ${varlz}" file OUTPUT_STRUC_FILE.thermo screen no
thermo          1
thermo_style    custom press vol density etotal ke pe lx ly lz

# Run a single point calculation (0 steps)
run 			0

unfix           thermo_print

# Output forces for all atoms
dump            1 all custom 1 OUTPUT_STRUC_FILE.forces id type x y z fx fy fz
run             0
undump          1

# Save the final configuration
write_data      OUTPUT_STRUC_FILE
"""
        lammps_script = lammps_script.replace("INPUT_STRUC_FILE", input_struc)
        lammps_script = lammps_script.replace("OUTPUT_STRUC_FILE", output_struc)

        with open(script_path, "w") as f:
            f.write(lammps_script)

        return lammps_script

    def write_geom_as_lammps_data(self, geom: Atoms, output_path: str):
        charges = [self.charges_dict[n] for n in geom.get_atomic_numbers()]
        geom.set_initial_charges(charges)

        write_lammps_data(
            output_path,
            atoms=geom,
            atom_style="charge",
            masses=True,
            units="metal",
            specorder=self.specorder,
        )

    def single_point_calc(
        self, geom: Atoms, identifier: str, remove_files: bool = False
    ) -> Atoms:
        lammps_script = os.path.join(self.temp_dir, f"{identifier}.in")
        input_path = os.path.join(self.temp_dir, f"{identifier}_in.data")
        output_path = os.path.join(self.temp_dir, f"{identifier}_out.data")
        log_path = os.path.join(self.temp_dir, f"{identifier}.log")

        self.write_geom_as_lammps_data(geom, input_path)
        self.write_lammps_script(input_path, output_path, lammps_script)

        # Run LAMMPS
        result = subprocess.run(
            [
                self.lmp_exe,
                "-in",
                lammps_script,
                "-log",
                log_path,
                "-var",
                "pot_dir",
                self.pot_dir,
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"LAMMPS run failed for {identifier}")

        # Read the output file
        geom_o = read_lammps_data(
            output_path,
            Z_of_type={i + 1: Atom(at).number for i, at in enumerate(self.specorder)},
            sort_by_id=True,
            read_image_flags=True,
            units="metal",
            atom_style="charge",
        )
        geom_f = read(
            f"{output_path}.forces",
            format="lammps-dump-text",
            index="0",
            order=True,
            specorder=self.specorder,
        )
        # NOTE: somehow at.get_forces() does not work after at.wrap(), so I have to read it before doing the coordinate wrap
        forces = geom_f.get_forces()

        # wrap positions such that when compared to the original geometry, they are the same
        geom_o.wrap()
        geom_f.wrap()

        # Check if the atoms match exactly
        if not (
            np.allclose(geom_f.get_atomic_numbers(), geom_o.get_atomic_numbers())
            and np.allclose(geom_f.get_positions(), geom_o.get_positions())
            and np.allclose(geom_f.get_cell().array, geom_o.get_cell().array)
        ):
            warnings.warn(
                f"Mismatch between forces and output geometry for {identifier}"
            )
            logger(f"Mismatch between forces and output geometry for {identifier}")

            return None  # Note: the files were not removed if debug is needed

        with open(f"{output_path}.thermo", "r") as f:
            thermo_output = f.readlines()
            thermo_output = [ln.strip("\n").split(" ") for ln in thermo_output[1:]]

        header = ["press", "vol", "density", "etotal", "ke", "pe", "lx", "ly", "lz"]
        thermo_output = pd.DataFrame(thermo_output, columns=header, dtype=float)

        geom_o.info["energy"] = thermo_output["pe"].values[0]
        geom_o.arrays["forces"] = forces

        # Optional: Clean up the generated input and output files
        if remove_files:
            os.remove(lammps_script)
            os.remove(input_path)
            os.remove(output_path)
            os.remove(f"{output_path}.forces")
            os.remove(f"{output_path}.thermo")
            os.remove(log_path)

        return geom_o

    def calc(self, geoms: List[Atoms], remove_files: bool = False) -> List[Atoms]:
        output_geoms = []
        for i, geom in enumerate(geoms):
            output_geom = self.single_point_calc(
                geom, f"geom_{i}", remove_files=remove_files
            )
            output_geoms.append(output_geom)

        # clean up the temporary directory
        if remove_files:
            self.close()

        return output_geoms


def handle_calc(config):
    # read dataset
    logger("Reading dataset")

    dset = Dataset.from_file(config["enhsamp"]["sampled_path"])

    if config["system"] == "ala":
        from .ala import Energy

        # initialize calculator
        logger("Initializing OpenMM calculator")
        calculator = OpenMMCalculator(
            params=config.get("openmm"),
        )

        # run single point calculation for each geometry in the dataset
        logger("Running single point calculations")
        energies, energy_grads = [], []
        for i, geom in enumerate(tqdm(dset)):
            energy, forces = calculator.single_point_calc(geom)
            energy = Energy.scale_energy(energy, config["openmm"]["forcefield"])
            energies.append(torch.FloatTensor([energy]).squeeze())
            energy_grads.append(-torch.FloatTensor(forces))

        dset.props["energy"] = energies
        dset.props["energy_grad"] = energy_grads

        # save dataset
        logger("Saving dataset")
        dset.save(config["enhsamp"]["sampled_path"])

        logger("Done")

        return dset

    elif config["system"] == "silica":
        from .silica import Energy

        geoms = []
        for i, d in enumerate(dset):
            geom = Atoms(
                symbols=d["nxyz"][:, 0],
                positions=d["nxyz"][:, 1:],
                cell=d["lattice"],
                pbc=True,
            )
            geoms.append(geom)

        # initialize calculator
        logger("Initializing LAMMPS calculator")
        if os.getenv("LMP_EXE") is not None:
            calculator = LAMMPSCHIKCalculator(
                lmp_exe=os.getenv("LMP_EXE"),
            )
        else:
            calculator = LAMMPSCHIKCalculator()

        # run single point calculation for each geometry in the dataset
        logger("Running single point calculations")
        output_geoms = calculator.calc(
            geoms=geoms,
            remove_files=True,
        )

        energies, energy_grads = [], []
        for i, geom in enumerate(output_geoms):
            if geom is None:
                energies.append(torch.FloatTensor([np.nan]))
                energy_grads.append(torch.FloatTensor([np.nan]))
                continue

            energy = geom.info["energy"]
            forces = geom.arrays["forces"]
            # energy = Energy.scale_energy(energy)
            energies.append(torch.FloatTensor([energy]).squeeze())
            energy_grads.append(-torch.FloatTensor(forces))

        dset.props["energy"] = energies
        dset.props["energy_grad"] = energy_grads

        # save dataset
        logger("Saving dataset")
        dset.save(config["enhsamp"]["sampled_path"])

        logger("Done")

        return output_geoms

    else:
        raise ValueError(f"System {config['system']} not supported")
