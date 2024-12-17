import weakref
from ase import units
from ase.parallel import world
from ase.utils import IOContext

class MDLogger(IOContext):
    """Additional Class for logging biased molecular dynamics simulations.

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.

    header:        Whether to print the header into the logfile.

    logfile:       File name or open file, "-" meaning standard output.

    mode="a":      How the file is opened if logfile is a filename.
    """

    def __init__(
        self, dyn, atoms, logfile, header=True, peratom=False, verbose=False, mode="a"
    ):
        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None

        self.atoms = atoms

        self.stress = False
        if (
            "stress" in self.atoms.calc.implemented_properties
            and self.atoms.cell.rank == 3
        ):
            self.stress = True

        self.logfile = self.openfile(logfile, comm=world, mode=mode)

        if self.dyn is not None:
            self.hdr = "%-10s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            raise ValueError("A dynamics object has to be attached to the logger!")

        self.hdr_properties = []
        for prop in self.atoms.calc.implemented_properties:
            if prop == "stress" and self.stress is True:
                self.hdr += "%12s " % ("Pressure[GPa]",)
                self.fmt += "%12.5f "
                self.hdr += "%12s " % ("Volume[A^3]",)
                self.fmt += "%12.5f "
                self.hdr_properties.append(prop)
                continue

            if prop in ["free_energy", "energy_grad", "forces", "node_energy"]:
                continue

            self.hdr_properties.append(prop)

            if prop == "cv_vals":
                prop = "CV"  # rename the property for the header

            self.hdr += "%12s " % (prop,)
            self.fmt += "%12.5f "

        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

        self.verbose = verbose
        if verbose:
            print(self.hdr)

    def __del__(self):
        self.close()

    def __call__(self):
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()

        for prop in self.hdr_properties:
            if prop == "stress" and self.stress is True:
                stress = (
                    self.atoms.get_stress(voigt=False, include_ideal_gas=True)
                    * 1.60219e-19
                    / 1e-30
                    / 1e9
                )  # convert to GPa
                vol = self.atoms.get_volume()
                dat += (-stress.trace() / 3, vol)
                continue

            val = self.atoms.calc.get_property(prop).squeeze().item()
            dat += (val,)

        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt[-1] % dat)
