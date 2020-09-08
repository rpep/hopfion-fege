import sys
sys.path.append('../src')
from helpers import finmag
import commands
import os
import time
import pprint
import dolfin as df
from aeon import timer
from finmag.util.meshes import from_geofile
from finmag import Simulation
from finmag.energies import Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# Setup
setupstart = time.time()
mesh = df.Mesh(os.path.join(MODULE_DIR, "bar.xml.gz"))
sim = Simulation(mesh, Ms=0.86e6, unit_length=1e-9)
sim.set_m((1, 0, 1))
demag = Demag()
demag.parameters["phi_1_solver"] = "minres"
demag.parameters["phi_2_solver"] = "gmres"
demag.parameters["phi_2_preconditioner"] = "sor"
sim.add(demag)
sim.add(Exchange(13.0e-12))

# Dynamics
dynamicsstart = time.time()
sim.run_until(3.0e-10)
endtime = time.time()

# Write output to results.rst
output = open(os.path.join(MODULE_DIR, "results.rst"), "a")
output.write("\nFinmag results:\n")
output.write("---------------\n")
output.write("Setup: %.3f sec.\n" % (dynamicsstart - setupstart))
output.write("Dynamics: %.3f sec.\n" % (endtime - dynamicsstart))
output.write("\nFinmag solver parameters:\n")
output.write("-------------------------\n")
pp = pprint.PrettyPrinter()
output.write("\nfirst linear solve\n{}\n".format(pp.pformat(demag._poisson_solver.parameters.to_dict())))
output.write("\nsecond linear solve\n{}\n\n".format(pp.pformat(demag._laplace_solver.parameters.to_dict())))
output.write(str(timer))
output.close()

# Cleanup
files = ["bar_bi.xml", "bar.grid", "bar_mat.xml", "bar.neutral", "bar.xml.bak", "run_nmag_log.log", "bar_dat.ndt"]
for file in files:
    fname = os.path.join(MODULE_DIR, file)
    if os.path.isfile(fname):
        os.remove(fname)
