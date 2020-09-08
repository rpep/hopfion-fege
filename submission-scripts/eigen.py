import helpers as hlp

finmag=hlp.finmag
import os
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
from finmag import NormalModeSimulation as Sim
from finmag.energies import Exchange, DMI, UniaxialAnisotropy, Zeeman
from finmag.util.dmi_helper import find_skyrmion_center_2d
from finmag.util.helpers import set_logging_level
from finmag.util import meshes
import finmag

rdir = '../results/relax'
os.chdir(rdir)

h5_file = df.HDF5File(df.mpi_comm_world(), 'relaxed-nanotrack-1e-7.h5', 'r')
mesh = df.Mesh()
h5_file.read(mesh, 'mesh', False)


sim = Sim(mesh=mesh, Ms=5.8e5, unit_length=1e-9, pbc=None)
sim.add(UniaxialAnisotropy(K1=6e5, axis=[0, 0, 1]))
sim.add(Exchange(A=1.5e-11))
sim.add(DMI(D=3e-3))
sim.add(Zeeman((0, 0, 1e5)))
sim.alpha = 0.5


print('Trying to load relaxed state')
m = hlp.load_h5_field(sim, 'relaxed-nanotrack-1e-7.h5')
sim.set_m(m)
sim.do_precession = False

print('Computing eigenmodes!\n\n\n')
n_eigenmodes = 50 # Calculate the first 50 eigenmodes.
complex_freqs = sim.compute_normal_modes(n_values=n_eigenmodes)
f_array = np.real(complex_freqs[0])

np.save('frequencies.npy', f_array)

sim.export_eigenmode_animations(n_eigenmodes, directory='eigenmode-anims',
                                num_cycles=1,
                                num_snapshots_per_cycle=30,
                                save_h5=True)



