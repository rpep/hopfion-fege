import sys
sys.path.append('../src/')
from helpers import finmag
import textwrap 
import os
import sys
import os; import dolfin as df; import gc; import sys;
import pickle; import numpy as np; from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, Demag, Zeeman, UniaxialAnisotropy
from dolfinh5tools import openh5



def load_h5_field(sim, filename):
    h5file = df.HDF5File(sim.mesh.mpi_comm(), filename, 'r')
    field = df.Function(sim.S3)
    h5file.read(field, 'm')
    h5file.close()
    return field

def save_h5_field(field, mesh, filename, fieldname='m'):
    h5file = df.HDF5File(mesh.mpi_comm(), filename, 'w')
    h5file.write(mesh, 'mesh')
    h5file.write(field, fieldname)
    del h5file


def load_m_h5(sim, filename):
    print("\n\nLoading {} to field\n\n".format(filename))
    h5file = df.HDF5File(sim.mesh.mpi_comm(), filename, 'r')
    field = df.Function(sim.S3)
    h5file.read(field, 'm')
    h5file.close()
    return field

def schedule_m_h5(sim, filename, inc=[0], **kwargs):
    print(kwargs)
    f = filename.split('.')
    filen = '.'.join(f[:-1])
    filename = "{}_{:06}.h5".format(filen, inc[0])
    print('Saving Field to h5 file {}'.format(filename))
    save_h5_field(sim.llg._m_field.f, sim.mesh, filename)
    print('Saved Field to h5 file (Snapshot #{})'.format(inc[0]))
    inc[0] += 1


def disk_with_internal_layers(d, layer_positions, lmax, name=''):
    """Creates a disk mesh with a flat interface inside.
    Args:
        d - disk diameter
        layer_positions - assuming layer 0 is at height 0, layer 1 is at layer_positions[0], etc...
        lmax - discretisation

    """
    # gmsh geometry script (obtained using gmsh)
    geo_script = textwrap.dedent("""\
    lmax = DefineNumber[ $lmax$, Name "Parameters/lmax" ];
    rad = DefineNumber[ $rad$, Name "Parameters/rad" ];
    Point(1) = {0, 0, 0, lmax};
    Point(2) = {rad, 0, 0, lmax};
    Point(3) = {-rad, 0, 0, lmax};
    Point(4) = {0, rad, 0, lmax};
    Point(5) = {0, -rad, 0, lmax};
    Circle(1) = {4, 1, 2};
    Circle(2) = {2, 1, 5};
    Circle(3) = {5, 1, 3};
    Circle(4) = {3, 1, 4};
    Line Loop(5) = {4, 1, 2, 3};
    Ruled Surface(6) = {5};
    """)
    for i, l in enumerate(layer_positions):
        if i == 0:
            geo_script += textwrap.dedent("""\
            
                out1[] = Extrude {{0, 0, {}}} {{
                    Surface{{6}};
                }};

            """).format(l)
        else:
            geo_script += textwrap.dedent("""\
            
                out{}[] = Extrude {{0, 0, {}}} {{
                    Surface{{out{}[0]}};
                }};
                
            """).format(i+1, l-layer_positions[i-1], i)

    # Replace parameters in the gmsh geometry script.
    geo_script = geo_script.replace('$rad$', str(d/2.))
    geo_script = geo_script.replace('$lmax$', str(lmax))

    #print(geo_script)
    
    # Write the geometry script to the .geo file.
    # basename = 'disk_with_boundary-{}-{}-{}-{}-{}'.format(name, str(d/2.0), jid, aid, '_'.join(layers))
    basename = 'disk_with_boundary-{}-{}'.format(name, str(d/2.0))
    print('\n\nMESH FILENAMES = {}\n\n'.format(basename))
    geo_file = open(basename + '.geo', 'w')
    geo_file.write(geo_script)
    geo_file.close()

    # Create a 3d mesh.
    gmsh_command = 'gmsh {}.geo -3 -optimize_netgen -o {}.msh'.format(basename, basename)
    os.system(gmsh_command)

    # Convert msh mesh format to the xml (required by dolfin).
    dc_command = 'dolfin-convert {}.msh {}.xml'.format(basename, basename)
    os.system(dc_command)

    # Load the mesh and create a dolfin object.
    mesh = df.Mesh('{}.xml'.format(basename))

    # Delete all temporary files.
    # os.system('rm {}.geo {}.msh {}.xml'.format(basename, basename, basename))

    return mesh


L = 70.0
top_and_bottom_height = 0
middle_height = L
diameter=3*L

#layers = [top_and_bottom_height,
#          top_and_bottom_height+middle_height,
#          2*top_and_bottom_height+middle_height]

layers=[L]
mesh = disk_with_internal_layers(d=diameter, layer_positions=layers, lmax=3)


print(finmag.util.meshes.mesh_quality(mesh))

def K_init(K, layers):
    def wrapped_function(pos):
        x, y, z = pos
        # The mesh was created so that the boundary between grains is
        # at z=0.
        if z <= layers[0]:
            return K
        elif z > layers[0] and z <= layers[1]:
            return 0
        else:
            return K
    return wrapped_function



Ms = 3.84e5  # magnetisation saturation (A/m)
A = 8.78e-12  # exchange energy constant (J/m)
D = 1.58e-3  # DMI constant (J/m**2)

#K = 100*(4*np.pi*1e-7)*Ms**2
#print('K = {}'.format(K))

# initial magnetisation configuration


def m_init(pos):
    x, y, z = pos
    rho = (x**2 + y**2)**0.5
    theta = np.arctan2(y, x)
    if z > top_and_bottom_height + df.DOLFIN_EPS and z < middle_height+top_and_bottom_height - df.DOLFIN_EPS:
         mx = np.sin(2.0*np.pi*rho/L)*np.sin(theta)
         my = -np.sin(2.0*np.pi*rho/L)*np.cos(theta)
         mz = np.cos(2.0*np.pi*rho/L)
         return mx, my, mz
    else:
        return (0, 0, 1)


def m_init(pos):
    """
    Initial state corresponding to Hopf ansatz mapped to a cylinder;
    see Paul Sutcliffe's Hopfion paper for details
    """
    x, y, z = pos
    # Shift coordinates in the z-direction as our mesh is not origin centred.
    z -= L/2.0
    rho = (x**2 + y**2)**0.5
    theta = np.arctan2(y, x)
    omega = np.tan(np.pi*z/L)
    sigma = (1 + (2*z/L)**2)*(1/np.cos(np.pi*rho/(2*L)))/L
    gamma = sigma**2 * rho ** 2 + omega**2 / 4.0
    mx = 4*sigma*rho * (omega*np.cos(theta) - (gamma - 1)*np.sin(theta)) / (1 + gamma)**2
    my = 4*sigma*rho * (omega * np.sin(theta) + (gamma-1)*np.cos(theta)) / (1 + gamma)**2
    mz = 1 - (8 * sigma**2 * rho**2) / (1 + gamma)**2
    return mx, my, mz

name = "cylinder_rad_{}_h_{}".format(diameter,
                                            top_and_bottom_height,
                                            middle_height)

sim = Sim(mesh, Ms, unit_length=1e-9, name=name)


# Add energies. No Zeeman because the system is in zero field.
sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Demag())
#sim.add(UniaxialAnisotropy(K_init(K, layers=layers), (0, 0, 1)))

def borders_pinning(pos):
    x, y, z = pos[0], pos[1], pos[2]
    if (z < df.DOLFIN_EPS) or (z >  L*0.99):
        return True
    else:
        return False

# Pass the function to the 'pins' property of the simulation
sim.pins = borders_pinning

# Turn off precession.
sim.do_precession = False

# Initialise the system.
sim.set_m(m_init)

save_h5_field(sim.llg._m_field.f, sim.mesh, 'initial_m.h5')

sim.save_vtk('initial.pvd', overwrite=True)
sim.schedule('save_vtk', every=5e-12, filename='vtks/m.pvd')
sim.schedule(schedule_m_h5, every=5e-12, filename='h5s/m.h5')
# relax with high precision
sim.relax(1e-6)

