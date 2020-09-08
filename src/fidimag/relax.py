import numpy as np
import fidimag
from matplotlib import cm
import matplotlib.pyplot as plt

L = 70 # Helical Length
D = 3*L # Diameter
h = L # Height

Lx = D
Ly = D
Lz = h
d = 2.5

Ms = 3.84e5  # magnetisation saturation (A/m)
Aex = 8.78e-12  # exchange energy constant (J/m)
Dbulk = 1.58e-3  # DMI constant (J/m**2)

nx = int(Lx/d)
ny = int(Ly/d)
nz = int(Lz/d)

mesh = fidimag.common.CuboidMesh(nx=nx, ny=ny, nz=nz,
                                 dx=d, dy=d, dz=d,
                                 x0=-Lx/2.0, y0 =-Ly/2.0,
                                 z0=-Lz/2.0,
                                 unit_length=1e-9)

def Ms_init(pos):
    x, y, z = pos
    if x**2 + y**2 <= D/2.0:
        return Ms
    else:
        return 0.0

def m_init(pos):
    """
    Initial state corresponding to Hopf ansatz mapped to a cylinder;
    see Paul Sutcliffe's Hopfion paper for details
    """
    x, y, z = pos
    # Shift coordinates in the z-direction as our mesh is not origin centred.
    #z -= L/2.0
    rho = (x**2 + y**2)**0.5
    theta = np.arctan2(y, x)
    omega = np.tan(np.pi*z/L)
    sigma = (1 + (2*z/L)**2)*(1/np.cos(np.pi*rho/(2*L)))/L
    gamma = sigma**2 * rho ** 2 + omega**2 / 4.0
    mx = 4*sigma*rho * (omega*np.cos(theta) - (gamma - 1)*np.sin(theta)) / (1 + gamma)**2
    my = 4*sigma*rho * (omega * np.sin(theta) + (gamma-1)*np.cos(theta)) / (1 + gamma)**2
    mz = 1 - (8 * sigma**2 * rho**2) / (1 + gamma)**2
    return mx, my, mz
    
def pins_init(pos):
    x, y, z = pos
    
    if z == -L/2.0 + d/2.0 or z == L/2.0 - d/2.0:
        return 1
    else:
        return 0
    
sim = fidimag.micro.Sim(mesh)
sim.set_Ms(Ms)
sim.set_m(m_init)
sim.set_pins(pins_init)
sim.driver.alpha = 1.0

sim.add(fidimag.micro.UniformExchange(Aex))
sim.add(fidimag.micro.DMI(Dbulk))
sim.add(fidimag.micro.Demag())

sim.relax(stopping_dmdt=0.1, save_vtk_steps=25)
sim.save_vtk()
