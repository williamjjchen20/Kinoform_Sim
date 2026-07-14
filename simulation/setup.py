from ..optics import *

## Building functionalities 
def build_simulation(N, Lx, Ly = None, Lz=1000):
    return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

# simulation checks
def rayleigh(wavelength, f, D):
    return wavelength * f/D

def run(simulation: SimulationObject):
    pass



