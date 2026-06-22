from ..optics import *

def build_simulation(N, Lx, Ly = None, Lz=1000):
    return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

