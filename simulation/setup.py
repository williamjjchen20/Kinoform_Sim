from ..optics import *

def build_simulation(N=2048, Lx=1.5e-4, Ly=1.5e-4, Lz=10000):
    return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)


    
    

    
    
    