from classes import *

def initialize(L, N, dim=1):
    Lx, Lz = L, 100*L
    Ly = None if dim == 1 else L
    simulation = SimulationObject(Lx=Lx, Nx=N, Ly=Ly, Ny=N, Lz=Lz)
    
    return simulation

    
root = "test_figs"
if __name__ == "__main__":
    L  = 5.e-3
    N=512
    dim=2
    
    simulation = initialize(L, N, dim=dim)
    source = GaussianBeam(energy=1.96, simulation=simulation, z=0)#Waveform(energy=1.96, simulation=simulation, z=0, distribution_func=gaussian_beam_2D)
    aperture = SingleSlit(simulation=simulation, z=0.5, width=0.5e-3, height=0.5e-4)
    

    
    
    