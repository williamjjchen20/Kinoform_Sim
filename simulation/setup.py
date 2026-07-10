from ..optics import *

def build_simulation(N, Lx, Ly = None, Lz=1000):
    return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

def main():
    pass

def rayleigh(wavelength, f, D):
    return wavelength * f/D
    

if __name__ == "__main__":
    main()
