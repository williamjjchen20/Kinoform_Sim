import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, functools

from propagators import angular_spectrum_method
from classes import *


savedir ="test_figs/simulation_figs"
if __name__ == "__main__":
    width, height = 0.06, 0.06
    Lx, Ly, Lz = width, height, 10000
    N=1024
    
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    source = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=1.0)
    print(source.simulation)
    
    source.view(savedir=savedir)
    lens = ThinLens(f=1.0, R=0.25, simulation=simulation, z=0.5, wavelength=source.wavelength)
    z = np.abs(lens.f)
    
    ## Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*z/(2*lens.R) 
    print("Nyquist Sampling Rate:", f_s)
    assert(Lx/N < f_s and Ly/N < f_s)
