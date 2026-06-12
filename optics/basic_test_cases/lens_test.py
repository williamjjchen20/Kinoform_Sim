import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, functools

from propagators import *
from classes import *


savedir ="test_figs/simulation_figs"

def test_standard_lens():
    width, height = 0.06, 0.06
    Lx, Ly, Lz = width, height, 10000
    N=1024
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True)
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)
    
    source = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=1.0)
    lens = ThinLens(f=10, R=0.01, n=1.5, simulation=simulation, z=0)
    lens.init_transmittance(source)
    lens.view(ax=ax[0, 0])
    source.view(ax=ax[0, 1])
    
    # ## Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R) 
    print("Nyquist Sampling Rate:", f_s)
    assert(Lx/N < f_s and Ly/N < f_s)
    
    z1, z2 = lens.center[-1], lens.f
    source.propagate(z1, propagator)
    lens.transform(source)
    source.propagate(z2, propagator)
    source.view(ax=ax[0, 2])
    
    ax[0,0].set(title=f"Thin Lens (R={lens.R*1000} mm, f={lens.f} m)")
    ax[0,1].set(title=f"Initial Beam")
    ax[0,2].set(title=f"Propagated Beam (z={lens.f} m)")
    plt.savefig(os.path.join(savedir, "2D_Thin_Lens"))

if __name__ == "__main__":
    test_standard_lens()