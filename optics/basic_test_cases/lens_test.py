import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import xraylib as xrl
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
    lens = OpticalLens(f=10, R=0.01, n=1.5, simulation=simulation, z=0)
    lens.init_transmittance(source)
    lens.view(ax=ax[0, 0])
    source.view(ax=ax[0, 1])
    
    # ## Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R) 
    print("Nyquist Sampling Rate:", f_s)
    assert(Lx/N < f_s and Ly/N < f_s)
    
    z1, z2 = lens.center[-1], lens.f
    print("Initial Max Intensity:", np.max(source.intensity()))
    source.propagate(z1, propagator)
    lens.transform(source)
    print("Post-Lens Max Intensity:", np.max(source.intensity()))
    source.propagate(z2, propagator)
    source.view(ax=ax[0, 2])
    print("Final Max Intensity:", np.max(source.intensity()))
    
    ax[0,0].set(title=f"Transmittance Phase (R={lens.R*1000} mm, f={lens.f} m)")
    ax[0,1].set(title=f"Initial Beam")
    ax[0,2].set(title=f"Propagated Beam (z={lens.f} m)")
    plt.savefig(os.path.join(savedir, "2D_Thin_Lens"))
    
def test_lens_xray():
    print("Testing Parabolic Lens (X-ray)...")
    width, height = 1.5e-4, 1.5e-4
    Lx, Ly, Lz = width, height, 10000
    N=2048
    
    # parameters for simulation
    E = 8.5e3 # keV
    f = 1. # m
    R = 5e-5
    n = xrl.Refractive_Index("Si", E/1000, 2.329)
    print("Refractive Index:", n)
    assert (R < Lx and R < Ly)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), squeeze=False, sharey=True)
    plt.subplots_adjust(wspace=2.0)
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)
    
    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = XrayParabolicLens(f=f, R=R, n=n, simulation=simulation, z=0)   
    lens.profile(source.wavelength, ax=plt.figure().gca(), savedir=savedir)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True) 
    
    lens.init_transmittance(source)
    lens.view(ax=ax[0, 0], show_label=True)
    source.view(ax=ax[0, 1])
    
    ## Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R) 
    # print("Nyquist Sampling Rate:", f_s)
    # print("Required (Nx, Ny):", Lx/f_s, Ly/f_s)
    assert(Lx/N < f_s and Ly/N < f_s)
    
    z1, z2 = lens.center[-1], lens.f
    print("Initial Max Intensity:", np.max(source.intensity()))
    source.propagate(z1, propagator)
    lens.transform(source)
    print("Post-Lens Max Intensity:", np.max(source.intensity()))
    source.propagate(z2, propagator)
    source.view(ax=ax[0, 2], show_label=True)
    print("Final Max Intensity:", np.max(source.intensity()))
    
    ax[0,0].set(title=rf"Transmittance Phase(R={lens.R*1e6} $\mu m$, f={lens.f} m)")
    ax[0,1].set(title=f"Initial Beam")
    ax[0,2].set(title=f"Propagated Beam (z={lens.f} m)")
    plt.savefig(os.path.join(savedir, "Xray_Parabolic_Lens"))


def test_kinoform():
    print("Testing Kinoform (X-ray)...")
    width, height = 1.5e-4, 1.5e-4
    Lx, Ly, Lz = width, height, 10000
    N=2048
    
    # parameters for simulation
    E = 8.5e3 # keV
    f = 1. # m
    R = 5e-5
    n = xrl.Refractive_Index("Si", E/1000, 2.329)
    print("Refractive Index:", n)
    assert (R < Lx and R < Ly)
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)
    
    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(f=f, R=R, n=n, simulation=simulation, z=0)    
    lens.profile(source.wavelength, ax=plt.figure().gca(),savedir=savedir)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True)
    lens.init_transmittance(source)
    lens.view(ax=ax[0, 0], show_label=True)
    source.view(ax=ax[0, 1])
    
    ## Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R) 
    # print("Nyquist Sampling Rate:", f_s)
    # print("Required (Nx, Ny):", Lx/f_s, Ly/f_s)
    assert(Lx/N < f_s and Ly/N < f_s)
    
    z1, z2 = lens.center[-1], lens.f
    print("Initial Max Intensity:", np.max(source.intensity()))
    source.propagate(z1, propagator)
    lens.transform(source)
    print("Post-Lens Max Intensity:", np.max(source.intensity()))
    source.propagate(z2, propagator)
    source.view(ax=ax[0, 2], show_label=True)
    print("Final Max Intensity:", np.max(source.intensity()))
    
    ax[0,0].set(title=f"Transmittance Phase (R={lens.R*1000} mm, f={lens.f} m)")
    ax[0,1].set(title=f"Initial Beam")
    ax[0,2].set(title=f"Propagated Beam(z={lens.f} m)")
    plt.savefig(os.path.join(savedir, "Ideal_Kinoform"))

if __name__ == "__main__":
    # test_standard_lens()
    test_lens_xray()
    # test_kinoform()
    