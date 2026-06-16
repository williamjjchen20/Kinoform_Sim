import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import xraylib as xrl
import os, sys, functools

from propagators import *
from metrics import *
from classes import *

savedir = "test_figs/"

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
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny = N)
    propagator = functools.partial(angular_spectrum_method, dim=2)
    
    source = GaussianBeam(energy=E, simulation=simulation, z=0, w0=4e-5)
    source.view(savedir=savedir)
    print(source.intensity().shape)
    fwhm = FWHM(source)
    print(fwhm)
    Imax, Iavg = intensity_stats(source)
    print(Imax, Iavg)
    P = total_power(source)
    print(P)
    
if __name__ == "__main__":
    test_kinoform()