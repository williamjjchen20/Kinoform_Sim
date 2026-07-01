import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import os, sys, functools
from pathlib import Path

from ..propagators import *
from ..classes import *
from ..metrics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/propagation_test").resolve()

F = WaveFunctions()

def test_gaussian_1D(propagation_func):
    print("Testing 1D Gaussian Beam...")
    Lx, Lz = 6e-3, 1e5
    N = 512
    z = 2.5
    w0 = 0.5e-3

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True)
    plt.subplots_adjust(wspace=0.3)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)
    print(wave)

    ax[0, 0].plot(wave.grid, wave.intensity(), color="Black", alpha=0.7, label="Simulated")
    ax[0, 0].set(title="Initial Beam (z=0 m)")

    wave.propagate(z, propagation_func(simulation))
    print("Peak intensity:", np.max(wave.intensity()))
    ax[0, 1].plot(wave.grid, wave.intensity(), color="Black", label="Simulated")

    x = np.linspace(-Lx/2, Lx/2, N)
    U0 = F.gaussian_beam_1D(x, z=0, wavelength=wave.wavelength, w0=w0)
    ax[0, 0].plot(x, np.abs(U0)**2, color="Red", alpha=0.7, ls="--", label="Theoretical")
    U = F.gaussian_beam_1D(x, z=z, wavelength=wave.wavelength, w0=w0)
    I = np.abs(U)**2
    print("Peak intensity:", np.max(I))
    ax[0, 1].plot(x, I, color="Red", ls="--", label="Theoretical")
    ax[0, 1].set(title=f"Propagated Beam (z={z:.2f} m)")

    ax[0, 0].legend()
    ax[0, 1].legend()

    I[I == 0] = 1e-10
    err = np.abs(wave.intensity() - I)
    rel_err = err / I
    print("Maximum Error:", np.max(err))
    print("Average Relative Error:", np.mean(rel_err))
    ax[0, 2].plot(x, rel_err)
    ax[0, 2].set(title="Relative Error")
    plt.savefig(os.path.join(savedir, "1D_Gaussian_Beam_Validation"))


def test_gaussian_2D(propagation_func):
    print("Testing 2D Gaussian Beam...")
    Lx, Ly, Lz = 1e-2, 1e-2, 100
    N = 512
    z = 3.0
    w0 = 0.5e-3

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), squeeze=False)
    plt.subplots_adjust(wspace=0.3)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)
    print(wave)

    im = ax[0, 0].imshow(wave.intensity(), cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)")

    wave.propagate(z, AngularSpectrum(simulation))
    print("Peak intensity:", np.max(wave.intensity()))
    ax[0, 1].imshow(wave.intensity(), cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Simulated Beam (z={z:.2f} m)")

    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = F.gaussian_beam_2D(X, Y, z=z, wavelength=wave.wavelength, w0=w0)
    I = np.abs(U)**2
    print("Peak intensity:", np.max(I))
    ax[0, 2].imshow(I, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 2].set(title=f"Theoretical Beam (z={z:.2f} m)")

    I[I == 0] = 1e-10
    err = np.abs(wave.intensity() - I)
    rel_err = err / I
    print("Maximum Error:", np.max(err))
    print("Average Relative Error:", np.mean(rel_err))
    ax[0, 3].imshow(rel_err, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 3].set(title="Relative Error")

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "2D_Gaussian_Beam_Validation"))


def test_constant_1D(propagation_func):
    print("Testing 1D Constant Beam...")
    Lx, Lz = 6e-3, 1e5
    N = 512
    z = 2.5

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True)
    plt.subplots_adjust(wspace=0.3)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave = ConstantBeam(energy=1.96, simulation=simulation, z=0)
    print(wave)

    ax[0, 0].plot(wave.grid, wave.intensity(), color="Black", alpha=0.7, label="Simulated")
    ax[0, 0].set(title="Initial Beam (z=0 m)")

    wave.propagate(z, propagation_func(simulation))
    print("Peak intensity:", np.max(wave.intensity()))
    ax[0, 1].plot(wave.grid, wave.intensity(), color="Black", label="Simulated")

    x = np.linspace(-Lx/2, Lx/2, N)
    U0 = F.const_wave_1D(x, z=0, wavelength=wave.wavelength)
    ax[0, 0].plot(x, np.abs(U0)**2, color="Red", alpha=0.7, ls="--", label="Theoretical")
    U = F.const_wave_1D(x, z=z, wavelength=wave.wavelength)
    I = np.abs(U)**2
    print("Peak intensity:", np.max(I))
    ax[0, 1].plot(x, I, color="Red", ls="--", label="Theoretical")
    ax[0, 1].set(title=f"Propagated Beam (z={z:.2f} m)")

    ax[0, 0].legend()
    ax[0, 1].legend()

    I[I == 0] = 1e-10
    err = np.abs(wave.intensity() - I)
    rel_err = err / I
    print("Maximum Error:", np.max(err))
    print("Average Relative Error:", np.mean(rel_err))
    ax[0, 2].plot(x, rel_err)
    ax[0, 2].set(title="Relative Error")
    plt.savefig(os.path.join(savedir, "1D_Constant_Beam_Validation"))


if __name__ == "__main__":
    # test_gaussian_1D(AngularSpectrum)
    # test_gaussian_2D(AngularSpectrum)
    test_constant_1D(ScaledAngularSpectrum)
