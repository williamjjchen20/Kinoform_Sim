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
    N = 3000
    z = 1.5
    w0 = 0.5e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)
    print(wave)

    x = np.linspace(-Lx/2, Lx/2, N)
    U0 = F.gaussian_beam_1D(x, z=0, wavelength=wave.wavelength, w0=w0)
    I0_th = np.abs(U0)**2
    I0_th_norm = I0_th / I0_th.max()
    I0_sim_norm = wave.intensity() / wave.intensity().max()

    wave.propagate(z, propagation_func(simulation))
    I_sim = wave.intensity()

    U = F.gaussian_beam_1D(x, z=z, wavelength=wave.wavelength, w0=w0)
    I_th = np.abs(U)**2

    print("Peak intensity (sim):", np.max(I_sim))
    print("Peak intensity (theory):", np.max(I_th))
    print("Peak Location:", wave.grid[np.argmax(I_sim)])

    I_sim_norm = I_sim / I_sim.max()
    I_th_norm = I_th / I_th.max()
    err = np.abs(I_sim_norm - I_th_norm)

    print(f"  Max Absolute Error (normalized): {err.max():.6e}")
    print(f"  Mean Absolute Error (normalized): {err.mean():.6e}")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), squeeze=False)
    plt.subplots_adjust(wspace=0.35)

    ax[0, 0].plot(x * 1e3, I0_sim_norm, color="black", alpha=0.7, label="Simulated")
    ax[0, 0].plot(x * 1e3, I0_th_norm, color="red", alpha=0.7, ls="--", label="Theoretical")
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlabel="x [mm]", ylabel="Normalized Intensity")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(x * 1e3, I_sim_norm, color="black", label="Simulated")
    ax[0, 1].plot(x * 1e3, I_th_norm, color="red", ls="--", label="Theoretical")
    ax[0, 1].set(title=f"Propagated Beam (z={z:.2f} m)", xlabel="x [mm]", ylabel="Normalized Intensity")
    ax[0, 1].legend(fontsize=8)

    ax[0, 2].plot(x * 1e3, err, color="red", linewidth=1.0)
    ax[0, 2].set(title="Absolute Error (normalized)", xlabel="x [mm]", ylabel="|Sim - Theory|")

    plt.savefig(os.path.join(savedir, "1D_Gaussian_Beam_Validation"))


def test_gaussian_2D(propagation_func):
    print("Testing 2D Gaussian Beam...")
    Lx, Ly, Lz = 6e-3, 6e-3, 1e5
    N = 3000
    z = 1.5
    w0 = 0.5e-3

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)
    print(wave)

    wave.propagate(z, propagation_func(simulation))
    I_sim = wave.intensity()

    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = F.gaussian_beam_2D(X, Y, z=z, wavelength=wave.wavelength, w0=w0)
    I_th = np.abs(U)**2

    print("Peak intensity (sim):", np.max(I_sim))
    print("Peak intensity (theory):", np.max(I_th))

    I_sim_norm = I_sim / I_sim.max()
    I_th_norm = I_th / I_th.max()
    err = np.abs(I_sim_norm - I_th_norm)

    print(f"  Max Absolute Error (normalized): {err.max():.6e}")
    print(f"  Mean Absolute Error (normalized): {err.mean():.6e}")

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    im0 = ax[0, 0].imshow(I_sim_norm, cmap="Greys_r", extent=extent)
    ax[0, 0].set(title=f"Simulated (z={z} m)", xlabel="x [m]", ylabel="y [m]")

    ax[0, 1].imshow(I_th_norm, cmap="Greys_r", extent=extent)
    ax[0, 1].set(title=f"Theoretical (z={z} m)", xlabel="x [m]", ylabel="y [m]")

    ax[0, 2].imshow(err, cmap="Greys_r", extent=extent)
    ax[0, 2].set(title="Absolute Error (normalized)", xlabel="x [m]", ylabel="y [m]")

    cy = N // 2
    x_slice = np.linspace(-Lx/2, Lx/2, N)

    ax[1, 0].plot(x_slice * 1e3, I_sim_norm[cy, :], color="red", ls="--", label="Simulated", zorder=10)
    ax[1, 0].plot(x_slice * 1e3, I_th_norm[cy, :], color="black", ls="-", label="Theoretical")
    ax[1, 0].set(title="Central Slice", xlabel="x [mm]", ylabel="Normalized Intensity")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogy(x_slice * 1e3, I_sim_norm[cy, :], color="red", ls="--", label="Simulated", zorder=10)
    ax[1, 1].semilogy(x_slice * 1e3, I_th_norm[cy, :], color="black", ls="-", label="Theoretical")
    ax[1, 1].set(title="Central Slice - Log", xlabel="x [mm]", ylabel="Normalized Intensity")
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(x_slice * 1e3, err[cy, :], color="red", linewidth=1.0)
    ax[1, 2].set(title="Central Slice Error", xlabel="x [mm]", ylabel="|Sim - Theory|")

    fig.colorbar(im0, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "2D_Gaussian_Beam_Validation"))


def test_constant_1D(propagation_func):
    print("Testing 1D Constant Beam...")
    Lx, Lz = 6e-3, 1e5
    N = 512
    z = 2.5

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), squeeze=False, sharey=True)
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

def test_zoom_1D():
    print("Testing 1D Scaled ASM (zoom) on Gaussian Beam...")
    Lx, Lz = 6e-3, 1e5
    N = 3000
    z = 3.
    w0 = 0.5e-2
    Rx = 0.02

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)

    x_in = wave.grid
    x_out = Rx * x_in

    wave.propagate(z, ScaledAngularSpectrum(simulation, Rx=Rx))
    I_sim = wave.intensity()

    U_th = F.gaussian_beam_1D(x_out, z=z, wavelength=wave.wavelength, w0=w0)
    I_th = np.abs(U_th)**2

    I_sim_norm = I_sim / I_sim.max()
    print("Simulated Peak Location:", wave.grid[np.argmax(I_sim_norm)])
    I_th_norm = I_th / I_th.max()
    print("Theoretical Peak Location:", x_out[np.argmax(I_th_norm)])
    err = np.abs(I_sim_norm - I_th_norm)

    print(f"  Rx = {Rx}")
    print(f"  Max Absolute Error (normalized): {err.max():.6e}")
    print(f"  Mean Absolute Error (normalized): {err.mean():.6e}")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.35)

    ax[0].plot(x_out * 1e3, I_sim_norm, color="red", ls="--", label="Scaled ASM", zorder=10)
    ax[0].plot(x_out * 1e3, I_th_norm, color="black", ls="-", label="Theoretical")
    ax[0].set(title=f"Gaussian (z={z} m, Rx={Rx})",
              xlabel="x [mm]", ylabel="Norm. Intensity")
    ax[0].tick_params(axis='y', labelsize=7)
    ax[0].legend(fontsize=8)

    ax[1].semilogy(x_out * 1e3, I_sim_norm, color="red", ls="--", label="Scaled ASM", zorder=10)
    ax[1].semilogy(x_out * 1e3, I_th_norm, color="black", ls="-", label="Theoretical")
    ax[1].set(title=f"Log Scale (z={z} m, Rx={Rx})",
              xlabel="x [mm]", ylabel="Norm. Intensity")
    ax[1].tick_params(axis='y', which='both', labelsize=7)
    ax[1].legend(fontsize=8)

    ax[2].plot(x_out * 1e3, err, color="red", linewidth=1.0)
    ax[2].set(title="Absolute Error (normalized)",
              xlabel="x [mm]", ylabel="|Sim - Theory|")
    ax[2].tick_params(axis='y', labelsize=7)

    plt.savefig(os.path.join(savedir, "1D_Scaled_ASM_Gaussian"))
    plt.show()

def test_zoom_2D():
    print("Testing 2D Scaled ASM (zoom) on Gaussian Beam...")
    Lx, Ly, Lz = 6e-3, 6e-3, 100
    N = 3000
    z = 3.
    w0 = 0.5e-3
    Rx, Ry = 0.02, 0.02

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=w0)

    x_in, y_in = wave.grid
    x_out, y_out = Rx * x_in, Ry * y_in

    wave.propagate(z, ScaledAngularSpectrum(simulation, Rx=Rx, Ry=Ry))
    I_sim = wave.intensity()

    U_th = F.gaussian_beam_2D(x_out, y_out, z=z, wavelength=wave.wavelength, w0=w0)
    I_th = np.abs(U_th)**2

    I_sim_norm = I_sim / I_sim.max()
    I_th_norm = I_th / I_th.max()
    err = np.abs(I_sim_norm - I_th_norm)

    print(f"  Rx = {Rx}, Ry = {Ry}")
    print(f"  Max Absolute Error (normalized): {err.max():.6e}")
    print(f"  Mean Absolute Error (normalized): {err.mean():.6e}")

    Lx_out, Ly_out = Rx * Lx, Ry * Ly
    extent_out = [-Lx_out/2, Lx_out/2, -Ly_out/2, Ly_out/2]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 6), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    im0 = ax[0, 0].imshow(I_sim_norm, cmap="Greys_r", extent=extent_out)
    ax[0, 0].set(title=f"Scaled ASM (z={z} m, Rx={Rx})",
                 xlabel="x [m]", ylabel="y [m]")

    ax[0, 1].imshow(I_th_norm, cmap="Greys_r", extent=extent_out)
    ax[0, 1].set(title=f"Theoretical (z={z} m)",
                 xlabel="x [m]", ylabel="y [m]")

    ax[0, 2].imshow(err, cmap="Greys_r", extent=extent_out)
    ax[0, 2].set(title="Absolute Error (normalized)",
                 xlabel="x [m]", ylabel="y [m]")

    cy = N // 2
    x_slice = np.linspace(-Lx_out/2, Lx_out/2, N)

    ax[1, 0].plot(x_slice * 1e3, I_sim_norm[cy, :], color="red", ls="--", label="Scaled ASM", zorder=10)
    ax[1, 0].plot(x_slice * 1e3, I_th_norm[cy, :], color="black", ls="-", label="Theoretical")
    ax[1, 0].set(title="Central Slice",
                 xlabel="x [mm]", ylabel="Normalized Intensity")
    ax[1, 0].tick_params(axis='y', labelsize=7)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogy(x_slice * 1e3, I_sim_norm[cy, :], color="red", ls="--", label="Scaled ASM", zorder=10)
    ax[1, 1].semilogy(x_slice * 1e3, I_th_norm[cy, :], color="black", ls="-", label="Theoretical")
    ax[1, 1].set(title="Central Slice - Log",
                 xlabel="x [mm]", ylabel="Norm. Intensity")
    ax[1, 1].tick_params(axis='y', which='both', labelsize=7)
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(x_slice * 1e3, err[cy, :], color="red", linewidth=1.0)
    ax[1, 2].set(title="Central Slice Error",
                 xlabel="x [mm]", ylabel="|Sim - Theory|")
    ax[1, 2].tick_params(axis='y', labelsize=7)

    fig.colorbar(im0, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "2D_Scaled_ASM_Gaussian"))
    plt.show()

if __name__ == "__main__":
    test_gaussian_1D(AngularSpectrum)
    test_gaussian_2D(AngularSpectrum)
    test_constant_1D(AngularSpectrum)
    test_zoom_1D()
    test_zoom_2D()
