import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, functools
from pathlib import Path

from ..propagators import *
from ..classes import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/diffraction_test").resolve()

A = DiffractionPatterns()

def test_slit_aperture_1D(width):
    print("Testing 1D Single Slit Diffraction...")
    Lx, Lz = 5 * width, 100
    N = 5000
    slit_width = 0.5e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave = ConstantBeam(energy=1.96, simulation=simulation, z=0)
    aperture = SingleSlit(simulation=simulation, z=1.0, width=slit_width)
    propagator = AngularSpectrum(simulation)

    # Plot aperture transmission
    plt.figure()
    plt.plot(aperture.grid, aperture.field)  # type: ignore
    plt.title("Slit Aperture")
    plt.xlabel("x [m]")
    plt.ylabel("Transmission")
    plt.savefig(os.path.join(savedir, "Slit_Aperture_1D"))

    # Propagate
    z = 1.5
    aperture.transform(wave)
    wave.propagate(z, propagator)

    # Simulated intensity
    I_sim = wave.intensity()

    # Theoretical intensity
    x = np.linspace(-Lx / 2, Lx / 2, N)
    U_th = A.single_slit_1D(x, z=z, wavelength=wave.wavelength, width=slit_width)
    I_th = np.abs(U_th) ** 2

    # Normalize both to their peaks
    I_sim_norm = I_sim / I_sim.max()
    I_th_norm = I_th / I_th.max()

    # Error
    I_th_safe = np.where(I_th_norm > 1e-10, I_th_norm, 1e-10)
    err_abs = np.abs(I_sim_norm - I_th_norm)
    err_rel = err_abs / I_th_safe

    print(f"  Max Absolute Error (normalized): {err_abs.max():.6e}")
    print(f"  Max Relative Error:              {err_rel.max():.6e}")

    # --- Plotting ---
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.35)

    # Panel 1: Linear scale overlay
    ax[0].plot(x * 1e3, I_sim_norm, color="red", ls="--", label="Simulated", zorder=10)
    ax[0].plot(x * 1e3, I_th_norm, color="black", ls="-", label="Theoretical", zorder=1)
    ax[0].set(title=f"Diffraction Pattern (z={z} m)",
              xlabel="x [mm]", ylabel="Normalized Intensity",
              xlim=(-width * 1e3, width * 1e3))
    ax[0].legend(fontsize=8)

    # Panel 2: Log scale overlay
    ax[1].semilogy(x * 1e3, I_sim_norm, color="red", ls="--", label="Simulated", zorder=10)
    ax[1].semilogy(x * 1e3, I_th_norm, color="black", ls="-", label="Theoretical", zorder=1)
    ax[1].set(title=f"Diffraction Pattern - Log (z={z} m)",
              xlabel="x [mm]", ylabel="Normalized Intensity",
              xlim=(-width * 1e3, width * 1e3))
    ax[1].legend(fontsize=8)

    # Panel 3: Relative error
    ax[2].plot(x * 1e3, err_abs, color="red", linewidth=1.0)
    ax[2].set(title="Absolute Error",
              xlabel="x [mm]", ylabel="|Sim - Theory|",
              xlim=(-width * 1e3, width * 1e3))

    plt.savefig(os.path.join(savedir, "1D_SingleSlit_Diffraction"))
    plt.show()



def test_slit_aperture_2D(width, height):
    print("Testing 2D Single Slit Diffraction...")
    Lx, Ly, Lz = 2*width, 2*height, 100
    N = 1024
    slit_width, slit_height = 0.5e-3, 0.25e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = ConstantBeam(energy=1.96, simulation=simulation, z=0)
    aperture = SingleSlit(simulation=simulation, z=0.5, width=slit_width, height=slit_height)
    propagator = AngularSpectrum(simulation)

    plt.figure()
    plt.gca().set_facecolor("black")
    plt.imshow(aperture.field, cmap="gray", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2]) #type: ignore
    plt.gca().set(xlim=(-width, width), ylim=(-height, height))
    plt.colorbar()
    plt.savefig(os.path.join(savedir, "Slit_Aperture_2D"))

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), squeeze=False)
    plt.subplots_adjust(wspace=0.3)

    norm = colors.LogNorm(vmin=1e-6, vmax=wave.intensity().max())
    im = ax[0, 0].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlim=(-width, width), ylim=(-height, height))

    z = 0.7
    aperture.transform(wave)
    wave.propagate(z, propagator)

    norm = colors.LogNorm(vmin=5e-4, vmax=wave.intensity().max())
    ax[0, 1].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Simulated Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))

    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = A.single_slit_2D(X, Y, z=z, wavelength=wave.wavelength, width=slit_width, height=slit_height)
    I_th = np.abs(U)**2
    ax[0, 2].imshow(I_th, norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 2].set(title=f"Theoretical Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))

    I_th[I_th == 0] = 1e-10
    err = np.abs(wave.intensity()-I_th)
    ax[0, 3].imshow(err/I_th, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 3].set(title="Relative Error", xlim=(-width, width), ylim=(-height, height))
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "2D_Constant_Diffraction"))

def test_circular_aperture_1D(width, height):
    print("Testing 2D Circular Aperture Diffraction (1D Central Slice)...")
    width, height = 1.5e-3, 1.5e-3
    Lx, Ly, Lz = width, height, 100
    N = 5000
    E = 8e3
    slit_radius = 5e-6#0.25e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = ConstantBeam(energy=E, simulation=simulation, z=0)
    # aperture = CircularAperture(simulation=simulation, z=0.0, radius=slit_radius)
    aperture = XrayParabolicLens(f=1.,R=slit_radius,n=1.5,simulation=simulation,z=0.)
    propagator = AngularSpectrum(simulation)

    z = 1.0
    aperture.transform(wave)
    wave.propagate(z, propagator)

    # Simulated intensity
    I_sim = wave.intensity()

    # Theoretical intensity
    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = A.circular(X, Y, z=z, wavelength=wave.wavelength, radius=slit_radius)
    I_th = np.abs(U)**2

    # Central slices (horizontal, through middle row)
    mid = N // 2
    I_sim_slice = I_sim[mid, :]
    I_th_slice = I_th[mid, :]

    # Normalize both to their peak for shape comparison
    I_sim_norm = I_sim_slice / I_sim_slice.max()
    I_th_norm = I_th_slice / I_th_slice.max()

    # --- Plotting ---
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    plt.subplots_adjust(wspace=0.35)

    # Panel 1: Linear scale overlay
    ax[0].plot(x * 1e3, I_sim_norm, label="Simulated", linewidth=1.5)
    ax[0].plot(x * 1e3, I_th_norm, '--', label="Theoretical", linewidth=1.5)
    ax[0].axhline(0.5, color='gray', linestyle=':', linewidth=0.8, label="Half Max")
    ax[0].set(title=f"Central Slice (z={z} m)",
              xlabel="x [mm]", ylabel="Normalized Intensity",
              xlim=(-width * 1e3, width * 1e3))
    ax[0].legend(fontsize=8)

    # Panel 2: Log scale overlay
    ax[1].semilogy(x * 1e3, I_sim_norm, label="Simulated", linewidth=1.5)
    ax[1].semilogy(x * 1e3, I_th_norm, '--', label="Theoretical", linewidth=1.5)
    ax[1].set(title=f"Central Slice - Log (z={z} m)",
              xlabel="x [mm]", ylabel="Intensity",
              xlim=(-width * 1e3, width * 1e3))
    ax[1].legend(fontsize=8)

    # Panel 3: Residual error
    err_slice = np.abs(I_sim_slice - I_th_slice)
    ax[2].plot(x * 1e3, err_slice, color='red', linewidth=1.0)
    ax[2].set(title="Absolute Error (Central Slice)",
              xlabel="x [mm]", ylabel="|Sim - Theory|",
              xlim=(-width * 1e3, width * 1e3))

    plt.savefig(os.path.join(savedir, "Circular_Aperture_Central_Slice"))
    plt.show()

    
def test_circular_aperture_2D(width, height):
    print("Testing 2D Circular Aperture Diffraction...")
    Lx, Ly, Lz = 3*width, 3*height, 100
    N = 1024
    slit_radius = 0.25e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = ConstantBeam(energy=1.96, simulation=simulation, z=0)
    aperture = CircularAperture(simulation=simulation, z=0.5, radius=slit_radius)
    propagator = AngularSpectrum(simulation)

    plt.figure()
    plt.gca().set_facecolor("black")
    plt.imshow(aperture.field, cmap="gray", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2]) #type: ignore
    plt.gca().set(xlim=(-width, width), ylim=(-height, height))
    plt.colorbar()
    plt.savefig(os.path.join(savedir, "Circular_Aperture"))

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), squeeze=False)
    plt.subplots_adjust(wspace=0.3)

    norm = colors.LogNorm(vmin=1e-6, vmax=wave.intensity().max())
    im = ax[0, 0].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlim=(-width, width), ylim=(-height, height))

    z = 1.0
    aperture.transform(wave)
    wave.propagate(z, propagator)

    norm = colors.LogNorm(vmin=5e-4, vmax=wave.intensity().max())
    ax[0, 1].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Simulated Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))

    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = A.circular(X, Y, z=z, wavelength=wave.wavelength, radius=slit_radius)
    I_th = np.abs(U)**2
    ax[0, 2].imshow(I_th, norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 2].set(title=f"Theoretical Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))

    I_th[I_th == 0] = 1e-10
    err = np.abs(wave.intensity()-I_th)
    print("Maximum Error:", np.max(err))
    idx = np.unravel_index(np.argmax(err), err.shape)
    print("Maximum Relative Error:", np.max(err)/I_th[idx])
    ax[0, 3].imshow(err, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 3].set(title="Error", xlim=(-width, width), ylim=(-height, height))
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "Circular_Aperture_Diffraction"))

def test_gaussian(width, height):
    print("Testing 2D Gaussian Beam Diffraction")
    Lx, Ly, Lz = 10*width, 10*height, 100
    N = 1024
    slit_width, slit_height = 0.7e-3, 0.35e-3

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=0.5e-3)
    aperture = SingleSlit(simulation=simulation, z=0.5, width=slit_width, height=slit_height)
    propagator = AngularSpectrum(simulation)

    norm = colors.LogNorm(vmin=1e-6, vmax=wave.intensity().max())

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), squeeze=False)
    im = ax[0, 0].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlim=(-width, width), ylim=(-height, height))

    z1, z2 = 0.5, 0.5
    wave.propagate(z1, propagator)
    aperture.transform(wave)
    wave.propagate(z2, propagator)

    norm = colors.LogNorm(vmin=5e-4, vmax=wave.intensity().max())
    ax[0, 1].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Diffraction Pattern (z={z1+z2} m)", xlim=(-width, width), ylim=(-height, height))

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join(savedir, "2D_Gaussian_Diffraction"))


if __name__ == "__main__":
    width, height = 0.5e-2, 0.5e-2
    # test_slit_aperture_1D(width)
    # test_slit_aperture_2D(width, height)
    test_circular_aperture_1D(width, height)
    # test_gaussian(width, height)
