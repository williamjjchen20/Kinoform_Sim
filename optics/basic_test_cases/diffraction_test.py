import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, functools

from angular_spectrum import angular_spectrum_method
from classes import SimulationObject, WaveFunctions, ApertureFunctions, Waveform, Aperture

F = WaveFunctions()
A = ApertureFunctions()

def slit_aperture_1D(x, z=0, width=0.5e-3):
    field = np.zeros_like(x)
    mask = np.abs(x) <= width/2
    field[mask] = 1.0
    return field
    
def slit_aperture_2D(X, Y, z=0, width=0.5e-3, height=0.5e-3):
    field = np.zeros_like(X)
    mask = np.abs(X) <= width/2
    mask &= np.abs(Y) <= height/2 
    field[mask] = 1.0
    return field

def test_slit_aperture_1D(width):
    print("Testing 1D Single Slit Diffraction...")
    Lx, Lz = 5*width, 100
    N=512
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    wave_distribution = F.const_wave_1D
    wave = Waveform(energy=1.96, simulation=simulation, z=0, distribution_func=wave_distribution)
    transmission_func = functools.partial(slit_aperture_1D, width=0.5e-3)
    aperture = Aperture(simulation=simulation, z=1.0, transmission_func=transmission_func)

    plt.plot(wave.grid, aperture.field)
    plt.savefig(os.path.join("test_figs/", "Slit_Aperture_1D"))
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), squeeze=False, sharey=True)
    ax[0, 0].plot(wave.grid, wave.intensity(), color="Black")
    ax[0, 0].set(title="Initial Beam (z=0 m)")
    
    z = 0.5
    aperture.diffract(wave)
    wave.propagate(z, functools.partial(angular_spectrum_method, dim=1))
    
    I = wave.intensity()
    norm = I.max()
    ax[0, 1].plot(wave.grid, I, color="Red", ls="--", label="Simulated", zorder=10)
    
    x = np.linspace(-Lx/2, Lx/2, N)
    U = A.single_slit_1D(x, z=z, wavelength=wave.wavelength, width=0.5e-3)
    I_th = np.abs(U)**2
    I_th = I/I.max()*norm
    ax[0, 1].plot(x, I_th, ls="-", color="Black", label="Theoretical", zorder=1)
    ax[0, 1].set(title=f"Diffraction Pattern (z={z} m)")
    ax[0, 1].legend()

    err = np.abs(wave.intensity()-I)
    ax[0, 2].plot(err/I_th)
    ax[0, 2].set(title="Relative Error")
    plt.savefig(os.path.join("test_figs/", "1D_Constant_Diffraction"))
    
def test_slit_aperture_2D(width, height):
    print("Testing 2D Single Slit Diffraction...")
    Lx, Ly, Lz = 2*width, 2*height, 100
    N=1024
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    wave_distribution = functools.partial(F.const_wave_2D) 
    wave = Waveform(energy=1.96, simulation=simulation, z=0, distribution_func=wave_distribution)
    transmission_func = functools.partial(slit_aperture_2D, width=0.7e-3, height=0.35e-3)
    aperture = Aperture(simulation=simulation, z=0.5, transmission_func=transmission_func)

    plt.figure()
    plt.gca().set_facecolor("black")
    plt.imshow(aperture.field, cmap="gray", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2]) #type: ignore
    plt.gca().set(xlim=(-width, width), ylim=(-height, height))
    plt.colorbar()
    plt.savefig(os.path.join("test_figs/", "Slit_Aperture_2D"))

    plt.figure()    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), squeeze=False)
    plt.subplots_adjust(wspace=0.3)
    
    norm = colors.LogNorm(vmin=1e-6, vmax=wave.intensity().max())
    im=ax[0, 0].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlim=(-width, width), ylim=(-height, height))
    
    z=0.5
    aperture.diffract(wave)
    wave.propagate(z, functools.partial(angular_spectrum_method, dim=2))
    
    norm = colors.LogNorm(vmin=5e-4, vmax=wave.intensity().max())
    ax[0, 1].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Simulated Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))
    
    x, y = np.linspace(-Lx/2, Lx/2, N), np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)
    U = A.single_slit_2D(X, Y, z=z, wavelength=wave.wavelength, width=0.7e-3, height=0.35e-3)
    I_th = np.abs(U)**2
    # norm = colors.LogNorm(vmin=5e-4, vmax=I_th.max())
    ax[0, 2].imshow(I_th, norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 2].set(title=f"Theoretical Diffraction (z={z} m)", xlim=(-width, width), ylim=(-height, height))
    
    I_th[I_th == 0] = 1e-10 
    err = np.abs(wave.intensity()-I_th)
    ax[0, 3].imshow(err/I_th, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 3].set(title=f"Relative Error", xlim=(-width, width), ylim=(-height, height))
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join("test_figs/", "2D_Constant_Diffraction"))

def test_gaussian(width, height):
    print("Testing 2D Gaussian Beam Diffraction")
    Lx, Ly, Lz = 10*width, 10*height, 100
    N=1024
    
    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    
    wave_distribution = functools.partial(F.gaussian_beam_2D, w0=0.5e-3) 
    wave = Waveform(energy=1.96, simulation=simulation, z=0, distribution_func=wave_distribution)
    
    transmission_func = functools.partial(slit_aperture_2D, width=0.7e-3, height=0.35e-3)
    aperture = Aperture(simulation=simulation, z=0.5, transmission_func=transmission_func)

    norm = colors.LogNorm(vmin=1e-6, vmax=wave.intensity().max())

    plt.figure()    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), squeeze=False)
    im=ax[0, 0].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 0].set(title="Initial Beam (z=0 m)", xlim=(-width, width), ylim=(-height, height))
    
    z1, z2 = 0.5, 0.5
    wave.propagate(z1, functools.partial(angular_spectrum_method, dim=2))
    aperture.diffract(wave)
    wave.propagate(z2, functools.partial(angular_spectrum_method, dim=2))
    
    norm = colors.LogNorm(vmin=5e-4, vmax=wave.intensity().max())
    ax[0, 1].imshow(wave.intensity(), norm=norm, cmap="Greys_r", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    ax[0, 1].set(title=f"Diffraction Pattern (z={z1+z2} m)", xlim=(-width, width), ylim=(-height, height))
    
    
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
    plt.savefig(os.path.join("test_figs/", "2D_Gaussian_Diffraction"))

if __name__ == "__main__":
    width, height = 0.5e-2, 0.5e-2
    test_slit_aperture_1D(width)
    test_slit_aperture_2D(width, height)
    test_gaussian(width, height)