import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import xraylib as xrl
import os, sys, functools
from pathlib import Path

from ..propagators import *
from ..classes import *
from ..metrics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/lens_test").resolve()

def test_standard_lens_1D():
    print("Testing Standard Lens (1D)...")
    Lx, Lz = 0.025, 10000
    N = 4096

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
    propagator = functools.partial(angular_spectrum_method, dim=1)

    source = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=Lx/2)
    lens = OpticalLens(R=0.01, n=1.05, t0=0., R1=0.5, R2=-0.5, simulation=simulation, z=0.1)
    lens.init_transmittance(source)

    # Voelz and Roggemann (2009) sampling criterion
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R)
    print("Nyquist sampling rate, dx, dx<f_s:", f_s, Lx/N, (Lx/N) < f_s)
    assert (Lx/N) < f_s

    x = source.grid
    z1, z2 = lens.center[-1], lens.f

    # reference analytic Gaussian beam at each station (no lens applied)
    ref_initial   = GaussianBeam(energy=1.96, simulation=simulation, z=0,     w0=Lx/2)
    ref_pre_lens  = GaussianBeam(energy=1.96, simulation=simulation, z=z1,    w0=Lx/2)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8), squeeze=False, sharex=True)

    def _plot(col, title, wave, ref=None):
        I, ph = wave.intensity(), wave.phase()
        ax[0, col].plot(x, I, color="navy", label="sim")
        ax[1, col].plot(x, ph, color="navy", label="sim")
        if ref is not None:
            ax[0, col].plot(x, ref.intensity(), "--", color="crimson", label="analytic Gaussian")
            ax[1, col].plot(x, ref.phase(),     "--", color="crimson", label="analytic Gaussian")
        ax[0, col].set(title=title, ylabel="Intensity", yscale="log")
        ax[1, col].set(xlabel="x [m]", ylabel="Phase [rad]")
        ax[0, col].legend(fontsize=8); ax[1, col].legend(fontsize=8)

    print("Initial peak I, phase:", np.max(source.intensity()), source.phase()[N//2])
    _plot(0, "Initial source (z=0)", source, ref=ref_initial)

    source.propagate(z1, propagator)
    print("Pre-lens peak I, phase:", np.max(source.intensity()), source.phase()[N//2])
    _plot(1, f"Pre-lens (z={z1:g} m)", source, ref=ref_pre_lens)

    lens.transform(source)
    source.propagate(z2, propagator)
    print("Focal peak I, phase:", np.max(source.intensity()), source.phase()[N//2])
    _plot(2, f"Focal plane (z={z1+z2:g} m)", source, ref=None)

    fig.suptitle("1D Standard Lens: simulation vs. analytic Gaussian")
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, "1D_Thin_Lens"))


def test_standard_lens_2D():
    print("Testing Standard Lens (2D)...")
    Lx, Ly, Lz = 0.025, 0.025, 10000
    N = 5000

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)

    source = GaussianBeam(energy=1.96, simulation=simulation, z=0, w0=Lx/4)
    lens = OpticalLens(R=0.01, n=1.05, t0=0., R1=0.5, R2=-0.5, simulation=simulation, z=0.1)
    lens.init_transmittance(source)

    # Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R)
    print("Nyquist sampling rate, dx, dx<f_s:", f_s, Lx/N, (Lx/N) < f_s)
    assert (Lx/N) < f_s and (Ly/N) < f_s

    z1, z2 = lens.center[-1], lens.f
    print("z1 (source -> lens):", z1, "  z2 (lens -> focus):", z2)

    # propagation stations: 2 before the lens, then lens, then 3 after
    pre_steps  = [("z=0 (source)",           0.0),
                  (f"z={z1/2:g} m",          z1/2),
                  (f"z={z1:g} m (pre-lens)", z1/2)]
    post_steps = [(f"z={z1+z2/3:g} m",       z2/3),
                  (f"z={z1+2*z2/3:g} m",     z2/3),
                  (f"z={z1+z2:g} m (focus)", z2/3)]
    n_cols = len(pre_steps) + len(post_steps)

    fig, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(3.5*n_cols, 7),
                           squeeze=False, sharey=True)

    def _plot(col, title, wave):
        wave.view(ax=ax[0, col], show_cbar=(col == n_cols-1))
        ph = wave.phase()
        im = ax[1, col].imshow(ph, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                               origin="lower", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi)
        ax[0, col].set(title=title)
        ax[1, col].set(xlabel="x [m]", ylabel="y [m]" if col == 0 else "")
        if col == n_cols-1:
            fig.colorbar(im, ax=ax[1, col], fraction=0.046, pad=0.04, label="Phase [rad]")
        print(f"  {title}: I_max={np.max(wave.intensity()):.3e}  "
              f"phi[c]={ph[N//2, N//2]:+.3f}")

    col = 0
    for title, dz in pre_steps:
        if dz > 0: source.propagate(dz, propagator)
        _plot(col, title, source); col += 1

    lens.transform(source)

    for title, dz in post_steps:
        source.propagate(dz, propagator)
        _plot(col, title, source); col += 1

    fig.suptitle(f"2D Standard Lens (f={lens.f} m, R={lens.R*1000:g} mm): "
                 "intensity (top) and phase (bottom) along z")
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, "2D_Thin_Lens"))
    
    
def test_lens_xray():
    print("Testing Parabolic Lens (X-ray)...")
    Lx, Ly, Lz = 1.5e-4, 1.5e-4, 10000
    N = 5000

    # parameters for simulation
    E = 8.5e3       # eV
    f = 1.          # m
    R = 5e-5
    n = xrl.Refractive_Index("Si", E/1000, 2.329)
    print("Refractive Index:", n)
    assert (R < Lx and R < Ly)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = XrayParabolicLens(f=f, R=R, n=n, simulation=simulation, z=0)
    lens.plot_profile(ax=plt.figure().gca(), savedir=str(savedir))
    lens.init_transmittance(source)

    # Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R)
    print("Nyquist sampling rate, dx, dx<f_s:", f_s, Lx/N, (Lx/N) < f_s)
    assert (Lx/N) < f_s and (Ly/N) < f_s

    z1, z2 = lens.center[-1], lens.f
    print("z1 (source -> lens):", z1, "  z2 (lens -> focus):", z2)

    # lens sits at z=0, so all stations are post-lens steps toward focus
    post_steps = [(f"z={z2/4:g} m",          z2/4),
                  (f"z={z2/2:g} m",          z2/4),
                  (f"z={3*z2/4:g} m",        z2/4),
                  (f"z={z2:g} m (focus)",    z2/4)]
    n_cols = 1 + len(post_steps)   # +1 for the pre-lens / incident view

    fig, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(3.5*n_cols, 7),
                           squeeze=False, sharey=True)

    def _plot(col, title, wave):
        wave.view(ax=ax[0, col], show_cbar=(col == n_cols-1))
        ph = wave.phase()
        im = ax[1, col].imshow(ph, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                               origin="lower", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi)
        ax[0, col].set(title=title)
        ax[1, col].set(xlabel="x [m]", ylabel="y [m]" if col == 0 else "")
        if col == n_cols-1:
            fig.colorbar(im, ax=ax[1, col], fraction=0.046, pad=0.04, label="Phase [rad]")
        print(f"  {title}: I_max={np.max(wave.intensity()):.3e}  "
              f"phi[c]={ph[N//2, N//2]:+.3f}")

    col = 0
    _plot(col, "z=0 (incident)", source); col += 1

    lens.transform(source)

    for title, dz in post_steps:
        source.propagate(dz, propagator)
        _plot(col, title, source); col += 1

    fig.suptitle(rf"X-ray Parabolic Lens (R={lens.R*1e6:g} $\mu$m, f={lens.f} m): "
                 "intensity (top) and phase (bottom) along z")
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, "Xray_Parabolic_Lens"))


def test_kinoform():
    print("Testing Kinoform (X-ray)...")

    Lx, Ly, Lz = 1.5e-4, 1.5e-4, 10000
    N = 1024

    # parameters for simulation
    E = 8.5e3       # eV
    f = 1.          # m
    R = 5e-5
    n = xrl.Refractive_Index("Si", E/1000, 2.329)
    print("Refractive Index:", n)
    assert (R < Lx and R < Ly)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
    propagator = functools.partial(angular_spectrum_method, dim=2)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n, simulation=simulation, z=0)
    lens.plot_profile(ax=plt.figure().gca(), savedir=str(savedir))
    lens.init_transmittance(source)

    # Voelz and Roggemann (2009) sampling criterion to avoid aliasing
    f_s = source.wavelength*np.abs(lens.f)/(2*lens.R)
    print("Nyquist sampling rate, dx, dx<f_s:", f_s, Lx/N, (Lx/N) < f_s)
    assert (Lx/N) < f_s and (Ly/N) < f_s

    z1, z2 = lens.center[-1], lens.f
    print("z1 (source -> lens):", z1, "  z2 (lens -> focus):", z2)

    # lens sits at z=0, so all stations are post-lens steps toward focus
    post_steps = [(f"z={z2/4:g} m",          z2/4),
                  (f"z={z2/2:g} m",          z2/4),
                  (f"z={3*z2/4:g} m",        z2/4),
                  (f"z={z2:g} m (focus)",    z2/4)]
    n_cols = 1 + len(post_steps)   # +1 for the pre-lens / incident view

    fig, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(3.5*n_cols, 7),
                           squeeze=False, sharey=True)

    def _plot(col, title, wave):
        wave.view(ax=ax[0, col], show_cbar=(col == n_cols-1))
        ph = wave.phase()
        im = ax[1, col].imshow(ph, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                               origin="lower", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi)
        ax[0, col].set(title=title)
        ax[1, col].set(xlabel="x [m]", ylabel="y [m]" if col == 0 else "")
        if col == n_cols-1:
            fig.colorbar(im, ax=ax[1, col], fraction=0.046, pad=0.04, label="Phase [rad]")
        print(f"  {title}: I_max={np.max(wave.intensity()):.3e}  "
              f"phi[c]={ph[N//2, N//2]:+.3f}")

    col = 0
    _plot(col, "z=0 (incident)", source); col += 1

    lens.transform(source)

    for title, dz in post_steps:
        source.propagate(dz, propagator)
        _plot(col, title, source); col += 1
        
    fig.suptitle(rf"Kinoform Lens (R={lens.R*1e6:g} $\mu$m, f={lens.f} m): "
                 "intensity (top) and phase (bottom) along z")
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, "Kinoform_Lens"))

if __name__ == "__main__":
    # test_standard_lens_2D()
    # test_standard_lens_1D()
    # test_lens_xray()
    test_kinoform()
    