import numpy as np
import xraylib as xrl
import matplotlib.pyplot as plt
from pathlib import Path

from ..classes import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/error_test").resolve()
savedir.mkdir(parents=True, exist_ok=True)

def test_kinoform_etch(err):
    print("Testing Kinoform with systematic etch error (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    lens.add_error(LensErrors.periodic_etch, err=err, interval=N//1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Etched Kinoform profile (err={err:.1e} m)")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform__etch_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved etched kinoform profile to {out}.")


def test_kinoform_random_etch(max_err):
    print("Testing Kinoform with random etch error (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    lens.add_error(LensErrors.random_etch, max_err=max_err, interval=N//1000, distribution_func="gaussian", seed=67)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Etched Kinoform profile (max_err={max_err:.1e} m)")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform_random_etch_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved etched kinoform profile to {out}.")
    
def test_zone_removal(proportion):
    print("Testing Kinoform with zone removal (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    print("Zones:", lens.zones)
    n = lens.zones
    lens.add_error(LensErrors.zone_removal, m=-n, proportion=proportion, direction="in", extend=True, remove_last=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Tapered Kinoform profile (proportion={proportion})")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform_taper_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved etched kinoform profile to {out}.")
    
def test_gaussian_etch(max_err, invert=False):
    print("Testing Kinoform with Gaussian-distributed etch error (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    lens.add_error(LensErrors.gaussian_etch, max_err=max_err, invert=invert, seed=67)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Gaussian-etched Kinoform profile (max_err={max_err:.1e} m, invert={invert})")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"Kinoform_gaussian_etch_profile{'_inverted' if invert else ''}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved etched kinoform profile to {out}.")
    
def test_zone_placement(max_err):
    print("Testing Kinoform with zone placement error (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    err = lens.add_error(LensErrors.zone_placement, err=max_err, gap=True, seed=67)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Zone placement Kinoform profile (max_err={max_err:.1e} m")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"Kinoform_zone_placement_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")

def test_quantization():
    pass
    
if __name__ == "__main__":
    # max_err = 5e-8
    # test_kinoform_etch(max_err)
    # test_kinoform_random_etch(max_err)
    # test_gaussian_etch(max_err)
    # test_gaussian_etch(max_err, invert=True)
    # test_zone_removal(1e-6)
    test_zone_placement(1e-6)

    
    
