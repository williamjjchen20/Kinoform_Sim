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
    
def test_zone_removal(err):
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
    
    lens.add_error(LensErrors.zone_removal, direction="in", extend=True, remove_last=True, mutable=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Tapered Kinoform profile (proportion={err})")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform_removal_profile.png"
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
    R = 3e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)
    
    print(lens.zone_left, lens.zone_right, sep="\n")

    lens.add_error(LensErrors.zone_placement, err=[1e-6, 1e-6, 1e-6], mutable=True)
    
    print(lens.zone_left, lens.zone_right, sep="\n")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Zone placement Kinoform profile (max_err={max_err:.1e} m)")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"Kinoform_zone_placement_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")
    
def test_taper(max_err):
    print("Testing Kinoform with taper (1D)...")
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
    
    print(lens.zone_right)
    
    lens.add_error(LensErrors.kinoform_sidewall_taper, err=max_err, proportion=1.)
    # print(lens.R)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Tapered Kinoform profile (max_err={max_err:.1e} m)")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"Kinoform_taper_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")
    
def test_zone_quantization():
    print("Testing Kinoform with zone quantization (1D)...")
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

    # target the outermost few zones; anchor a 3-step staircase across each
    h = lens.height
    ms = np.arange(max(0, lens.zones - 4), lens.zones)
    zone_locations = np.asarray(lens.zone_locations)
    points = []
    for mi in ms:
        r_l, r_r = zone_locations[mi], zone_locations[mi + 1]
        points.append([(r_l, 0), (r_l + (r_r - r_l) / 4, h / 3), (r_l + 3 * (r_r - r_l) / 4, 2 * h / 3), (r_r, h)])
    points = np.array(points)

    lens.add_error(LensErrors.zone_quantization, points=points, m=ms)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Zone-quantized Kinoform profile (zones={list(ms)})")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform_zone_quantization_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")
    
def test_zone_warping():
    print("Testing Kinoform with zone quantization (1D)...")
    Lx, Lz = 5e-4, 10000
    N = 100000

    E = 8.e3
    f = 1.0
    R = 1e-4
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    beam_width = 1.5e-6
    
    print(lens.zone_widths[-1], beam_width/lens.zone_widths[-1])
    # lens.add_error(LensErrors.kinoform_sidewall_taper, err=1e-8, proportion=0.3)
    lens.add_error(LensErrors.kinoform_zone_warping, R_min=0, R_max=R, beam_width=beam_width)
    lens.add_error(LensErrors.cap_floor, h=0.02, proportion=True)


    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Zone-warped Kinoform profile (beam={beam_width} m)")
    ax.set_xlim(0, lens.R)

    out = savedir / "Kinoform_zone_warping_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")

def test_multierror():
    print("Testing Kinoform Multierror (1D)...")
    
    Lx, Lz = 6e-4, 10000
    N = 100000

    E = 8.e3
    f = 0.5
    R = 2e-4
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0, full=True)
    
    print(lens.R)
    
    beam_width = 3e-7

    print("Kinoform Height:", lens.height)
    print(lens.zone_widths[-1])
    lens.add_error(LensErrors.zone_placement, err=1e-7)
    lens.add_error(LensErrors.kinoform_zone_warping, R_min=0, R_max=lens.R, beam_width=beam_width)
    lens.add_error(LensErrors.kinoform_sidewall_taper, err=1e-7,proportion=1.)
    lens.add_error(LensErrors.cap_floor, h=0.01, proportion=True)
    lens.add_error(LensErrors.cap_height, h=0.98, proportion=True)
    print(lens.zone_locations[-1], lens.R)
    
    print("Outer Zone Width:", lens.zone_widths[-1])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask]*1e6, 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask]*1e6, lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [um]", ylabel="thickness [m]",
           title=f"Kinoform profile")
    ax.set_xlim(0.95*lens.R*1e6, 1.01*lens.R*1e6)

    out = savedir / f"Kinoform_error_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")
    
    
def test_reference():
    print("Testing Kinoform Reference (1D)...")
    '''
    Recreating AU kinoform lens manufacturing SEM snapshots in Gorelick et al. (2019)
    
    '''
    
    Lx, Lz = 5e-4, 10000
    N = 10000

    E = 6.e3
    f = 0.45
    R = 4.5e-5
    n = xrl.Refractive_Index("Au", E / 1000, 19.32)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0, zone_height=1.1e-6)
    # print(lens.zone_locations)
    lens.add_error(LensErrors.kinoform_sidewall_taper, err=1e-8,proportion=0.1)


    lens.add_error(LensErrors.cap_height, h=0.9, proportion=True)
    lens.add_error(LensErrors.cap_floor, h=0.05, proportion=True)
    lens.add_error(LensErrors.random_etch, max_err=5e-8, interval=1, distribution="gaussian")
    lens.add_error(LensErrors.gaussian_etch, max_err=1e-8, invert=True)
    print(lens.zone_widths[-1])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Kinoform profile")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"Kinoform_reference_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved kinoform profile to {out}.")
    
def test_FZP_error():
    print("Testing Kinoform Multierror (1D)...")
    
    Lx, Lz = 5e-4, 10000
    N = 10000

    E = 8.e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = FZP(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0, positive=True)
    print("FZP Height:", lens.height)

    lens.add_error(LensErrors.FZP_sidewall_taper, err=1e-6)

    print("Outer Zone Width:", lens.zone_widths[-1])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"FZP profile")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / f"FZP_error_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved FZP profile to {out}.")
    
if __name__ == "__main__":
    max_err = 5e-8
    # test_kinoform_etch(max_err)
    # test_kinoform_random_etch(max_err)
    # test_gaussian_etch(max_err)
    # test_gaussian_etch(max_err, invert=True)
    # test_zone_removal(1e-6)
    # test_zone_placement(1e-6)
    # test_taper(1e-6)
    # test_zone_quantization()
    # test_zone_warping()
    # test_multierror()
    # test_reference()
    test_FZP_error()

    
    
