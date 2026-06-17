import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xraylib as xrl
import os, functools
from pathlib import Path
import argparse

from ..propagators import *
from ..classes import *
from .metrics import *


script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./results").resolve()

def run_lens(label: str, lens_cls, simulation: SimulationObject, propagator, E: float, f: float, R: float, n: float | complex, w0=None):
    '''
    Initializes a source, applies a lens of class `lens_cls`, propagates to
    the focal plane, and returns (incident_wave, focal_wave, lens).
    
    Uses a ConstantBeam reference for fair power-in normalization.
    '''
    if w0 is None:
        source = ConstantBeam(energy=E, simulation=simulation, z=0)
    else:
        source = GaussianBeam(energy=E, simulation=simulation, z=0, w0=w0)

    lens = lens_cls(f=f, R=R, n=n, wavelength=source.wavelength, simulation=simulation, z=0)
    lens.init_transmittance(source)

    # Sampling check
    f_s = source.wavelength * np.abs(lens.f) / (2 * lens.R)
    if simulation.Lx / simulation.Nx >= f_s:
        print(f"[{label}] WARNING: under-sampled (dx={simulation.Lx/simulation.Nx:.3e} >= f_s={f_s:.3e})")

    # snapshot incident wave (after aperture mask but before phase) for power_in
    incident_power = total_power(source)

    # propagate to focal plane
    lens.transform(source)
    source.propagate(lens.f, propagator)

    return source, lens, incident_power


def collect_metrics(focal_wave, incident_power, label):
    I = focal_wave.intensity()
    fwhm = FWHM(focal_wave)
    I_max, I_avg = intensity_stats(focal_wave)
    P_focal = total_power(focal_wave)
    eff = P_focal / incident_power if incident_power > 0 else np.nan
    return {
        "label": label,
        "FWHM [m]": fwhm,
        "I_max": I_max,
        "I_avg": I_avg,
        "P_focal": P_focal,
        "P_incident": incident_power,
        "focusing_efficiency": eff,
    }


def print_comparison(metrics_list):
    keys = ["FWHM [m]", "I_max", "I_avg", "P_focal", "P_incident", "focusing_efficiency"]
    header = f"{'Metric':<25}" + "".join(f"{m['label']:>20}" for m in metrics_list)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<25}"
        for m in metrics_list:
            v = m[k]
            row += f"{v:>20.6e}" if isinstance(v, (int, float, np.floating)) else f"{str(v):>20}"
        print(row)
    print("=" * len(header))


def plot_comparison(lens_dict, savepath):
    '''
    Plot lens phase, focal-plane intensity (log), and central line cut for an
    arbitrary set of lenses.

    `lens_dict` maps label -> (focal_wave, lens). One row per lens, three
    columns: phase, focal intensity, central cut.
    '''
    n_lenses = len(lens_dict)
    if n_lenses == 0: raise Exception("No lenses added.")

    fig, ax = plt.subplots(nrows=n_lenses, ncols=3, figsize=(15, 4.5 * n_lenses), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    intensities = {label: wave.intensity() for label, (wave, _) in lens_dict.items()}
    vmax = max(I.max() for I in intensities.values())
    vmin = max(vmax * 1e-6, 1e-20)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    cmap_cycle = plt.get_cmap("tab10")

    for i, (label, (wave, lens)) in enumerate(lens_dict.items()):
        lens.plot_profile(ax=plt.figure().gca(), savedir=savedir, wavelength=wave.wavelength, label=label)
        Lx, Ly = wave.simulation.Lx, wave.simulation.Ly
        extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
        I = intensities[label]

        ax[i, 0].imshow(lens.angle(), cmap="twilight", extent=extent)
        ax[i, 0].set(title=f"{label} Lens Phase", xlabel="x [m]", ylabel="y [m]")

        im = ax[i, 1].imshow(I, norm=norm, cmap="inferno", extent=extent)
        ax[i, 1].set(title=f"{label} Focal Intensity", xlabel="x [m]", ylabel="y [m]")
        fig.colorbar(im, ax=ax[i, 1], fraction=0.046, pad=0.04)

        Nx = I.shape[1]
        x = np.linspace(-Lx/2, Lx/2, Nx)
        cy = I.shape[0] // 2
        ax[i, 2].plot(x, I[cy, :], color=cmap_cycle(i % 10))
        ax[i, 2].set(title=f"{label} Central Cut", xlabel="x [m]", ylabel="Intensity", yscale="log")

    fig.savefig(savepath)
    plt.close(fig)

def test_compare_xray_lenses(lens_dict):
    print("Comparing Lenses...")

    # Parameters
    N = 2048
    Lx = Ly = 1.5e-4
    Lz = 10000
    E = 8.5e3       # eV
    f = 1.0         # m
    R = 5e-5        # m
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    print(f"Refractive index n = {n}")
    print(f"Energy = {E} eV, f = {f} m, R = {R} m")

    propagator = functools.partial(angular_spectrum_method, dim=2)

    metrics = []
    results = {}
    for name in lens_dict:
        sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
        label = "".join([c for c in name if c.isalpha()])
        cls, _ = lens_dict[name]
        source, lens, P_in = run_lens(
            label, cls, sim, propagator, E, f, R, n
        )
        m = collect_metrics(source, P_in, name)
        metrics.append(m)
        results[name] = (source, lens)

    print_comparison(metrics)

    out = os.path.join(savedir, "Metrics_Lens_Comparison.png")
    plot_comparison(results, out)
    print(f"Saved comparison figure to {out}")

def take_user_input():
    done = False
    count = 1
    lens_dict = dict()
    while not done:
        print("Please input lens types to compare (Parabolic, Kinoform). ")
        lens = input("Type of lens to add: ")
        key = lens+str(count)
        
        # Lens type
        match lens:
            case "Parabolic":
                lens_dict[key] = [XrayParabolicLens]
            case "Kinoform":
                lens_dict[key] = [Kinoform]
            case "":
                done = True
                continue
            case _:
                raise Exception("Unknown lens type")
            
        # Quantization
        print("="*50)
        print("Please specify if lens should be quantized (skip if ideal)")
        N = input("N-level Approximation: ") 
        lens_dict[key].append(N)
        
        # Complete
        print("="*50)
        count+=1
        print(f"Added {lens} Lens!")
        
    return lens_dict
        
def main():
    lenses = take_user_input()
    test_compare_xray_lenses(lenses)

if __name__ == "__main__":
    main()