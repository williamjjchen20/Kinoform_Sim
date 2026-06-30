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
savedir = (script_dir / "./test_figs").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, required=True)
parser.add_argument("--dim", type=int, choices=[1, 2], default=2,
                    help="simulation dimensionality (1 or 2)")
parser.add_argument("--extend", action="store_true",
                    help="render 2D focal/lens views as 3D surfaces")

def run_lens(label: str, lens_cls, simulation: SimulationObject, propagator: Propagator, 
             E: float, f: float, R: float, n: float | complex, 
             err_func=None, w0=None):
    '''
    Initializes a source, applies a lens of class `lens_cls`, propagates to
    the focal plane, and returns (incident_wave, focal_wave, lens).
    
    `err_func` may be a single callable or a list of callables applied in order.
    Uses a ConstantBeam reference for fair power-in normalization.
    '''
        
    if w0 is None:
        source = ConstantBeam(energy=E, simulation=simulation, z=0)
    else:
        source = GaussianBeam(energy=E, simulation=simulation, z=0, w0=w0)

    lens = lens_cls(f=f, R=R, n=n, wavelength=source.wavelength, simulation=simulation, z=0)
    if err_func is not None:
        for ef in (err_func if isinstance(err_func, (list, tuple)) else [err_func]):
            lens.add_error(ef)
    lens.init_transmittance(source)

    # Sampling check
    f_s = source.wavelength * np.abs(lens.f) / (2 * lens.R)
    if simulation.Lx / simulation.Nx >= f_s:
        print(f"[{label}] WARNING: under-sampled (dx={simulation.Lx/simulation.Nx:.3e} >= f_s={f_s:.3e})")

    # snapshot incident wave (after aperture mask but before phase) for power_in
    source.filter(lens)
    incident_power = total_power(source)

    # propagate to focal plane
    lens.transform(source)
    source.propagate(lens.f, propagator)

    return source, lens, incident_power


def collect_metrics(P_in, focal_wave, lens, label):
    I = focal_wave.intensity()
    fwhm = FWHM(focal_wave)
    I_max, I_avg = intensity_stats(focal_wave)
    P_focal = focal_power(focal_wave, radius=1.22*focal_wave.wavelength*lens.f/(2*lens.R))
    eff = focal_efficiency(P_in, focal_wave, radius=1.22*focal_wave.wavelength*lens.f/(2*lens.R))
    return {
        "label": label,
        "FWHM [m]": fwhm,
        "I_max": I_max,
        "I_avg": I_avg,
        "P_focal": P_focal,
        "P_incident": P_in,
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

def plot_comparison_1D(results, savepath):
    '''
    Plot lens phase, focal-plane intensity (log), and central line cut for an
    arbitrary set of lenses.

    `results` maps label -> (focal_wave, lens). One row per lens, three
    columns: phase, focal intensity, central cut.
    '''
    n_lenses = len(results)
    if n_lenses == 0: raise Exception("No lenses added.")

    fig, ax = plt.subplots(
        nrows=n_lenses, ncols=3,
        figsize=(12, 3.0 * n_lenses),
        squeeze=False,
        constrained_layout=True,
    )

    cmap_cycle = plt.get_cmap("tab10")

    for i, (name, (wave, lens, labels)) in enumerate(results.items()):

        lens.plot_profile(ax=ax[i, 0], savedir=savedir, labels=labels)
        
        ## Lens phase plot
        lens_ax = lens.view(ax=ax[i, 1], color=cmap_cycle(i%10))
        lens_ax.set(title=f"{name} Lens Phase")

        ## Wave intensity slice plot
        wave_ax = wave.view(ax=ax[i, 2], color=cmap_cycle(i%10))
        wave_ax.set(title=f"{name} Focal Intensity")

    fig.savefig(savepath)
    plt.close(fig)

def plot_comparison_2D(results, savepath, extend=False):
    '''
    Plot lens phase, focal-plane intensity (log), and central line cut for an
    arbitrary set of lenses.

    `results` maps label -> (focal_wave, lens). One row per lens, three
    columns: phase, focal intensity, central cut.
    '''
    n_lenses = len(results)
    if n_lenses == 0: raise Exception("No lenses added.")

    fig, ax = plt.subplots(
        nrows=n_lenses, ncols=4,
        figsize=(24, 5.0 * n_lenses),
        squeeze=False,
        constrained_layout=True,
    )

    cmap_cycle = plt.get_cmap("tab10")

    for i, (name, (wave, lens, labels)) in enumerate(results.items()):
        # prof_fig, prof_ax = plt.subplots()
        
        lens.plot_profile(ax=ax[i, 0], labels=labels)
        
        ## Lens phase plot
        lens_ax = lens.view(ax=ax[i, 1], cmap="twilight", labels=labels, show_cbar=True)
        lens_ax.set(title=f"{name} Lens Phase")
        
        ## Wave intensity/phase plot
        wave_ax = wave.view(ax=ax[i, 2], cmap="inferno", xlim=(-lens.R/2, lens.R/2), ylim=(-lens.R/2, lens.R/2), 
                            labels=labels, extend=extend, show_cbar=True)
        wave_ax.set(title=f"{name} Focal Intensity")

        ## Wave intensity slice plot
        x_scale_factor = labels.get("x_scale_factor", 1.0)
        xlabel = labels.get("xlabel", "x [m]")
        
        I = wave.intensity()
        Lx = wave.simulation.Lx
        Nx = I.shape[1]
        x = np.linspace(-Lx/2, Lx/2, Nx)
        cy = I.shape[0] // 2
        ax[i, 3].plot(x*x_scale_factor, I[cy, :], color=cmap_cycle(i % 10))
        ax[i, 3].set(xlim=(-lens.R/2*x_scale_factor, lens.R/2*x_scale_factor))
        ax[i, 3].set(title=f"{name} Central Cut", xlabel=xlabel, ylabel="Intensity", yscale="log")

    fig.savefig(savepath)
    plt.close(fig)

def test_compare_xray_lenses(lens_dict, N, dim, extend=False):
    print("Comparing Lenses...")
    
    # Parameters
    Lx = 1.5e-4
    Ly = 1.5e-4 if dim == 2 else None
    Lz = 10000
    E = 8e3       # eV
    f = 1.0         # m
    R = 5e-5        # m
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    
    print(f"Refractive index n = {n}")
    print(f"Energy = {E} eV, f = {f} m, R = {R} m")

    propagator = Propagator(angular_spectrum_method, dim=dim)

    metrics = []
    results = {}
    for name in lens_dict:
        sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
        label = "".join([c for c in name if c.isalpha()])
        cls, n_quantized, err_func = lens_dict[name]
        source, lens, P_in = run_lens(
            label, cls, sim, propagator, E, f, R, n,
            err_func=err_func
        )
        m = collect_metrics(P_in, source, lens, name)
        metrics.append(m)
        
        plot_labels = {
            "label": name,
            "xlabel": r"x $[\mu m]$",
            "x_scale_factor": 1e6,
            "ylabel": r"y $[\mu m]$",
            "y_scale_factor": 1e6,
            "title": rf"{name} profile (f={lens.f:.3g} m, R={lens.R *1e6:.3g} $\mu m$)"
        }
        
        results[name] = (source, lens, plot_labels)

    print_comparison(metrics)

    if dim == 1:
        out = os.path.join(savedir, "Metrics_Lens_Comparison_1D.png")
        plot_comparison_1D(results, out)
    else:
        suffix = "3D" if extend else "2D"
        out = os.path.join(savedir, f"Metrics_Lens_Comparison_{suffix}.png")
        plot_comparison_2D(results, out, extend=extend)

    print(f"Saved comparison figure to {out}")

def take_user_input():
    iter_count = 1
    lens_dict = dict()
    BAR  = "#" * 60
    RULE = "-" * 60

    print(BAR)
    print("  Lens Comparison Setup")
    print(BAR)

    while iter_count < 5:
        print()
        print(f"[Lens #{iter_count}]   (blank lens type to finish)")
        print(RULE)
        print("  Available types: Parabolic, Kinoform")
        lens = input("  > Lens type: ").strip()

        # Lens type
        match lens:
            case "Parabolic":
                lens_dict_key_cls = XrayParabolicLens
            case "Kinoform":
                lens_dict_key_cls = Kinoform
            case "":
                print()
                print(BAR)
                print(f"  Done. {iter_count - 1} lens(es) configured.")
                print(BAR)
                break
            case _:
                raise Exception("Unknown lens type")
        key = lens + str(iter_count)
        lens_dict[key] = [lens_dict_key_cls]

        # Quantization
        print()
        print(f"  Quantization for {key}  (blank for ideal)")
        N = input("  > N-level approximation: ").strip()
        lens_dict[key].append(N if N else None)

        # Errors (multiple allowed)
        print()
        print(f"  Errors for {key}  (blank Error Type to stop adding)")
        print("  Available: Periodic Etch, Random Etch, Removal, Taper")
        errs = []
        while True:
            err_type = input(f"  > Error #{len(errs)+1} type: ").strip()
            match err_type:
                case "Periodic Etch":
                    err = float(input("      Error: "))
                    interval = int(input("      Interval: "))
                    errs.append(functools.partial(LensErrors.periodic_etch, err=err, interval=interval))
                case "Random Etch":
                    err = float(input("      Max Error: "))
                    interval = int(input("      Interval: "))
                    errs.append(functools.partial(LensErrors.random_etch, max_err=err, interval=interval))
                case "Removal":
                    if lens != "Kinoform": raise Exception("Phase wrapped lens required!")
                    m = int(input("      Lateral zone to start taper (-1 = outermost): "))
                    proportion = float(input("      Proportion: "))
                    errs.append(functools.partial(LensErrors.zone_removal, m=m, proportion=proportion, extend=True, remove_last=True))
                case "Taper":
                    if lens != "Kinoform": raise Exception("Phase wrapped lens required!")
                    err = float(input("      Error: "))
                    errs.append(functools.partial(LensErrors.sidewall_taper, err=err))
                case "":
                    break
                case _:
                    raise Exception("Unknown error type")
            print(f"    + added {err_type}")
        lens_dict[key].append(errs if errs else None)

        # Complete
        print()
        print(RULE)
        print(f"  Added {key}: quantization={lens_dict[key][1]}, errors={len(errs)}")
        print(RULE)
        iter_count += 1

    return lens_dict
        
def main():
    args = parser.parse_args()
    lenses = take_user_input()
    print(lenses)
    test_compare_xray_lenses(lenses, args.N, args.dim, extend=args.extend)

if __name__ == "__main__":
    main()