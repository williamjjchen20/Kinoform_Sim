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
parser.add_argument("--extend", action="store_true", default=False,
                    help="render 2D focal/lens views as 3D surfaces")
parser.add_argument("--mark-fwhm", action="store_true", default=True,
                    help="mark FWHM on the central-cut slice plots")

def run_lens(lens_cls, simulation: SimulationObject, propagator: Propagator, 
             E: float, f: float, R: float, n: float | complex, 
             err_func=None, w0=None, zone_height=None):
    '''
    Initializes a source, applies a lens of class `lens_cls`, propagates to
    the focal plane, and returns (incident_wave, focal_wave, lens).
    
    `err_func` may be a single callable or a list of callables applied in order.
    Uses a ConstantBeam reference for fair power-in normalization.
    `zone_height` is forwarded to Kinoform/FZP constructors when provided.
    '''
        
    if w0 is None:
        source = ConstantBeam(energy=E, simulation=simulation, z=0)
    else:
        source = GaussianBeam(energy=E, simulation=simulation, z=0, w0=w0)

    lens_kwargs = dict(f=f, R=R, n=n, wavelength=source.wavelength, simulation=simulation, z=0)
    if zone_height is not None and lens_cls in (Kinoform, FZP):
        lens_kwargs["zone_height"] = zone_height
    lens = lens_cls(**lens_kwargs) #type: ignore
    
    if err_func is not None:
        for ef in (err_func if isinstance(err_func, (list, tuple)) else [err_func]):
            lens.add_error(ef)
    lens.init_transmittance(source)

    # Sampling check
    f_s = source.wavelength * np.abs(lens.f) / (2 * lens.R)
    if simulation.Lx / simulation.Nx >= f_s:
        print(f"[WARNING: under-sampled (dx={simulation.Lx/simulation.Nx:.3e} >= f_s={f_s:.3e})")

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
    I_max, I_avg = max_intensity(focal_wave), mean_intensity(focal_wave)
    P_focal = focal_power(focal_wave, radius=1.22*focal_wave.wavelength*lens.f/(2*lens.R))
    eff = focal_efficiency(P_in, focal_wave, radius=1.22*focal_wave.wavelength*lens.f/(2*lens.R))
    return {
        "label": label,
        "fwhm": fwhm,
        "I_max": I_max,
        "I_avg": I_avg,
        "P_focal": P_focal,
        "P_in": P_in,
        "eff": eff,
    }

def print_comparison(metrics_list: list[dict]):
    keys = metrics_list[0].keys()
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

def plot_comparison_1D(results, savepath, mark_fwhm=False):
    '''
    Plot lens profile, lens phase, focal-plane intensity, and a zoomed
    central cut for an arbitrary set of lenses.

    `results` maps label -> (focal_wave, lens, labels). One row per lens,
    four columns: profile, phase, focal intensity, central cut (log-scale,
    clipped near the geometric focus).
    '''
    n_lenses = len(results)
    if n_lenses == 0: raise Exception("No lenses added.")

    figw, figh_per_row = 8, 2.0
    fig, ax = plt.subplots(
        nrows=n_lenses, ncols=2,
        figsize=(figw, figh_per_row * n_lenses),
        squeeze=False,
        constrained_layout=True,
    )
    # title_fs = min(figw, figh_per_row) * 7.5#max(6, min(figw, figh_per_row) * 1.2)

    cmap_cycle = plt.get_cmap("tab10")

    for i, (name, (wave, lens, labels)) in enumerate(results.items()):
        
        try:
            zones = labels.get("zones")
            R_min = lens.zone_locations[-zones]
        except:
            R_min = -lens.R
            
        if i != n_lenses-1: 
            labels["xlabel"] = None
            
        text_scale = 1.2
            
        lens_axis =  ax[i, 0]
        lens.plot_profile(ax=lens_axis, labels=labels, R_min = R_min)
        title_fs = lens_axis.title.get_fontsize() * text_scale
        ax[i, 0].title.set_fontsize(title_fs)

        ## Lens phase plot
        phase_axis = ax[i, 1]
        lens_labels = dict(labels)
        lens_labels["xlim"] = (-lens.R/4, lens.R/4)
        lens_labels["ylim"] = None
        lens_labels["y_scale_factor"] = 1.
        lens_ax = lens.view(ax=phase_axis, labels=lens_labels, color=cmap_cycle(i%10))
        lens_ax.set(title=f"{name} Lens Phase")
        lens_ax.title.set_fontsize(title_fs)

        ## Wave intensity full view
        wave_axis = ax[i, 1]
        wave_labels = dict(labels)
        wave_labels["xlim"] = None
        wave_labels["y_scale_factor"] = 1.
        wave_labels["title"] = "Focal Intensity"
        wave_ax = wave.view(ax=wave_axis, labels=wave_labels, color=cmap_cycle(i%10))
        # wave_ax.set(title=f"{name} Focal Intensity")
        title_fs = wave_axis.title.get_fontsize() * text_scale
        wave_ax.title.set_fontsize(title_fs)

        if mark_fwhm:
            fwhm = labels.get("fwhm")
            x_scale_factor = labels.get("x_scale_factor", 1.0)
            if fwhm is not None:
                wave_axis.axhline(0.5, color="gray", lw=0.8, ls="--")
                wave_axis.axvspan(-fwhm/2 * x_scale_factor, fwhm/2 * x_scale_factor,
                                 color="crimson", alpha=0.15,
                                 label=f"FWHM={fwhm:.2e} m")
                wave_axis.legend(fontsize=8)

    fig.savefig(savepath)
    plt.close(fig)

def plot_comparison_2D(results, savepath, extend=False, mark_fwhm=False):
    '''
    Plot lens phase, focal-plane intensity (log), and central line cut for an
    arbitrary set of lenses.

    `results` maps label -> (focal_wave, lens). One row per lens, three
    columns: phase, focal intensity, central cut.
    '''
    n_lenses = len(results)
    if n_lenses == 0: raise Exception("No lenses added.")

    figw, figh_per_row = 24, 5.0
    fig, ax = plt.subplots(
        nrows=n_lenses, ncols=4,
        figsize=(figw, figh_per_row * n_lenses),
        squeeze=False,
        constrained_layout=True,
    )
    title_fs = max(6, min(figw / 4, figh_per_row) * 3)

    cmap_cycle = plt.get_cmap("tab10")

    for i, (name, (wave, lens, labels)) in enumerate(results.items()):
        # prof_fig, prof_ax = plt.subplots()
        try:
            zones = labels.get("zones")
            R_min = lens.zone_locations[-zones]
        except:
            R_min = -lens.R
            
        lens.plot_profile(ax=ax[i, 0], labels=labels, R_min=R_min)
        ax[i, 0].title.set_fontsize(title_fs)
        
        ## Lens phase plot
        lens_labels = dict(labels)
        lens_labels["extent"] = [-2*lens.R, 2*lens.R, -2*lens.R, 2*lens.R]
        lens_labels["xlim"] = None
        lens_labels["ylim"] = None
    
        lens_ax = lens.view(ax=ax[i, 1], labels=lens_labels, show_cbar=True)
        lens_ax.set(title=f"{name} Lens Phase")
        lens_ax.title.set_fontsize(title_fs)

        ## Wave intensity/phase plot: zoom around focal spot
        # focal_extent = [-lens.R/2, lens.R/2, -lens.R/2, lens.R/2]
        wave_labels = dict(labels)
        wave_labels["xlim"] = None
        wave_labels["ylim"] = None
        wave_ax = wave.view(ax=ax[i, 2],
                            labels=wave_labels,
                            _3d=extend, show_cbar=True)
        wave_ax.set(title=f"{name} Focal Intensity")
        wave_ax.title.set_fontsize(title_fs)

        ## Wave intensity slice plot
        x_scale_factor = labels.get("x_scale_factor", 1.0)
        xlabel = labels.get("xlabel", "x [m]")
        yscale = labels.get("yscale", "linear")
        ylim = labels.get("ylim", None)
        
        I = wave.intensity()
        x = labels["extent"][0]
        Nx = I.shape[1]
        x = np.linspace(-x, x, Nx)
        cy = I.shape[0] // 2
        ax[i, 3].plot(x*x_scale_factor, I[cy, :], color=cmap_cycle(i % 10))
        # ax[i, 3].set(xlim=(-lens.R/2*x_scale_factor, lens.R/2*x_scale_factor))
        ax[i, 3].set(title=f"{name} Central Cut", xlabel=xlabel, ylim=ylim, ylabel="Intensity", yscale=yscale)
        ax[i, 3].title.set_fontsize(title_fs)

        if mark_fwhm:
            fwhm = labels.get("fwhm")
            if fwhm is not None:
                ax[i, 3].axhline(0.5 * I[cy, :].max(), color="gray", lw=0.8, ls="--")
                ax[i, 3].axvspan(-fwhm/2 * x_scale_factor, fwhm/2 * x_scale_factor,
                                 color="crimson", alpha=0.15,
                                 label=f"FWHM={fwhm:.2e} m")
                ax[i, 3].legend(fontsize=8)

    fig.savefig(savepath)
    plt.close(fig)

def test_compare_xray_lenses(lens_dict, N, dim, extend=False, mark_fwhm=False):
    print("Comparing Lenses...")
    
    # Parameters
    Lx = 3e-4
    Ly = Lx if dim == 2 else None
    Lz = 10000
    E = 8e3        # eV
    f = 0.1        # m
    R = 1e-4       # m
    f = 1.0        # m
    R = 5e-5       # m
    
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    
    print(f"Refractive index n = {n}")
    print(f"Energy = {E} eV, f = {f} m, R = {R} m")
    
    ## Initialize box sizes
    ref_sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=1024)
    ref_source, ref_lens, P_in = run_lens(
        Kinoform, ref_sim, AngularSpectrum(ref_sim), E, f, R, n
    )
    m = collect_metrics(P_in, ref_source, ref_lens, "reference")
    ref_fwhm = m.get("fwhm", np.inf)
    ref_I = m.get("I_max", 1.)
    # print(ref_I)
    if ref_fwhm < Lx:
        Lx_zoom = ref_fwhm*10
        Rx = Lx_zoom/Lx
        if dim == 2:
            Ly_zoom = ref_fwhm*10
            Ry = Ly_zoom/Ly
            print(f"Zooming by: {Rx:3f}, {Ry:3f}")
        else:
            Ly_zoom = None
            Ry = 1.
            print(f"Zooming by: {Rx:3f}")
    else:
        raise Exception("Invalid FWHM!")
    
    
    ## Collect metrics for all lenses
    metrics = []
    results = {}
    for name in lens_dict:
        sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)
        propagator = ScaledAngularSpectrum(sim, Rx=Rx, Ry=Ry)
        
        entry = lens_dict[name]
        cls, err_func, zone_height = entry[0], entry[1], entry[2]
        display_zones = entry[3] if len(entry) > 3 else None
        source, lens, P_in = run_lens(
            cls, sim, propagator, E, f, R, n,
            err_func=err_func, zone_height=zone_height
        )
        m = collect_metrics(P_in, source, lens, name)
        metrics.append(m)
        
        plot_labels = {
            "label": name,
            "extent": [-Lx_zoom/2, Lx_zoom/2, -Ly_zoom/2, Ly_zoom/2] if dim == 2 else [-Lx_zoom/2, Lx_zoom/2], #type:ignore
            "xlabel": r"x $[\mu m]$",
            "xlim": None,
            "xscale": "linear",
            "x_scale_factor": 1e6,
            "ylabel": r"y $[\mu m]$",
            "ylim": (0, 1.5*ref_I),
            "yscale": "linear",
            "y_scale_factor": 1e6,
            "zones": display_zones,
            "title": rf"{name}", #(E={E/1000} keV, f={lens.f:.3g} m, R={lens.R *1e6:.3g} $\mu m$)",
            "fwhm": m.get("fwhm"),
        }
        
        results[name] = (source, lens, plot_labels)

    print_comparison(metrics)

    if dim == 1:
        out = os.path.join(savedir, "Metrics_Lens_Comparison_1D.png")
        plot_comparison_1D(results, out, mark_fwhm=mark_fwhm)
    else:
        suffix = "3D" if extend else "2D"
        out = os.path.join(savedir, f"Metrics_Lens_Comparison_{suffix}.png")
        plot_comparison_2D(results, out, extend=extend, mark_fwhm=mark_fwhm)

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
        print("  Available types: Parabolic, FZP, Kinoform")
        lens = input("  > Lens type: ").strip()

        # Lens type
        match lens:
            case "Parabolic":
                lens_dict_key_cls = XrayParabolicLens
            case "FZP":
                lens_dict_key_cls = FZP
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
            
        # Lens Name
        key = input("  > Lens name: ").strip()
        key = key if key else lens + str(iter_count)

        # Zone height (Kinoform / FZP only)
        zone_height = None
        if lens in ("Kinoform", "FZP"):
            zh_str = input("  > Zone height (blank = default λ/2δ for FZP, λ/δ for Kinoform): ").strip()
            zone_height = float(zh_str) if zh_str else None

        # Number of outer zones to display in plot_profile (Kinoform / FZP only)
        display_zones = None
        if lens in ("Kinoform", "FZP"):
            dz_str = input("  > Outer zones to display in profile plot (blank = all): ").strip()
            display_zones = int(dz_str) if dz_str else None

        lens_dict[key] = [lens_dict_key_cls, None, zone_height, display_zones]

        # Errors (multiple allowed)
        print()
        print(f"  Errors for {key}  (blank Error Type to stop adding)")
        print("  Available: Cap Height, Cap Floor, Periodic Etch, Random Etch, Gaussian Etch, Removal, Taper, Zone Warping")
        errs = []

        def _opt(prompt, cast, default):
            s = input(prompt).strip()
            return default if s == "" else cast(s)

        while True:
            err_type = input(f"  > Error #{len(errs)+1} type: ").strip()
            match err_type:
                case "Cap Height":
                    h = float(input("      Height (or proportion if proportion=True): "))
                    proportion = _opt("      Proportion? [False]: ", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), False)
                    errs.append(functools.partial(LensErrors.cap_height, h=h, proportion=proportion))
                case "Cap Floor":
                    h = float(input("      Floor (or proportion if proportion=True): "))
                    proportion = _opt("      Proportion? [False]: ", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), False)
                    errs.append(functools.partial(LensErrors.cap_floor, h=h, proportion=proportion))
                case "Periodic Etch":
                    err = float(input("      Error: "))
                    interval = _opt("      Interval [1]: ", int, 1)
                    errs.append(functools.partial(LensErrors.periodic_etch, err=err, interval=interval))
                case "Random Etch":
                    err = float(input("      Max Error: "))
                    interval = _opt("      Interval [1]: ", int, 1)
                    distribution = _opt("      Distribution (uniform/gaussian/cauchy/exponential) [uniform]: ", str, None)
                    seed = _opt("      Seed [0]: ", int, None)
                    errs.append(functools.partial(LensErrors.random_etch, max_err=err, interval=interval, distribution=distribution, seed=seed))
                case "Gaussian Etch":
                    err = float(input("      Max Error: "))
                    invert = _opt("      Invert? [False]: ", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), False)
                    seed = _opt("      Seed [0]: ", int, None)
                    errs.append(functools.partial(LensErrors.gaussian_etch, max_err=err, invert=invert, seed=seed))
                case "Removal":
                    if lens != "Kinoform" and lens != "FZP": raise Exception("Phase wrapped lens required!")
                    m = int(input("      Lateral zone to start taper (-1 = outermost): "))
                    err = float(input("      Error: "))
                    direction = _opt("      Direction (in/out) [out]: ", str, "out")
                    extend = _opt("      Extend? [True]: ", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), True)
                    remove_last = _opt("      Remove last? [True]: ", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), True)
                    errs.append(functools.partial(LensErrors.zone_removal, m=m, err=proportion, direction=direction, extend=extend, remove_last=remove_last))
                case "Taper":
                    err = float(input("      Error: "))
                    proportion = _opt("      Proportion [1.0]: ", float, 1.)
                    if lens == "Kinoform":
                        errs.append(functools.partial(LensErrors.kinoform_sidewall_taper, err=err, proportion=proportion))
                    elif lens == "FZP":
                        errs.append(functools.partial(LensErrors.kinoform_sidewall_taper, err=err, proportion=proportion))
                    else:
                        raise Exception("Phase wrapped lens required!")
                case "Zone Warping":
                    if lens != "Kinoform": raise Exception("Kinoform required for zone warping!")
                    beam_width = float(input("      Beam width [m]: "))
                    R_min = _opt("      R_min [blank = 0]: ", float, None)
                    R_max = _opt("      R_max [blank = lens R]: ", float, None)
                    # mag = _opt("      Magnitude [1.0]: ", float, 1.)
                    errs.append(functools.partial(LensErrors.kinoform_zone_warping, beam_width=beam_width, R_min=R_min, R_max=R_max))
                        
                case "":
                    break
                case _:
                    raise Exception("Unknown error type")
            print(f"    + added {err_type}")
        lens_dict[key][1] = errs if errs else None

        # Complete
        print()
        print(RULE)
        print(f"  Added {key}: errors={len(errs)}, zone_height={lens_dict[key][2]}, display_zones={lens_dict[key][3]}")
        print(RULE)
        iter_count += 1

    return lens_dict
        
def main():
    args = parser.parse_args()
    lenses = take_user_input()
    print(lenses)
    test_compare_xray_lenses(lenses, args.N, args.dim, extend=args.extend, mark_fwhm=args.mark_fwhm)

if __name__ == "__main__":
    main()