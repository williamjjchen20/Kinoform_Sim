import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xraylib as xrl
import os, functools
from pathlib import Path
import argparse
import time, tracemalloc

from ..propagators import *
from ..classes import *
from .metrics import *
from .lens_comparison import load_lens_params

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./test_figs").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, required=True)
parser.add_argument("--dim", type=int, choices=[1, 2], default=2,
                    help="simulation dimensionality (1 or 2)")
parser.add_argument("--precision", type=str, choices=["single", "double"], default="single",
                    help="simulation precision (single or double)")
parser.add_argument("--extend", action="store_true", default=False,
                    help="render 2D focal/lens views as 3D surfaces")
parser.add_argument("--mark-fwhm", action="store_true", default=True,
                    help="mark FWHM on the central-cut slice plots")
parser.add_argument("--params", type=str, default=None,
                    help="saved parameter JSON (see simulation/run.py); skips the prompts, requires --spread")
parser.add_argument("--spread", type=float, default=None,
                    help="energy spread in each direction [%%], used with --params")
parser.add_argument("--samples", type=int, default=7,
                    help="number of energy samples, used with --params")

MATERIAL, DENSITY = "Si", 2.329

def make_spectrum(E0, spread_pct, n_samples):
    '''
    Evenly spaced energies over E0*(1 +/- spread_pct/100) with flat weights.
    '''
    offsets = np.linspace(-spread_pct, spread_pct, n_samples) if n_samples > 1 else np.zeros(1)
    return E0*(1 + offsets/100), np.ones(n_samples)

def check_spectral_sampling(wavelength, f, R, spread_pct, n_samples):
    '''
    Warn if neighbouring energies focus further apart than the depth of focus,
    using the refractive worst case f ~ E^2. The summed focal spot then depends
    on where the samples happen to land instead of representing the bandwidth.
    '''
    if n_samples < 2 or spread_pct == 0: return
    focal_step = 2*f*(2*spread_pct/100)/(n_samples-1)
    depth_of_focus = 2*wavelength*(f/(2*R))**2
    if focal_step > depth_of_focus:
        needed = int(np.ceil(4*f*(spread_pct/100)/depth_of_focus)) + 1
        print(f"[WARNING: under-sampled spectrum (focal step {focal_step:.3e} m > depth of focus "
              f"{depth_of_focus:.3e} m). Use about {needed} or more energy samples.")

def build_lens(lens_cls, simulation: SimulationObject, E0: float, f: float, R: float, n: float | complex,
               err_func=None, zone_height=None, material=MATERIAL, density=DENSITY, lens_opts=None):
    '''
    Builds a lens of class `lens_cls` designed for E0 and applies its errors.

    `n` is the design index at E0 and sets the lens profile.
    `err_func` may be a single callable or a list of callables applied in order.
    `zone_height` is forwarded to Kinoform/FZP constructors when provided.
    `lens_opts` holds extra lens constructor arguments (Kinoform `full`, FZP `p`).
    '''
    design_wavelength = const.h*const.c/(E0*const.e)
    lens_kwargs = dict(f=f, R=R, n=n, wavelength=design_wavelength, simulation=simulation, z=0,
                       material=material, density=density)
    if zone_height is not None and lens_cls in (Kinoform, FZP):
        lens_kwargs["zone_height"] = zone_height
    lens_kwargs.update(lens_opts or {})
    lens = lens_cls(**lens_kwargs) #type: ignore

    if err_func is not None:
        for ef in (err_func if isinstance(err_func, (list, tuple)) else [err_func]):
            lens.add_error(ef)
    return lens

def run_lens(lens, propagator: Propagator, energies: np.ndarray, weights: np.ndarray, E0: float, w0=None):
    '''
    Sends one beam through `lens`, propagates to the design focal plane, and
    returns (focal_wave, incident_power). Empty `energies` means a monochromatic
    beam at E0. The transmittance uses the refractive index at each beam energy,
    so a lens can be reused for several beams.
    '''
    simulation = lens.simulation
    waveform_cls = ConstantBeam if w0 is None else GaussianBeam
    if energies.size == 0:
        source = waveform_cls(E0, simulation, 0, w0=w0)
    else:
        source = PolychromaticWaveform(energies, weights, waveform_cls, simulation, 0, w0=w0)

    lens.init_refractive_index(source)

    # Sampling check: the shortest wavelength is the most demanding
    f_s = np.min(source.wavelength) * np.abs(lens.f) / (2 * lens.R)
    if simulation.Lx / simulation.Nx >= f_s:
        print(f"[WARNING: under-sampled (dx={simulation.Lx/simulation.Nx:.3e} >= f_s={f_s:.3e})")

    # snapshot incident wave (after aperture mask but before phase) for power_in
    source.filter(lens)
    incident_power = total_power(source)

    # propagate to focal plane
    lens.transform(source)
    source.propagate(lens.f, propagator)

    return source, incident_power


def collect_metrics(P_in, focal_wave, lens, label, wavelength):
    '''
    `wavelength` is the design wavelength: it sets one Airy radius for the whole beam.
    '''
    fwhm = FWHM(focal_wave)
    I_max, I_avg = max_intensity(focal_wave), mean_intensity(focal_wave)
    radius = 1.22*wavelength*lens.f/(2*lens.R)
    P_focal = focal_power(focal_wave, radius=radius)
    eff = focal_efficiency(P_in, focal_wave, radius=radius)
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
    Plot lens profile and focal-plane intensity for each beam.

    `results` maps label -> (focal_wave, lens, labels). One row per beam,
    two columns: profile, focal intensity.
    '''
    n_beams = len(results)
    if n_beams == 0: raise Exception("No beams added.")

    figw, figh_per_row = 8, 2.0
    fig, ax = plt.subplots(
        nrows=n_beams, ncols=2,
        figsize=(figw, figh_per_row * n_beams),
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
            
        if i != n_beams-1:
            labels["xlabel"] = None
            
        text_scale = 1.2
            
        lens_axis =  ax[i, 0]
        lens.plot_profile(ax=lens_axis, labels=labels, R_min = R_min)
        title_fs = lens_axis.title.get_fontsize() * text_scale
        ax[i, 0].title.set_fontsize(title_fs)

        ## Wave intensity full view (no lens phase plot: it differs per energy)
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
    Plot lens profile, focal-plane intensity, and central line cut for each beam.

    `results` maps label -> (focal_wave, lens, labels). One row per beam, four
    columns: profile, empty (no lens phase), focal intensity, central cut.
    '''
    n_beams = len(results)
    if n_beams == 0: raise Exception("No beams added.")

    figw, figh_per_row = 24, 5.0
    fig, ax = plt.subplots(
        nrows=n_beams, ncols=4,
        figsize=(figw, figh_per_row * n_beams),
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
        
        ## No lens phase plot: it differs per energy, so that panel stays empty
        ax[i, 1].set_axis_off()

        ## Wave intensity plot: zoom around focal spot
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

def test_compare_beams(lens_cls, lens_name, err_func, zone_height, display_zones,
                       N, dim, precision, E0, spread_pct, n_samples, extend=False, mark_fwhm=False,
                       f=0.1, R=1e-4, material=MATERIAL, density=DENSITY, w0=None, lens_opts=None):
    print(f"Comparing monochromatic and polychromatic beams through {lens_name}...")

    # Parameters
    Lx = 3e-4
    Ly = Lx if dim == 2 else None
    Lz = 10000
    # f = 1.0        # m
    # R = 5e-5       # m

    energies, weights = make_spectrum(E0, spread_pct, n_samples)
    design_wavelength = const.h*const.c/(E0*const.e)
    n = xrl.Refractive_Index(material, E0 / 1000, density)

    print(f"Design refractive index n = {n}")
    print(f"Energy = {E0} eV +/- {spread_pct}% ({energies[0]:.2f}-{energies[-1]:.2f} eV, {n_samples} samples), f = {f} m, R = {R} m")
    check_spectral_sampling(design_wavelength, f, R, spread_pct, n_samples)

    mono = (np.array([]), np.array([])) # empty energies: monochromatic at E0

    ## Initialize box sizes
    ref_sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N, precision=precision)
    ref_lens = build_lens(Kinoform, ref_sim, E0, f, R, n, material=material, density=density)
    ref_source, P_in = run_lens(ref_lens, AngularSpectrum(ref_sim), *mono, E0, w0=w0)

    ## Initialize zoom parameters: an ideal monochromatic focus sets a shared window and y-limit
    m = collect_metrics(P_in, ref_source, ref_lens, "reference", design_wavelength)
    ref_fwhm = m.get("fwhm", np.inf)
    ref_I = m.get("I_max", 1.)
    # print(ref_I)
    if ref_fwhm < Lx:
        Lx_zoom = min(ref_fwhm*10, Lx) # coarse grids can widen the spot past the box
        Rx = Lx_zoom/Lx
        if dim == 2:
            Ly_zoom = min(ref_fwhm*10, Ly) #type: ignore
            Ry = Ly_zoom/Ly
            print(f"Zooming by: {Rx:3f}, {Ry:3f}")
        else:
            Ly_zoom = None
            Ry = 1.
            print(f"Zooming by: {Rx:3f}")
    else:
        raise Exception("Invalid FWHM!")
    
    ## One lens for both beams, so errors (including random ones) are identical
    sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N, precision=precision)
    propagator = ScaledAngularSpectrum(sim, Rx=Rx, Ry=Ry)
    lens = build_lens(lens_cls, sim, E0, f, R, n, err_func=err_func, zone_height=zone_height,
                      material=material, density=density, lens_opts=lens_opts)

    ## Collect metrics for each beam
    metrics = []
    results = {}
    beams = {"Mono": mono, f"Poly +/-{spread_pct:g}%": (energies, weights)}
    for beam_label, (beam_energies, beam_weights) in beams.items():
        source, P_in = run_lens(lens, propagator, beam_energies, beam_weights, E0, w0=w0)
        m = collect_metrics(P_in, source, lens, beam_label, design_wavelength)
        metrics.append(m)

        name = f"{lens_name} {beam_label}"

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
            "title": rf"{name}", #(E={E0/1000} keV, f={lens.f:.3g} m, R={lens.R *1e6:.3g} $\mu m$)",
            "fwhm": m.get("fwhm"),
        }
        
        results[name] = (source, lens, plot_labels)

    print_comparison(metrics)

    if dim == 1:
        out = os.path.join(savedir, f"Metrics_Polychromatic_Comparison_1D_{spread_pct:g}pct.png")
        plot_comparison_1D(results, out, mark_fwhm=mark_fwhm)
    else:
        suffix = "3D" if extend else "2D"
        out = os.path.join(savedir, f"Metrics_Polychromatic_Comparison_{suffix}_{spread_pct:g}pct.png")
        plot_comparison_2D(results, out, extend=extend, mark_fwhm=mark_fwhm)

    print(f"Saved comparison figure to {out}")

def _opt(prompt, cast, default):
    s = input(prompt).strip()
    return default if s == "" else cast(s)

def check_beam(E0, spread_pct, n_samples):
    if E0 <= 0: raise ValueError("Central energy must be positive")
    if not 0 <= spread_pct < 100: raise ValueError("Energy spread must be in [0, 100) %")
    if n_samples < 1: raise ValueError("Need at least one energy sample")

def take_beam_input():
    BAR  = "#" * 60

    print(BAR)
    print("  Polychromatic Beam Setup")
    print(BAR)

    E0 = _opt("  > Central energy [eV] [8000]: ", float, 8e3)
    spread_pct = _opt("  > Energy spread in each direction [%] [1.0]: ", float, 1.)
    n_samples = _opt("  > Energy samples [7]: ", int, 7)

    return E0, spread_pct, n_samples

def take_user_input():
    BAR  = "#" * 60
    RULE = "-" * 60

    print(BAR)
    print("  Lens Setup")
    print(BAR)

    print("  Available types: Parabolic, FZP, Kinoform")
    lens = input("  > Lens type: ").strip()

    # Lens type
    match lens:
        case "Parabolic":
            lens_cls = XrayParabolicLens
        case "FZP":
            lens_cls = FZP
        case "Kinoform":
            lens_cls = Kinoform
        case _:
            raise Exception("Unknown lens type")
        
    # Lens Name
    key = input("  > Lens name: ").strip()
    key = key if key else lens 

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

    # Errors (multiple allowed)
    print()
    print(f"  Errors for {key}  (blank Error Type to stop adding)")
    print("  Available: Cap Height, Cap Floor, Periodic Etch, Random Etch, Gaussian Etch, Removal, Taper, Zone Warping")
    errs = []

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
                errs.append(functools.partial(LensErrors.zone_removal, m=m, err=err, direction=direction, extend=extend, remove_last=remove_last))
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

    # Complete
    print()
    print(RULE)
    print(f"  Added {key}: errors={len(errs)}, zone_height={zone_height}, display_zones={display_zones}")
    print(RULE)

    return lens_cls, key, (errs if errs else None), zone_height, display_zones

def main():
    args = parser.parse_args()
    if args.params:
        if args.spread is None: parser.error("--params requires --spread")
        saved = load_lens_params(args.params)
        settings = saved["settings"]
        E0, spread_pct, n_samples = settings.pop("E"), args.spread, args.samples
        lens_cls, lens_name, err_func = saved["lens_cls"], saved["name"], None
        zone_height, display_zones = saved["zone_height"], saved["display_zones"]
    else:
        E0, spread_pct, n_samples = take_beam_input()
        lens_cls, lens_name, err_func, zone_height, display_zones = take_user_input()
        settings = {}
    check_beam(E0, spread_pct, n_samples)
    time_s = time.time(); tracemalloc.start()
    test_compare_beams(lens_cls, lens_name, err_func, zone_height, display_zones,
                       args.N, args.dim, args.precision, E0, spread_pct, n_samples,
                       extend=args.extend, mark_fwhm=args.mark_fwhm, **settings)
    time_e = time.time(); _, peak = tracemalloc.get_traced_memory()
    
    print(f"Simulation took {time_e-time_s:3f} seconds, with peak memory usage of {peak/10**9:3f} GB.")
    
if __name__ == "__main__":
    main()