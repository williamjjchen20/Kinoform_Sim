import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xraylib as xrl
import os, functools, inspect
from pathlib import Path

from ..propagators import *
from ..classes import *
from .metrics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./results").resolve()
savedir.mkdir(parents=True, exist_ok=True)
    
def error_metrics(source_factory, lens_factory, propagator, error_func, sweep_param,
                  err_values, metrics=("P_eff", "FWHM", "Strehl"),
                  labels=None, E_range=None, error_kwargs=None, savepath=None):
    '''
    Sweep an error magnitude (etch depth, taper proportion, zone #, ...) and
    record the requested focal-plane metric fields for each value of E in `E_range`.

    args
    - source_factory: callable() -> fresh Waveform (with its own SimulationObject)
    - lens_factory:   callable(source, wavelength) -> fresh ThinLens bound to that source
    - propagator:     propagation function passed to wave.propagate
    - error_func:     staticmethod from LensErrors (e.g. LensErrors.random_etch)
    - sweep_param:    name of the kwarg on error_func that takes err_values
    - err_values:     iterable of magnitudes to sweep over
    - metrics:        subset of ("P_eff", "FWHM", "Strehl") to record
    - error_kwargs:   extra kwargs forwarded to error_func (e.g. {"interval": 3, "seed": 0})
    - savepath:       optional path to save the sweep plot

    Returns a list (per E_off) of lists of metric arrays, ordered to match `metrics`.
    '''
    error_kwargs = dict(error_kwargs or {})
    err_values = np.asarray(list(err_values), dtype=float)

    # Design-energy reference (used only to fix the lens wavelength)
    design_source = source_factory()

    results = []
    if E_range is None: E_range = [design_source.energy]

    for E_off in E_range:
        print("="*50)
        print("E [eV]:", E_off)
        out = {m: np.zeros(len(err_values)) for m in metrics}

        # Per-energy ideal reference: design-wavelength lens, no errors,
        # illuminated by an off-energy source. Defines Strehl=1 at err=0.
        ref_source = source_factory(E=E_off)
        ref_lens = lens_factory(ref_source, wavelength=design_source.wavelength)
        ref_source.filter(ref_lens)
        ref_lens.init_transmittance(ref_source)
        ref_lens.transform(ref_source)
        ref_source.propagate(ref_lens.f, propagator)

        for i, e in enumerate(err_values):
            source = source_factory(E=E_off)
            ## lens designed for the design energy, illuminated at E_off
            lens = lens_factory(source, wavelength=design_source.wavelength)
            source.filter(lens)
            P_in = total_power(source)

            lens.add_error(error_func, **{sweep_param: float(e)}, **error_kwargs)
            lens.init_transmittance(source)
            lens.transform(source)
            source.propagate(lens.f, propagator)

            Imax, _ = intensity_stats(source)
            if "P_eff" in out:
                out["P_eff"][i] = focal_efficiency(
                    P_in=P_in, wave_out=source,
                    radius=1.22*source.wavelength*lens.f/(2*lens.R))
            if "FWHM" in out:
                try: out["FWHM"][i] = FWHM(source)
                except Exception: out["FWHM"][i] = np.nan
            if "Strehl" in out:
                out["Strehl"][i] = strehl_ratio(ref_source, source)

            print(f"[{i+1}/{len(err_values)}] {sweep_param}={e:.3e}  I_max={Imax:.3e}  "
                  + "  ".join(f"{k}={out[k][i]:.3e}" for k in out))

        results.append([out[m] for m in metrics])

    if savepath is not None:
        plot_sweep(err_values, results, labels or {}, savepath)

    return results

def plot_sweep(err, vals, labels, savepath):
    if not isinstance(vals, list): vals = [vals]
    assert isinstance(vals, list)
    n_metrics = len(vals[0])
    n_series  = len(vals)

    fig, ax = plt.subplots(1, n_metrics, figsize=(4.5*n_metrics, 4), constrained_layout=True, squeeze=False)
    ax = ax[0]
    fig.suptitle(labels.get("title", ""))

    # per-axis labels / scales (length n_metrics)
    xlabel  = labels.get("xlabel",  [""] * n_metrics)
    ylabel  = labels.get("ylabel",  [""] * n_metrics)
    xscale  = labels.get("xscale",  ["linear"] * n_metrics)
    yscale  = labels.get("yscale",  ["linear"] * n_metrics)
    # multiplicative factors applied before plotting (length n_metrics)
    x_scale_factor  = labels.get("x_scale_factor",  1.0)
    y_scale_factor = labels.get("y_scale_factor", [1.0] * n_metrics)

    # per-series style (length n_series)
    label   = labels.get("label",   [""]   * n_series)
    marker  = labels.get("marker",  ["o"]  * n_series)
    color   = labels.get("color",   [None] * n_series)

    err_plot = np.asarray(err) * x_scale_factor

    for i, val in enumerate(vals):
        for j, metric in enumerate(val):
            ax[j].plot(err_plot, np.asarray(metric) * y_scale_factor[j],
                       color=color[i], marker=marker[i], label=label[i])
            ax[j].set(xlabel=xlabel[j], ylabel=ylabel[j],
                      xscale=xscale[j], yscale=yscale[j])

    if any(lbl for lbl in label):
        ax[-1].legend()

    fig.savefig(savepath)
    plt.close(fig)
    print(f"Saved sweep figure to {savepath}.")

def main():
    Lx, Lz, N = 1.5e-4, 10000, 2048
    # SIM = {"Lx": Lx, "Ly": Lx, "Nx": N, "Ny": N, "Lz": Lz}
    E, f, R = 8.e3, 1.0, 5e-5
    dim = 2
    
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    def source_factory(E=E) -> Waveform:
        if dim == 1:
            sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N)
        else:
            sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
        return ConstantBeam(energy=E, simulation=sim, z=0)

    propagator = AngularSpectrum(source_factory().simulation)
    
    def lens_factory(src, wavelength = None, f=f, R=R, n=n) -> Kinoform:
        if wavelength is None: wavelength = src.wavelength
        return Kinoform(wavelength=wavelength, f=f, R=R, n=n,
                        simulation=src.simulation, z=0)
    
    ## Initialize reference parameters
    ref_source = source_factory()
    ref_lens = lens_factory(ref_source)
        
    E_range = np.array([0.99*E, E, 1.01*E])
    metrics=("P_eff", "FWHM", "Strehl")
    
    # Random etch error
    match input("Analyze random etch? (y/n): "):
        case "y": 
            err_values = np.linspace(0, 2e-6, 12)
            labels = {
                    "xlabel": [r"Maximum Etch Depth $[\mu m]$"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1e6,                 # m -> um
                    "y_scale_factor": [1.0, 1e6, 1.0],     # FWHM m -> um
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Random Etch for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "etch_error_vs_intensity_random.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.random_etch, "max_err", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                error_kwargs={"interval": 3, "seed": 42},
                savepath=out,
            )
        case _:
            print("Skipping random etch.")
    
    ### Periodic etch error
    match input("Analyze periodic etch? (y/n): "):
        case "y": 
            err_values = -np.linspace(0, 2e-6, 12)
            labels = {
                    "xlabel": [r"Etch Depth $[\mu m]$"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  -1e6,                 # m -> um
                    "y_scale_factor": [1.0, 1e6, 1.0],     # FWHM m -> um
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Periodic Etch for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "etch_error_vs_intensity_periodic.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.periodic_etch, "err", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                error_kwargs={"interval": 3},
                savepath=out,
            )
        case _:
            print("Skipping periodic etch.")
            
    ### Kinoform sidewall taper
    match input("Analyze kinoform sidewall taper? (y/n): "):
        case "y":
            err_values = np.linspace(0, 2e-7, 10)
            p = 1.0
            labels = {
                    "xlabel": [r"Sidewall Taper Error $[\mu m]$"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1e6,                 # m -> um
                    "y_scale_factor": [1.0, 1e6, 1.0],
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Sidewall Taper (proportion={p}) for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "sidewall_taper_vs_intensity.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.kinoform_sidewall_taper, "err", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                error_kwargs={"proportion": p},
                savepath=out,
            )
        case _:
            print("Skipping kinoform sidewall taper.")

    ### Cap height (sweep by proportion of full kinoform height)
    match input("Analyze cap height? (y/n): "):
        case "y":
            err_values = np.linspace(1.0, 0.5, 10)
            labels = {
                    "xlabel": [r"Cap Height Proportion"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1.0,
                    "y_scale_factor": [1.0, 1e6, 1.0],
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Cap Height for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "cap_height_vs_intensity.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.cap_height, "h", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                error_kwargs={"proportion": True},
                savepath=out,
            )
        case _:
            print("Skipping cap height.")

    ### Cap floor (sweep by proportion of full kinoform height)
    match input("Analyze cap floor? (y/n): "):
        case "y":
            err_values = np.linspace(1e-3, 0.5, 10)
            labels = {
                    "xlabel": [r"Cap Floor Proportion"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1.0,
                    "y_scale_factor": [1.0, 1e6, 1.0],
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Cap Floor for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "cap_floor_vs_intensity.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.cap_floor, "h", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                error_kwargs={"proportion": True},
                savepath=out,
            )
        case _:
            print("Skipping cap floor.")
            
    ### Zone warping (sweep beam_width)
    match input("Analyze zone warping? (y/n): "):
        case "y":
            err_values = np.logspace(-8, -6, 10)
            labels = {
                    "xlabel": [r"Beam Width $[\mu m]$"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
                    "xscale": ["log"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1e6,
                    "y_scale_factor": [1.0, 1e6, 1.0],
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": f"Zone Warping for E={E/1000} keV Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                    "color": ["red", "orange", "green", "blue", "purple"]
                }
            out = savedir / "zone_warping_vs_intensity.png"
            error_metrics(
                source_factory, lens_factory, propagator,
                LensErrors.kinoform_zone_warping, "beam_width", err_values,
                metrics=metrics,
                E_range=E_range,
                labels=labels,
                savepath=out,
            )
        case _:
            print("Skipping zone warping.")
            
if __name__ == "__main__":
    main()