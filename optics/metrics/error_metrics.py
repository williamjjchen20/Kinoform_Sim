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
    

def _error_magnitude_param(error_func):
    '''
    Inspect a LensErrors function and return the name of its magnitude
    parameter (the first non-`lens` positional, e.g. `err` or `max_err`).
    '''
    params = list(inspect.signature(error_func).parameters)
    if not params:
        raise ValueError("error_func has no parameters")
    return params[1] if params[0] == "lens" else params[0]


def etch_error_vs_intensity(source_factory, lens_factory, propagator, error_func,
                            err_values, labels=None, E_range = None, error_kwargs=None, savepath=None):
    '''
    Sweep an etch error magnitude and record focal-plane intensity stats.

    args
    - source_factory: callable() -> fresh Waveform (with its own SimulationObject)
    - lens_factory:   callable(source, wavelength) -> fresh ThinLens bound to that source
    - propagator:     propagation function passed to wave.propagate
    - error_func:     staticmethod from LensErrors (e.g. LensErrors.random_etch)
    - err_values:     iterable of magnitudes to sweep over
    - error_kwargs:   extra kwargs forwarded to error_func (e.g. {"count": 50, "seed": 0})
    - savepath:       optional path to save the sweep plot

    Returns a dict with arrays: err, I_max, I_avg, P_focal, fwhm, strehl.
    '''
    error_kwargs = dict(error_kwargs or {})
    mag_name = _error_magnitude_param(error_func)
    err_values = np.asarray(list(err_values), dtype=float)

    # Reference (no-error) focal wave for Strehl normalization
    ref_source = source_factory()
    ref_lens = lens_factory(ref_source)
    ref_lens.init_transmittance(ref_source)
    ref_lens.transform(ref_source)
    ref_source.propagate(ref_lens.f, propagator)

    results = []
    if E_range is None: E_range = [ref_source.energy]

    for E_off in E_range:
        print("="*50)
        print("E [eV]:", E_off)
        P_eff = np.zeros_like(err_values)
        fwhm  = np.zeros_like(err_values)
        strehl = np.zeros_like(err_values)

        for i, e in enumerate(err_values):
            source = source_factory(E=E_off)
            ## create reference source lens for comparison
            lens = lens_factory(source, wavelength=ref_source.wavelength)
            source.filter(lens)
            source_in = source
            
            lens.add_error(error_func, **{mag_name: float(e)}, **error_kwargs)
            lens.init_transmittance(source)
            lens.transform(source)
            source.propagate(lens.f, propagator)

            Imax, _ = intensity_stats(source)
            P_eff[i] = focal_efficiency(wave_in=source, wave_out=source_in, radius=1.22*source.wavelength*lens.f/(2*lens.R))
            try:
                fwhm[i] = FWHM(source)
            except Exception:
                fwhm[i] = np.nan
            strehl[i] = strehl_ratio(ref_source, source)

            print(f"[{i+1}/{len(err_values)}] {mag_name}={e:.3e}  "
                f"I_max={Imax:.3e}  Strehl={strehl[i]:.3f}  "
                f"FWHM={fwhm[i]:.3e}  P_eff={P_eff[i]:.3e}")

        result = [P_eff, fwhm, strehl]
        results.append(result)

    if savepath is not None:
        if labels is None:
            labels = {}
        plot_sweep(err_values, results, labels, savepath)

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
    E, f, R = 8.5e3, 1.0, 5e-5
    dim = 2
    
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    propagator = functools.partial(angular_spectrum_method, dim=dim)

    def source_factory(E=E) -> Waveform:
        if dim == 1:
            sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N)
        else:
            sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
        return ConstantBeam(energy=E, simulation=sim, z=0)
    
    def lens_factory(src, wavelength = None, f=f, R=R, n=n) -> Kinoform:
        if wavelength is None: wavelength = src.wavelength
        return Kinoform(wavelength=wavelength, f=f, R=R, n=n,
                        simulation=src.simulation, z=0)
        
    ### Random etch error
    err_values = np.linspace(0, 5e-6, 12)
    E_range = np.linspace(0.9*E, 1.1*E, 3)
    n_metrics = 3
    
    labels = {
            "xlabel": [r"Maximum Etch Error $[\mu m]$"] * n_metrics,
            "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
            "xscale": ["linear"] * n_metrics,
            "yscale": ["linear", "linear", "linear"],
            "x_scale_factor":  1e6,                 # m -> um
            "y_scale_factor": [1.0, 1e6, 1.0],     # FWHM m -> um
            "label": [f"E={E_i/1000} keV" for E_i in E_range],
            "title": f"Random Etch for E={E/1000} keV Kinoform",
            "marker": ["o", "^", "s", "d", "x"],
            "color": ["red", "orange", "green", "blue", "purple"]
        }
    out = savedir / "etch_error_vs_intensity_random.png"
    etch_error_vs_intensity(
        source_factory, lens_factory, propagator,
        LensErrors.random_etch, err_values,
        E_range=E_range,
        labels=labels,
        error_kwargs={"interval": 3, "seed": 42},
        savepath=out,
    )
    
    err_values = -np.linspace(0, 2.5e-6, 12)
    labels = {
            "xlabel": [r"Maximum Etch Error $[\mu m]$"] * n_metrics,
            "ylabel": ["Focal Efficiency", r"FWHM $[\mu m]$", "Strehl Ratio"],
            "xscale": ["linear"] * n_metrics,
            "yscale": ["linear", "linear", "linear"],
            "x_scale_factor":  -1e6,                 # m -> um
            "y_scale_factor": [1.0, 1e6, 1.0],     # FWHM m -> um
            "label": [f"E={E_i/1000} keV" for E_i in E_range],
            "title": f"Periodic Etch for E={E/1000} keV Kinoform",
            "marker": ["o", "^", "s", "d", "x"],
            "color": ["red", "orange", "green", "blue", "purple"]
        }
    out = savedir / "etch_error_vs_intensity_periodic.png"
    etch_error_vs_intensity(
        source_factory, lens_factory, propagator,
        LensErrors.periodic_etch, err_values,
        E_range=E_range,
        labels=labels,
        error_kwargs={"interval": 3},
        savepath=out,
    )
    
if __name__ == "__main__":
    main()