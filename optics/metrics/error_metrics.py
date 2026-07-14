import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xraylib as xrl
import os, functools, inspect, argparse
from pathlib import Path

from ..propagators import *
from ..classes import *
from .metrics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./test_figs").resolve()
savedir.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, required=True,
                    help="grid resolution (Nx, and Ny when dim=2)")
parser.add_argument("--dim", type=int, choices=[1, 2], default=2,
                    help="simulation dimensionality (1 or 2)")
    
## Metrics that need per-run context (P_in, ref wave, ...) are bound below
## with functools.partial so every metric can be called uniformly as `f(wave)`.
_CONTEXT_METRICS = {"strehl_ratio", "focal_efficiency"}

def _bind_metric(metric_func, *, ref_source, P_in, radius):
    '''Wrap a metric so it can be called as `metric(wave)`.'''
    name = metric_func.__name__
    if name == "strehl_ratio":
        return functools.partial(strehl_ratio, ref_source)
    if name == "focal_efficiency":
        return functools.partial(focal_efficiency, P_in, radius=radius)
    return metric_func

def _safe_call(f, wave):
    try:
        return f(wave)
    except Exception:
        return np.nan

def error_metrics(source_factory, lens_factory, propagator, error_func, sweep_param,
                  err_values, metrics=(focal_efficiency, FWHM),
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
    - metrics:        tuple of metric callables. Each is called as `metric(wave)`;
                      strehl_ratio and focal_efficiency are auto-bound with the
                      per-run reference wave / P_in / airy radius.
    - error_kwargs:   extra kwargs forwarded to error_func (e.g. {"interval": 3, "seed": 0})
    - savepath:       optional path to save the sweep plot

    Returns (results, refs), where
      - results is a list (per E_off) of lists of metric arrays, ordered to match `metrics`.
      - refs is a list (per E_off) of dicts {metric_name: reference value} from the
        ideal (error-free, design-wavelength) reference lens illuminated at E_off.
    '''
    error_kwargs = dict(error_kwargs or {})
    err_values = np.asarray(list(err_values), dtype=float)
    metric_names = [m.__name__ for m in metrics]

    # Design-energy reference (used only to fix the lens wavelength)
    design_source = source_factory()

    results = []
    refs = []
    if E_range is None: E_range = [design_source.energy]

    for E_off in E_range:
        print("="*50)
        print("E [eV]:", E_off)
        out = {name: np.zeros(len(err_values)) for name in metric_names}

        # Per-energy ideal reference: design-wavelength lens, no errors,
        # illuminated by an off-energy source. Defines Strehl=1 at err=0.
        ref_source = source_factory(E=E_off)
        ref_lens = lens_factory(ref_source, wavelength=design_source.wavelength)
        ref_source.filter(ref_lens)
        ref_P_in = total_power(ref_source)
        ref_lens.init_transmittance(ref_source)
        ref_lens.transform(ref_source)
        ref_source.propagate(ref_lens.f, propagator)

        # Reference metric values (used for ylim_spread around baseline). Bind
        # the reference against itself so strehl_ratio(ref, ref) = 1.
        ref_radius = 1.22*ref_source.wavelength*ref_lens.f/(2*ref_lens.R)
        ref_vals = {}
        for m, name in zip(metrics, metric_names):
            bound = _bind_metric(m, ref_source=ref_source,
                                 P_in=ref_P_in, radius=ref_radius)
            ref_vals[name] = _safe_call(bound, ref_source)
        refs.append(ref_vals)
        print("ref:", "  ".join(f"{k}={v:.3e}" for k, v in ref_vals.items()))

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

            radius = 1.22*source.wavelength*lens.f/(2*lens.R)
            for m, name in zip(metrics, metric_names):
                bound = _bind_metric(m, ref_source=ref_source,
                                     P_in=P_in, radius=radius)
                out[name][i] = _safe_call(bound, source)

            print(f"[{i+1}/{len(err_values)}] {sweep_param}={e:.3e}  "
                  + "  ".join(f"{k}={out[k][i]:.3e}" for k in out))

        results.append([out[name] for name in metric_names])

    if savepath is not None:
        plot_sweep(err_values, results, labels or {}, savepath,
                   refs=refs, metrics=metric_names)

    return results, refs

def _draw_sweep_row(ax_row, err, vals, labels, row_title=None,
                    refs=None, metrics=None):
    '''Draw one sweep (all metrics) into the provided row of axes.

    Axis-limit controls (all optional, keyed in `labels`):
      - "xlim": list per metric of (lo, hi) tuples or None
      - "ylim": list per metric of (lo, hi) tuples or None (takes precedence)
      - "ylim_spread": list per metric of fractional half-widths of the reference
        value (e.g. 0.2 -> +/-20% of ref). When set, ylim is drawn as
        [ref*(1-spread), ref*(1+spread)] * y_scale_factor. Use None to skip.
      - "ref_series_idx": index into `refs` (per-E_off) to use as the baseline
        for ylim_spread. Defaults to the "design" series inferred from labels,
        or 0.
    '''
    n_metrics = len(vals[0])
    n_series  = len(vals)

    xlabel  = labels.get("xlabel",  [""] * n_metrics)
    ylabel  = labels.get("ylabel",  [""] * n_metrics)
    xscale  = labels.get("xscale",  ["linear"] * n_metrics)
    yscale  = labels.get("yscale",  ["linear"] * n_metrics)
    x_scale_factor  = labels.get("x_scale_factor",  1.0)
    y_scale_factor = labels.get("y_scale_factor", [1.0] * n_metrics)
    xlim        = labels.get("xlim",        [None] * n_metrics)
    ylim        = labels.get("ylim",        [None] * n_metrics)
    ylim_spread = labels.get("ylim_spread", [None] * n_metrics)

    label     = labels.get("label",     [""]     * n_series)
    marker    = labels.get("marker",    ["o"]    * n_series)
    color     = labels.get("color",     [None]   * n_series)
    linestyle = labels.get("linestyle", ["-"]    * n_series)
    alpha     = labels.get("alpha",     [1.0]    * n_series)
    linewidth = labels.get("linewidth", [1.5]    * n_series)
    zorder    = labels.get("zorder",    [2]      * n_series)

    err_plot = np.asarray(err) * x_scale_factor

    for i, val in enumerate(vals):
        for j, metric in enumerate(val):
            ax_row[j].plot(err_plot, np.asarray(metric) * y_scale_factor[j],
                           color=color[i], marker=marker[i], label=label[i],
                           linestyle=linestyle[i], alpha=alpha[i],
                           linewidth=linewidth[i], zorder=zorder[i])
            ax_row[j].set(xlabel=xlabel[j], ylabel=ylabel[j],
                          xscale=xscale[j], yscale=yscale[j])

    # Pick the reference series for ylim_spread (prefer the "design" series
    # inferred from label text, else the widest linewidth, else 0).
    if refs:
        ref_idx = labels.get("ref_series_idx")
        if ref_idx is None:
            for i, lbl in enumerate(label):
                if isinstance(lbl, str) and "design" in lbl.lower():
                    ref_idx = i; break
        if ref_idx is None:
            try: ref_idx = int(np.argmax(linewidth))
            except Exception: ref_idx = 0
        ref_for_row = refs[ref_idx] if 0 <= ref_idx < len(refs) else None
    else:
        ref_for_row = None

    for j in range(n_metrics):
        if xlim[j] is not None:
            ax_row[j].set_xlim(xlim[j])
        if ylim[j] is not None:
            ax_row[j].set_ylim(ylim[j])
        elif ylim_spread[j] is not None and ref_for_row is not None and metrics:
            m_name = metrics[j]
            ref_v = ref_for_row.get(m_name)
            if ref_v is not None and np.isfinite(ref_v):
                spread = float(ylim_spread[j]*ref_v)
                lo = (ref_v - spread) * y_scale_factor[j]
                hi = (ref_v + spread) * y_scale_factor[j]
                ax_row[j].set_ylim(lo, hi)

    if any(lbl for lbl in label):
        ax_row[-1].legend()

    if row_title:
        # left-side row label, doesn't disturb per-axis titles
        ax_row[0].set_ylabel(f"{row_title}\n{ax_row[0].get_ylabel()}")

def plot_sweep(err, vals, labels, savepath, refs=None, metrics=None):
    if not isinstance(vals, list): vals = [vals]
    assert isinstance(vals, list)
    n_metrics = len(vals[0])

    fig, ax = plt.subplots(1, n_metrics, figsize=(4.5*n_metrics, 4), constrained_layout=True, squeeze=False)
    fig.suptitle(labels.get("title", ""))
    _draw_sweep_row(ax[0], err, vals, labels, refs=refs, metrics=metrics)

    fig.savefig(savepath)
    plt.close(fig)
    print(f"Saved sweep figure to {savepath}.")

def _draw_sweep_column(ax_col, err, vals, labels, col_title=None,
                       refs=None, metrics=None):
    '''Draw one sweep (all metrics) into the provided column of axes.

    Vertical transpose of `_draw_sweep_row`: metric j is drawn into ax_col[j].
    Supports the same axis-limit controls documented on `_draw_sweep_row`.
    '''
    n_metrics = len(vals[0])
    n_series  = len(vals)

    xlabel  = labels.get("xlabel",  [""] * n_metrics)
    ylabel  = labels.get("ylabel",  [""] * n_metrics)
    xscale  = labels.get("xscale",  ["linear"] * n_metrics)
    yscale  = labels.get("yscale",  ["linear"] * n_metrics)
    x_scale_factor  = labels.get("x_scale_factor",  1.0)
    y_scale_factor = labels.get("y_scale_factor", [1.0] * n_metrics)
    xlim        = labels.get("xlim",        [None] * n_metrics)
    ylim        = labels.get("ylim",        [None] * n_metrics)
    ylim_spread = labels.get("ylim_spread", [None] * n_metrics)

    label     = labels.get("label",     [""]     * n_series)
    marker    = labels.get("marker",    ["o"]    * n_series)
    color     = labels.get("color",     [None]   * n_series)
    linestyle = labels.get("linestyle", ["-"]    * n_series)
    alpha     = labels.get("alpha",     [1.0]    * n_series)
    linewidth = labels.get("linewidth", [1.5]    * n_series)
    zorder    = labels.get("zorder",    [2]      * n_series)

    err_plot = np.asarray(err) * x_scale_factor

    for i, val in enumerate(vals):
        for j, metric in enumerate(val):
            ax_col[j].plot(err_plot, np.asarray(metric) * y_scale_factor[j],
                           color=color[i], marker=marker[i], label=label[i],
                           linestyle=linestyle[i], alpha=alpha[i],
                           linewidth=linewidth[i], zorder=zorder[i])
            ax_col[j].set(xlabel=xlabel[j], ylabel=ylabel[j],
                          xscale=xscale[j], yscale=yscale[j])

    if any(lbl for lbl in label):
        ax_col[0].legend()

    if col_title:
        ax_col[0].set_title(f"{col_title}\n{ax_col[0].get_title()}")

def plot_sweeps_combined(sweeps, savepath, suptitle=""):
    '''
    Combine multiple sweeps into a single figure: one column per sweep, rows
    are metric fields. x-axis is shared down each column (same error units)
    and y-axis is shared across each row (same metric).

    `sweeps` is a list of dicts with keys: name, err, vals, labels, and optionally
    refs (list of {metric: value} per series) and metrics (metric-name tuple).
    '''
    if not sweeps:
        print("No sweeps to combine; skipping combined figure.")
        return
    n_cols = len(sweeps)
    n_metrics = len(sweeps[0]["vals"][0])

    fig, ax = plt.subplots(n_metrics, n_cols,
                           figsize=(4.5*n_cols, 4*n_metrics),
                           sharex="col", sharey="row",
                           constrained_layout=True, squeeze=False)
    if suptitle:
        fig.suptitle(suptitle)

    for c, sw in enumerate(sweeps):
        _draw_sweep_column(ax[:, c], sw["err"], sw["vals"], sw["labels"],
                           col_title=sw["name"],
                           refs=sw.get("refs"), metrics=sw.get("metrics"))

    fig.savefig(savepath)
    plt.close(fig)
    print(f"Saved combined sweep figure to {savepath}.")

def main():
    args = parser.parse_args()
    N = args.N
    dim = args.dim

    Lx, Lz = 3e-4, 10000
    # SIM = {"Lx": Lx, "Ly": Lx, "Nx": N, "Ny": N, "Lz": Lz}
    E, f, R = 8.e3, 0.1, 1e-4
    
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
        
    E_range = np.array([0.999*E, E, 1.001*E])
    metrics= (focal_efficiency, FWHM, max_intensity) #("P_eff", "FWHM", "Strehl")

    # Foreground the design-energy curve; push off-energy curves back
    # (dashed, no markers, lower alpha, thinner) so the reference reads first.
    design_idx = int(np.argmin(np.abs(E_range - E)))
    n_E = len(E_range)
    is_design   = [i == design_idx for i in range(n_E)]
    series_color     = ["black" if d else "gray" for d in is_design]
    series_marker    = ["o"     if d else ""     for d in is_design]
    series_linestyle = ["-"     if d else "--"   for d in is_design]
    series_alpha     = [1.0     if d else 0.55   for d in is_design]
    series_linewidth = [1.8     if d else 1.0    for d in is_design]
    series_zorder    = [3       if d else 2      for d in is_design]
    series_labels    = [f"E={E_i/1000} keV" + (" (design)" if is_design[i] else "")
                        for i, E_i in enumerate(E_range)]
    series_style = {
        "label":     series_labels,
        "marker":    series_marker,
        "color":     series_color,
        "linestyle": series_linestyle,
        "alpha":     series_alpha,
        "linewidth": series_linewidth,
        "zorder":    series_zorder,
    }

    # Collect each accepted sweep so we can render a single combined figure at the end.
    sweeps = []
    def _run(name, error_func, sweep_param, err_values, labels, savepath,
             error_kwargs=None):
        # override per-series styling with the shared design/off-energy scheme
        labels = {**series_style, **labels}
        results, refs = error_metrics(
            source_factory, lens_factory, propagator,
            error_func, sweep_param, err_values,
            metrics=metrics, E_range=E_range,
            labels=labels, error_kwargs=error_kwargs, savepath=savepath,
        )
        sweeps.append({"name": name, "err": err_values,
                       "vals": results, "labels": labels,
                       "refs": refs,
                       "metrics": [m.__name__ for m in metrics]})
        return results, refs

    # Random etch error
    shared_labels = {"xlabel": [r"{Error}"] * len(metrics),
                    "ylabel": ["Focal Efficiency", r"FWHM $[nm]$", r"$I_{max}$"],
                    "xscale": ["linear"] * len(metrics),
                    "yscale": ["linear", "linear", "linear"],
                    "x_scale_factor":  1.0,
                    "y_scale_factor": [1.0, 1e9, 1.0],
                    "label": [f"E={E_i/1000} keV" for E_i in E_range],
                    "title": rf"Cap Floor for (E={E/1000} keV, f={f} m, R={R*1e6} $\mu m$) Kinoform",
                    "marker": ["o", "^", "s", "d", "x"],
                     }
    match input("Analyze random etch? (y/n): "):
        case "y": 
            err_values = np.linspace(0, 2e-6, 12)
            labels = {**shared_labels,
                    "xlabel": [r"Maximum Roughness $[\mu m]$"] * len(metrics),
                    # per-metric percentage around the design reference;
                    # None skips a metric and falls back to autoscale.
                    "ylim_spread": [0.2, 0.2, 0.2],
                    "x_scale_factor":  1e6,                 # m -> um
                }
            out = savedir / "etch_error_vs_intensity_random.png"
            _run("Random Etch", LensErrors.random_etch, "max_err", err_values,
                 labels, out, error_kwargs={"interval": 3, "seed": 42})
        case _:
            print("Skipping random etch.")
    
    ### Periodic etch error
    match input("Analyze periodic etch? (y/n): "):
        case "y": 
            err_values = -np.linspace(0, 2e-6, 12)
            labels = { **shared_labels,
                    "xlabel": [r"Roughness $[\mu m]$"] * len(metrics),
                    "ylim_spread": [0.2, 0.2, 0.2],
                    "x_scale_factor":  -1e6,                 # m -> um
                }
            out = savedir / "etch_error_vs_intensity_periodic.png"
            _run("Periodic Etch", LensErrors.periodic_etch, "err", err_values,
                 labels, out, error_kwargs={"interval": 3})
        case _:
            print("Skipping periodic etch.")
            
    ### Kinoform sidewall taper
    match input("Analyze kinoform sidewall taper? (y/n): "):
        case "y":
            err_values = np.linspace(0, 1e-7, 10)
            p = 1.0
            labels = { **shared_labels,
                    "xlabel": [r"Sidewall Taper Error $[nm]$"] * len(metrics),
                    "x_scale_factor":  1e9,                 # m -> um
                }
            out = savedir / "sidewall_taper_vs_intensity.png"
            _run("Sidewall Taper", LensErrors.kinoform_sidewall_taper, "err", err_values,
                 labels, out, error_kwargs={"proportion": p})
        case _:
            print("Skipping kinoform sidewall taper.")

    ### Cap height (sweep by proportion of full kinoform height)
    match input("Analyze cap height? (y/n): "):
        case "y":
            err_values = np.linspace(1.0, 0.9, 10)
            labels = { **shared_labels,
                    "xlabel": [r"Cap Height Proportion"] * len(metrics),
                    "x_scale_factor": 1.0,
                    "ylim_spread": [0.2, 0.2, 0.2],
                }
            out = savedir / "cap_height_vs_intensity.png"
            _run("Cap Height", LensErrors.cap_height, "h", err_values,
                 labels, out, error_kwargs={"proportion": True})
        case _:
            print("Skipping cap height.")

    ### Cap floor (sweep by proportion of full kinoform height)
    match input("Analyze cap floor? (y/n): "):
        case "y":
            err_values = np.linspace(1e-3, 0.1, 10)
            labels = { **shared_labels,
                    "xlabel": [r"Cap Floor Proportion"] * len(metrics),
                    "x_scale_factor": 1.0,
                    "ylim_spread": [0.2, 0.2, 0.2]
                }
            out = savedir / "cap_floor_vs_intensity.png"
            _run("Cap Floor", LensErrors.cap_floor, "h", err_values,
                 labels, out, error_kwargs={"proportion": True})
        case _:
            print("Skipping cap floor.")
            
    ### Zone warping (sweep beam_width)
    match input("Analyze zone warping? (y/n): "):
        case "y":
            err_values = np.logspace(-8, -6, 10)
            labels = { **shared_labels,
                    "xlabel": [r"Beam Width $[nm]$"] * len(metrics),
                    "xscale": ["log"] * len(metrics),
                }
            out = savedir / "zone_warping_vs_intensity.png"
            _run("Zone Warping", LensErrors.kinoform_zone_warping, "beam_width",
                 err_values, labels, out)
        case _:
            print("Skipping zone warping.")

    ### Combined figure across all accepted sweeps
    if sweeps:
        combined_out = savedir / "error_metrics_combined.png"
        plot_sweeps_combined(
            sweeps, combined_out,
            suptitle=rf"Error Metrics Combined (E={E/1000} keV, f={f} m, R={R*1e6} $\mu m$)"
        )

if __name__ == "__main__":
    main()