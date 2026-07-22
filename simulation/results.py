import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import time
from pathlib import Path

import xraylib as xrl
from ..optics import classes
from .plotting import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "test_figs/").resolve()

## Global param definitions
Lx, Lz = 3e-4, 10000
N = 150000

E = 8.e3
f = 0.1
R = 1e-4
# f = 1.0
# R = 5e-5
n = xrl.Refractive_Index("Si", E / 1000, 2.329)

seed = 42
def build_errors(lens_type, height):
    if lens_type is Kinoform:
        return [(LensErrors.kinoform_height_shrink, {"height": height, "proportion": False}),
                (LensErrors.kinoform_zone_shrink, {"err": 5e-9}),
                (LensErrors.kinoform_sidewall_taper, {"err":1e-8, "proportion":1., "zone_shift":False}),
                (LensErrors.cap_height, {"h": 0.99*height, "proportion": False}),
                (LensErrors.cap_floor, {"h": 0.02, "proportion": True}),
                (LensErrors.random_etch, {"max_err": 5e-9, "interval": 2, "distribution": "gaussian", "seed": seed})]
    else:
        return [(LensErrors.FZP_sidewall_taper, {"err": 1e-8, "proportion": 1.0}),
                (LensErrors.cap_floor, {"h": 0.02, "proportion": True}),
                (LensErrors.cap_height, {"h": height, "proportion": False}),
                (LensErrors.random_etch, {"max_err": 5e-9, "interval": 2, "distribution": "gaussian", "seed": seed})]
        
        
## Results

## Propagation plots

def sideview_plot():

    savedir.mkdir(parents=True, exist_ok=True)

    # E, f, R = 8e3, 0.1, 1e-4
    # E, f, R = 8e3, 1.0, 5e-5
    # Lx, N = 3e-4, 100000
    # Lz = 2*f
    # n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N, Ny=N)
    source = GaussianBeam(energy=E, simulation=sim, z=0, w0=5e-4)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=f/2)

    dz = 0.001
    z_lim = f + lens.z
    assert (z_lim <= Lz)
    
    t_start = time.time()
    view = simulation_sideview(sim, z_max=Lz, dz=dz)
    
    I = np.abs(view)**2
    t_end = time.time()
    print(f"Time Taken: {t_end-t_start} s")

    fig, ax = plt.subplots(figsize=(10, 5))
    print(I.max())
    norm = colors.LogNorm(vmin=1e-5, vmax=I.max())
    im = ax.imshow(I, norm=norm,
              extent=[0, Lz, -Lx*1e6 / 2, Lx*1e6 / 2], #type: ignore
              aspect="auto", origin="lower", cmap="inferno")
    
    show = False
    if show:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.08)
        cbar.set_label("Intensity")
        
        ax.set(xlabel="z [m]", ylabel=r"x $[\mu m]$",
               title=rf"Kinoform (E={E/1000} keV, f={f} m, R={R*1e6} $\mu m$) intensity")
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
    out = savedir / "kinoform_sideview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved side-view to {out}")
    
    
def plot_aberrated_intensity(lens_types: type[Kinoform] | type[FZP] | list[type[Kinoform] | type[FZP]],
               heights: float | list[float]):
    from ..optics.metrics.lens_comparison import collect_metrics, print_comparison

    if not isinstance(lens_types, list):
        lens_types = [lens_types]
    if not isinstance(heights, list):
        heights = [heights] * len(lens_types) # type:ignore

    Lx, Lz = 3e-4, 10000
    N = 200000

    E = 8.e3
    f = 0.1
    R = 1e-4
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)


    metrics_list = []
    focal_waves  = []

    for lens_type, height in zip(lens_types, heights): # type:ignore
        err = build_errors(lens_type, height)
        label = f"Aberrated {lens_type.__name__}"

        simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
        propagator = AngularSpectrum(simulation)
        source = ConstantBeam(energy=E, simulation=simulation, z=0)

        if lens_type is Kinoform:
            lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                        simulation=simulation, z=0)
        else:
            lens = FZP(wavelength=source.wavelength, f=f, R=R, n=n,
                simulation=simulation, z=0)

        for error_func, kwargs in err:
            lens.add_error(error_func, **kwargs)

        h, outer_width = lens.zone_heights[-1], lens.zone_widths[-1]
        print(f"[{lens_type.__name__}] Outer zone width: {outer_width}")
        print(f"[{lens_type.__name__}] Outer zone aspect ratio: {h/outer_width}")

        lens.init_transmittance(source)
        source.filter(lens)
        P_in = total_power(source)
        lens.transform(source)
        source.propagate(lens.f, propagator)

        metrics_list.append(collect_metrics(P_in, source, lens, label))
        focal_waves.append((label, source, lens))

    print_comparison(metrics_list)

    scale = R / 2
    cmap_cycle = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, ((label, wave, lens), m) in enumerate(zip(focal_waves, metrics_list)):
        x = np.asarray(wave.grid)
        I = np.asarray(wave.intensity())
        ax.plot(x * 1e6, I, color=cmap_cycle(i), label=label)
        fwhm = m.get("fwhm")
        if fwhm is not None:
            scale = fwhm
            ax.axhline(0.5 * I.max(), color=cmap_cycle(i), lw=0.8, ls=":",
                       alpha=0.6)
            ax.axvspan(-fwhm / 2 * 1e6, fwhm / 2 * 1e6,
                       color=cmap_cycle(i), alpha=0.1)
            
    ax.set(xlabel=r"x $[\mu m]$", ylabel="Intensity",
           title=rf"Focal intensity (E={E/1000} keV, f={f} m, R={R*1e6:.0f} $\mu$m)")
    ax.set(xlim=(-10*scale*1e6, 10*scale*1e6))
    ax.legend(fontsize=8)
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / f"aberrated_lens_focal_slice.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved focal slice plot to {out}")
    
    
# Lens Profiles

def lens_profile_plot():
    savedir.mkdir(parents=True, exist_ok=True)

    E, f, R = 8e3, 0.1, 1e-4
    E, f, R = 8e3, 1.0, 5e-5
    Lx, N = 3e-4, 2000
    Lz = 2*f
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
    source = GaussianBeam(energy=E, simulation=sim, z=0, w0=R)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=f/2)
    
    plot_full_lens_profile(lens)
    
    
def lens_error_profile(lens_type: type[Kinoform] | type[FZP], err=None, ax=None,
                       lf=None, lR=None, title=None, show=False, savepath=None):

    _lf = lf if lf is not None else f
    _lR = lR if lR is not None else R

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    if lens_type is Kinoform:
        lens = Kinoform(wavelength=source.wavelength, f=_lf, R=_lR, n=n,
                        simulation=simulation, z=0)
    else:
        lens = FZP(wavelength=source.wavelength, f=_lf, R=_lR, n=n,
                   simulation=simulation, z=0)

    height, outer_width = lens.height, lens.zone_widths[-1]
    print("Outer zone width:", outer_width)
    print("Outer zone aspect ratio:", height/outer_width)

    labels = {
        "xlabel": r"x [$\mu$m]",
        "ylabel": r"thickness [$\mu$m]",
        "x_scale_factor": 1e6,
        "y_scale_factor": 1e6,
        "title": title
    }
    locations = lens.zone_locations
    ax = visualize_error(lens, err, ax=ax, R_min=locations[-5], labels=labels, show=show, savepath=savepath)

    height, outer_width = lens.zone_heights[-1], lens.zone_widths[-1]
    print("Outer zone width:", outer_width)
    print("Outer zone aspect ratio:", height/outer_width)

    return ax
    
# Error sweeps
    
def plot_sweep_intensity(sweep_name: str, error_func, sweep_param: str,
         err_values, error_kwargs: dict | None = None,
         xlabel: str = "Error", x_scale_factor: float = 1.0,
         xscale: str = "linear", invert_x: bool = False,
         n_slices: int = 8):
    '''
    Sweep `error_func` with parameter `sweep_param` over both lens configs and
    plot a grid of intensity slices — one subplot per lens config — with each
    slice coloured by error magnitude and the FWHM span highlighted.
    '''
    from ..optics.metrics.metrics import FWHM, total_power
    from ..optics.metrics.error_metrics import _safe_call

    Lx, Lz = 3e-4, 10000
    N = 100000
    E, f, R = 8.e3, 0.1, 1e-4
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    error_kwargs = dict(error_kwargs or {})
    err_values = np.asarray(list(err_values), dtype=float)
    print(err_values)
    
    lens_configs = [
        (Kinoform, {"f": 0.5, "R": 5e-5}, "Low-Resolution",  "tab:orange"),
        (Kinoform, {"f": 0.5, "R": 1e-4}, "High-Resolution", "tab:blue"),
    ]

    indices = np.round(np.linspace(0, len(err_values) - 1, n_slices)).astype(int)
    indices = np.unique(indices) #type: ignore
    # print(indices)
    cmap = plt.get_cmap("plasma")
    scaled = err_values[indices] * x_scale_factor
    norm_c = plt.Normalize(vmin=scaled.min(), vmax=scaled.max()) #type: ignore


    n_configs = len(lens_configs)
    fig, axes = plt.subplots(1, n_configs,
                             figsize=(6 * n_configs, 4),
                             constrained_layout=True)
    if n_configs == 1:
        axes = [axes]

    for ax, (lens_cls, params, name, _) in zip(axes, lens_configs):
        lf = params.get("f", f)
        lR = params.get("R", R)

        sim_ref = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
        src_ref = ConstantBeam(energy=E, simulation=sim_ref, z=0)
        lens_ref = lens_cls(wavelength=src_ref.wavelength, f=lf, R=lR,
                            n=n, simulation=sim_ref, z=0)
        src_ref.filter(lens_ref)
        lens_ref.init_transmittance(src_ref)
        lens_ref.transform(src_ref)
        src_ref.propagate(lens_ref.f, AngularSpectrum(sim_ref))

        ref_fwhm = _safe_call(FWHM, src_ref)
        x_ref = np.asarray(src_ref.grid)
        I_ref = np.asarray(src_ref.intensity())
        ax.plot(x_ref * 1e9, I_ref / I_ref.max(), color="black",
                lw=1.6, ls="--", label="Reference", zorder=10)
        if np.isfinite(ref_fwhm):
            ax.axvspan(-ref_fwhm / 2 * 1e9, ref_fwhm / 2 * 1e9,
                       color="black", alpha=0.07, zorder=0)

        scale = ref_fwhm if np.isfinite(ref_fwhm) else lR

        for idx in indices:
            e = err_values[idx]
            color = cmap(norm_c(e * x_scale_factor))

            sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
            src = ConstantBeam(energy=E, simulation=sim, z=0)
            lens = lens_cls(wavelength=src.wavelength, f=lf, R=lR,
                            n=n, simulation=sim, z=0)
            src.filter(lens)
            lens.add_error(error_func, **{sweep_param: float(e)}, **error_kwargs)
            lens.init_transmittance(src)
            lens.transform(src)
            src.propagate(lens.f, AngularSpectrum(sim))

            fwhm = _safe_call(FWHM, src)
            x = np.asarray(src.grid)
            I = np.asarray(src.intensity())
            I_norm = I / I_ref.max()

            label_val = e * x_scale_factor
            ax.plot(x * 1e9, I_norm, color=color, lw=1.1,
                    label=f"{label_val:.3g}", alpha=0.85)

            if np.isfinite(fwhm):
                ax.axhline(0.5 * I.max() / I_ref.max(), color=color,
                           lw=0.7, ls=":", alpha=0.5)
                ax.axvspan(-fwhm / 2 * 1e9, fwhm / 2 * 1e9,
                           color=color, alpha=0.08)

            print(f"  [{name}] {sweep_param}={e:.3e}  FWHM={fwhm:.3e}")

        xlim = 10 * scale * 1e9
        ax.set(xlabel=r"x [nm]", ylabel="Normalised Intensity",
               title=name,
               xlim=(-xlim, xlim))
        ax.set(yscale="linear")
        ax.legend(fontsize=7, title=xlabel, title_fontsize=7,
                  loc="upper right", ncol=2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=xlabel, fraction=0.046, pad=0.04)

    fig.suptitle(rf"{sweep_name}: intensity slices (E={E/1000} keV)")
    savedir.mkdir(parents=True, exist_ok=True)
    out = savedir / f"test_{sweep_name.lower().replace(' ', '_')}_slices.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved intensity-slice plot to {out}")
    
def plot_error_sweep(sweep_name: str, error_func, sweep_param: str,
                             err_values, error_kwargs: dict | None = None,
                             xlabel: str = "Error", x_scale_factor: float = 1.0,
                             xscale: str = "linear", invert_x=False, show_profile=False):
    '''
    Internal helper: sweeps `error_func` with parameter `sweep_param` over both
    FZP and Kinoform, recording focal_efficiency and FWHM, then produces a
    split-FWHM overlay figure.
    '''
    from ..optics.metrics.metrics import focal_efficiency, FWHM, total_power
    from ..optics.metrics.error_metrics import _bind_metric, _safe_call

    Lx, Lz = 3e-4, 10000
    N = 100000
    E, f, R = 8.e3, 1.0, 1e-4
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    error_kwargs = dict(error_kwargs or {})
    err_values = np.asarray(list(err_values), dtype=float)

    lens_configs = [
        (Kinoform, {"f": 0.1, "R": 1e-4}, "High-Resolution", "tab:blue"),
        (Kinoform, {"f": 1.0, "R": 5e-5}, "Low-Resolution", "tab:orange"),
    ]

    eff_data  = {}
    fwhm_data = {}
    ref_eff   = {}
    ref_fwhm  = {}

    for lens_cls, params, name, _ in lens_configs:
        sim_ref = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
        src_ref = ConstantBeam(energy=E, simulation=sim_ref, z=0)
        lens_ref = lens_cls(wavelength=src_ref.wavelength, f=params.get("f", f), R=params.get("R", R), 
                            n=n, simulation=sim_ref, z=0)
        src_ref.filter(lens_ref)
        P_in_ref = total_power(src_ref)
        lens_ref.init_transmittance(src_ref)
        lens_ref.transform(src_ref)
        src_ref.propagate(lens_ref.f, AngularSpectrum(sim_ref))

        radius_ref = 1.22 * src_ref.wavelength * lens_ref.f / (2 * lens_ref.R)
        ref_eff[name]  = _safe_call(
            _bind_metric(focal_efficiency, ref_source=src_ref,
                         P_in=P_in_ref, radius=radius_ref), src_ref)
        ref_fwhm[name] = _safe_call(FWHM, src_ref)
        print(f"[{name}] ref eff={ref_eff[name]:.3e}  ref FWHM={ref_fwhm[name]:.3e}")

        effs  = np.zeros(len(err_values))
        fwhms = np.zeros(len(err_values))
        for i, e in enumerate(err_values):
            sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
            src = ConstantBeam(energy=E, simulation=sim, z=0)
            lens = lens_cls(wavelength=src.wavelength, f=params.get("f", f), R=params.get("R", R), n=n,
                            simulation=sim, z=0)
            src.filter(lens)
            P_in = total_power(src)
            lens.add_error(error_func, **{sweep_param: float(e)}, **error_kwargs)
            lens.init_transmittance(src)
            lens.transform(src)
            src.propagate(lens.f, AngularSpectrum(sim))

            radius = 1.22 * src.wavelength * lens.f / (2 * lens.R)
            effs[i]  = _safe_call(
                _bind_metric(focal_efficiency, ref_source=src_ref,
                             P_in=P_in, radius=radius), src)
            fwhms[i] = _safe_call(FWHM, src)
            print(f"  [{name}] [{i+1}/{len(err_values)}] {sweep_param}={e:.3e}"
                  f"  eff={effs[i]:.3e}  FWHM={fwhms[i]:.3e}")

        eff_data[name]  = effs
        fwhm_data[name] = fwhms

    savedir.mkdir(parents=True, exist_ok=True)
    err_plot = err_values * x_scale_factor

    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    if show_profile:
        gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        ax_prof  = fig.add_subplot(gs[:, 0])
        ax_eff   = fig.add_subplot(gs[:, 1])
        ax_fwhm0 = fig.add_subplot(gs[0, 2])
        ax_fwhm1 = fig.add_subplot(gs[1, 2], sharex=ax_fwhm0)
        fwhm_axes = [ax_fwhm0, ax_fwhm1]

        mid_e = float(err_values[len(err_values) // 2])
        mid_err = [(error_func, {sweep_param: mid_e, **error_kwargs})]
        prof_cls, prof_params, _, _ = lens_configs[0]
        ax_prof = lens_error_profile(prof_cls, err=mid_err, ax=ax_prof,
                           lf=prof_params.get("f", f), lR=prof_params.get("R", R))
        ax_prof.set_title(f"Sample Profile: {sweep_name}",
                          fontsize=8)
    else:
        gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_eff   = fig.add_subplot(gs[:, 0])
        ax_fwhm0 = fig.add_subplot(gs[0, 1])
        ax_fwhm1 = fig.add_subplot(gs[1, 1], sharex=ax_fwhm0)
        fwhm_axes = [ax_fwhm0, ax_fwhm1]
        
        mid_e = float(err_values[len(err_values) // 2])
        mid_err = [(error_func, {sweep_param: mid_e, **error_kwargs})]
        prof_cls, prof_params, _, _ = lens_configs[0]
        
        prof_fig, prof_ax = plt.subplots(figsize=(5, 2.5))
        ax_prof = lens_error_profile(prof_cls, err=mid_err, ax=prof_ax,
                           lf=prof_params.get("f", f), lR=prof_params.get("R", R), title=sweep_name, show=True,
                           savepath=savedir / f"{sweep_name}_profile.png"
                           )
        ax_prof.set_title(f"Sample Profile: {sweep_name}",
                          fontsize=8)

    for (lens_cls, _, name, color), ax_f in zip(lens_configs, fwhm_axes):
        ax_eff.plot(err_plot, eff_data[name], color=color, label=name,
                    linewidth=1.8, zorder=2)
        if np.isfinite(ref_eff[name]):
            ax_eff.axhline(ref_eff[name], color=color, lw=0.9, ls="--", alpha=0.5)

        ax_f.plot(err_plot, fwhm_data[name] * 1e9, color=color, label=name,
                  linewidth=1.8, zorder=2)
        if np.isfinite(ref_fwhm[name]):
            ax_f.axhline(ref_fwhm[name] * 1e9, color=color, lw=0.9, ls="--", alpha=0.5)
            
        ymean = np.mean(ref_fwhm[name])*1e9
        ax_f.set(ylabel=f"FWHM [nm]", xscale=xscale, ylim=(0.5*ymean, 1.5*ymean))
        ax_f.legend(fontsize=8)

    ax_eff.set(xlabel=xlabel, ylabel="Focal Efficiency", xscale=xscale)
    ax_eff.legend(fontsize=8)
    ax_eff.xaxis.set_inverted(invert_x)
    
    ax_fwhm1.set(xlabel=xlabel)
    ax_fwhm1.xaxis.set_inverted(invert_x)
    plt.setp(ax_fwhm0.get_xticklabels(), visible=False)
    ax_fwhm0.xaxis.set_inverted(invert_x)

    design_names = " vs ".join(cfg[2] for cfg in lens_configs)
    fig.suptitle(rf"{sweep_name}: {design_names} (E={E/1000} keV)")

    out = savedir / f"{design_names.lower().replace(' ', '_')}_{sweep_name.lower().replace(' ', '_')}_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved error metrics plot to {out}")
    
    
    
# def lens_error_plot(sweep_name: str, error_func, sweep_param: str,
#                     err_values, error_kwargs: dict | None = None,
#                     xlabel: str = "Error", x_scale_factor: float = 1.0,
#                     xscale: str = "linear", invert_x: bool = False,
#                     n_slices: int = 8):
#     '''
#     Combined figure: intensity slices (top row, one panel per lens config) and
#     error-metric sweeps — focal efficiency (bottom-left) and per-config FWHM
#     (bottom-right, stacked) — all sharing the same sweep data so simulations
#     are only run once.
#     '''
#     from ..optics.metrics.metrics import focal_efficiency, FWHM, total_power
#     from ..optics.metrics.error_metrics import _bind_metric, _safe_call

#     Lx, Lz = 3e-4, 10000
#     N = 100000
#     E, f, R = 8.e3, 0.1, 1e-4
#     n = xrl.Refractive_Index("Si", E / 1000, 2.329)
#     error_kwargs = dict(error_kwargs or {})
#     err_values = np.asarray(list(err_values), dtype=float)

#     lens_configs = [
#         (Kinoform, {"f": 1.0, "R": 5e-5}, "Low-Resolution",  "tab:orange"),
#         (Kinoform, {"f": 0.1, "R": 1e-4}, "High-Resolution", "tab:blue"),
#     ]
#     n_configs = len(lens_configs)

#     slice_indices = np.unique(
#         np.round(np.linspace(0, len(err_values) - 1, n_slices)).astype(int)
#     )
#     cmap = plt.get_cmap("plasma")
#     scaled_slice = err_values[slice_indices] * x_scale_factor
#     norm_c = plt.Normalize(vmin=scaled_slice.min(), vmax=scaled_slice.max())

#     err_plot = err_values * x_scale_factor

#     n_bottom = 1 + n_configs
#     fig = plt.figure(figsize=(5 * n_bottom, 4 + 3), constrained_layout=True)
#     gs = fig.add_gridspec(2, n_bottom,
#                           height_ratios=[4, 3])

#     slice_axes = [fig.add_subplot(gs[0, c]) for c in range(n_configs)]
#     if n_configs < n_bottom:
#         for c in range(n_configs, n_bottom):
#             fig.add_subplot(gs[0, c]).set_visible(False)

#     ax_eff = fig.add_subplot(gs[1, 0])
#     fwhm_axes = [fig.add_subplot(gs[1, 1 + c]) for c in range(n_configs)]
#     for c in range(1, n_configs):
#         fwhm_axes[c].sharey(fwhm_axes[0])

#     eff_data  = {}
#     fwhm_data = {}
#     ref_eff   = {}
#     ref_fwhm  = {}

#     for col, (lens_cls, params, name, cfg_color) in enumerate(lens_configs):
#         lf = params.get("f", f)
#         lR = params.get("R", R)
#         ax_sl = slice_axes[col]

#         sim_ref = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
#         src_ref = ConstantBeam(energy=E, simulation=sim_ref, z=0)
#         lens_ref = lens_cls(wavelength=src_ref.wavelength, f=lf, R=lR,
#                             n=n, simulation=sim_ref, z=0)
#         src_ref.filter(lens_ref)
#         P_in_ref = total_power(src_ref)
#         lens_ref.init_transmittance(src_ref)
#         lens_ref.transform(src_ref)
#         src_ref.propagate(lens_ref.f, AngularSpectrum(sim_ref))

#         radius_ref = 1.22 * src_ref.wavelength * lens_ref.f / (2 * lens_ref.R)
#         ref_eff[name]  = _safe_call(
#             _bind_metric(focal_efficiency, ref_source=src_ref,
#                          P_in=P_in_ref, radius=radius_ref), src_ref)
#         ref_fwhm[name] = _safe_call(FWHM, src_ref)
#         print(f"[{name}] ref eff={ref_eff[name]:.3e}  ref FWHM={ref_fwhm[name]:.3e}")

#         x_ref = np.asarray(src_ref.grid)
#         I_ref = np.asarray(src_ref.intensity())
#         ax_sl.plot(x_ref * 1e9, I_ref / I_ref.max(), color="black",
#                    lw=1.6, ls="--", label="Reference", zorder=10)
#         if np.isfinite(ref_fwhm[name]):
#             ax_sl.axvspan(-ref_fwhm[name] / 2 * 1e9, ref_fwhm[name] / 2 * 1e9,
#                           color="black", alpha=0.07, zorder=0)

#         scale = ref_fwhm[name] if np.isfinite(ref_fwhm[name]) else lR

#         effs  = np.zeros(len(err_values))
#         fwhms = np.zeros(len(err_values))

#         for i, e in enumerate(err_values):
#             sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
#             src = ConstantBeam(energy=E, simulation=sim, z=0)
#             lens = lens_cls(wavelength=src.wavelength, f=lf, R=lR,
#                             n=n, simulation=sim, z=0)
#             src.filter(lens)
#             P_in = total_power(src)
#             lens.add_error(error_func, **{sweep_param: float(e)}, **error_kwargs)
#             lens.init_transmittance(src)
#             lens.transform(src)
#             src.propagate(lens.f, AngularSpectrum(sim))

#             radius = 1.22 * src.wavelength * lens.f / (2 * lens.R)
#             effs[i]  = _safe_call(
#                 _bind_metric(focal_efficiency, ref_source=src_ref,
#                              P_in=P_in, radius=radius), src)
#             fwhms[i] = _safe_call(FWHM, src)

#             if i in slice_indices:
#                 color = cmap(norm_c(e * x_scale_factor))
#                 x = np.asarray(src.grid)
#                 I = np.asarray(src.intensity())
#                 I_norm = I / I_ref.max()
#                 ax_sl.plot(x * 1e9, I_norm, color=color, lw=1.1,
#                            label=f"{e * x_scale_factor:.3g}", alpha=0.85)
#                 if np.isfinite(fwhms[i]):
#                     ax_sl.axhline(0.5 * I.max() / I_ref.max(), color=color,
#                                   lw=0.7, ls=":", alpha=0.5)
#                     ax_sl.axvspan(-fwhms[i] / 2 * 1e9, fwhms[i] / 2 * 1e9,
#                                   color=color, alpha=0.08)

#             print(f"  [{name}] [{i+1}/{len(err_values)}] {sweep_param}={e:.3e}"
#                   f"  eff={effs[i]:.3e}  FWHM={fwhms[i]:.3e}")

#         eff_data[name]  = effs
#         fwhm_data[name] = fwhms

#         xlim = 10 * scale * 1e9
#         ax_sl.set(xlabel="x [nm]", ylabel="Normalised Intensity",
#                   title=name, xlim=(-xlim, xlim))
#         ax_sl.legend(fontsize=7, title=xlabel, title_fontsize=7,
#                      loc="upper right", ncol=2)
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
#         sm.set_array([])
#         fig.colorbar(sm, ax=ax_sl, label=xlabel, fraction=0.046, pad=0.04)

#     for (_, _, name, color), ax_f in zip(lens_configs, fwhm_axes):
#         ax_eff.plot(err_plot, eff_data[name], color=color, label=name,
#                     linewidth=1.8, zorder=2)
#         if np.isfinite(ref_eff[name]):
#             ax_eff.axhline(ref_eff[name], color=color, lw=0.9, ls="--", alpha=0.5)

#         ax_f.plot(err_plot, fwhm_data[name] * 1e9, color=color, label=name,
#                   linewidth=1.8, zorder=2)
#         if np.isfinite(ref_fwhm[name]):
#             ax_f.axhline(ref_fwhm[name] * 1e9, color=color, lw=0.9, ls="--", alpha=0.5)
#         ymean = float(ref_fwhm[name]) * 1e9
#         ax_f.set(ylabel="FWHM [nm]", xscale=xscale,
#                  ylim=(0.5 * ymean, 1.5 * ymean), xlabel=xlabel)
#         ax_f.xaxis.set_inverted(invert_x)
#         ax_f.legend(fontsize=8)

#     ax_eff.set(xlabel=xlabel, ylabel="Focal Efficiency", xscale=xscale)
#     ax_eff.xaxis.set_inverted(invert_x)
#     ax_eff.legend(fontsize=8)

#     for ax_f in fwhm_axes[1:]:
#         ax_f.xaxis.set_inverted(invert_x)

#     fig.suptitle(rf"{sweep_name}: {' vs '.join(c[2] for c in lens_configs)} (E={E/1000} keV)")
#     savedir.mkdir(parents=True, exist_ok=True)
#     slug = sweep_name.lower().replace(" ", "_")
#     out = savedir / f"lens_error_plot_{slug}.png"
#     fig.savefig(out)
#     plt.close(fig)
#     print(f"Saved combined lens error plot to {out}")


if __name__ == "__main__":
    # lens_error_profile(Kinoform)
    # plot_error_sweep(
    #     "Manufacturing Height",
    #     LensErrors.kinoform_height_shrink, "height",
    #     err_values=np.linspace(1.0, 0.01, 50),
    #     error_kwargs={"proportion": True},
    #     xlabel="Height Proportion",
    #     x_scale_factor=1.0,
    #     invert_x=True,
    #     show_profile=False
    # )
    
    # plot_error_sweep(
    #     "Roughness",
    #     LensErrors.random_etch, "max_err",
    #     err_values=np.linspace(1e-9, 5e-7, 50),
    #     error_kwargs={"interval": 1, "seed": seed, "distribution": "gaussian"},
    #     x_scale_factor=1e9,
    #     xlabel=r"Maximum Etch Depth [nm]",
    #     show_profile=False
    # )
    
    # plot_error_sweep(
    #     "Sidewall Taper",
    #     LensErrors.kinoform_sidewall_taper, "err",
    #     err_values=np.linspace(0, 1e-7, 50),
    #     error_kwargs={"proportion": 1.0},
    #     xlabel=r"Taper [nm]",
    #     x_scale_factor=1e9,
    #     show_profile=False
    # )
    
    
    plot_error_sweep(
        "Zone Shrink",
        LensErrors.kinoform_zone_shrink, "err",
        err_values=np.linspace(0, 1e-7, 50),
        error_kwargs={"direction": "out"},
        xlabel=r"Shrink [nm]",
        x_scale_factor=1e9,
        show_profile=False
    )
    
    
    # lens_error_plot(
    #     "Sidewall Taper",
    #     LensErrors.kinoform_sidewall_taper, "err",
    #     err_values=np.linspace(0, 1e-7, 50),
    #     error_kwargs={"proportion": 1.0},
    #     xlabel=r"Taper [nm]",
    #     x_scale_factor=1e9,
    # )