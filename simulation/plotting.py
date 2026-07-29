import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import time
from pathlib import Path

import xraylib as xrl
from ..optics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "test_figs/").resolve()

def simulation_sideview(simulation: SimulationObject, z_max, dz):
    source = simulation.objects["source"]
    lens = simulation.objects["lens"]
    propagator = AngularSpectrum(simulation=simulation)
    Nz = int(z_max//dz)
    print("Nz:", Nz)
    print("Simulating ...")
    
    dim = simulation.dim
    view = np.zeros((simulation.Nx, Nz), dtype=np.complex128)
    
    lens_reached = False
    pending_dz = 0.0
    for i in range(Nz):
        source_z = np.round(source.z, 10) # nm accuracy
        lens_z = lens.z
        if i % 10 == 0: print(f"Source at z={source_z}", flush=True)
        if not lens_reached and source_z + dz >= lens_z:
            diff = lens_z - source_z
            source.propagate(diff, propagator)
            lens.transform(source)
            lens_reached = True
            pending_dz = dz - diff
            print("Source has reached lens")
        else:
            step = pending_dz + dz if pending_dz > 0 else dz
            source.propagate(step, propagator)
            pending_dz = 0.0
        
        if dim == 1:
            sl = source.field
        else:
            #y = 0 slice on meshgrid
            X, _ = source.grid
            cy = X.shape[0] // 2
            sl = source.field[cy,:] 
        view[:,i] = sl

    print("Simulation Complete.")
    return view

def visualize_error(lens, errors, ax=None, show=False, labels={},
                    R_min=None, R_max=None, savepath=None):
    '''
    Apply one or more errors to `lens` and plot the perturbed profile with the
    pristine (pre-error) profile overlaid as a faint dashed line.

    args
    - lens:    a built ThinLens (Kinoform / FZP / ...) whose `.profile` has been
               initialized. The lens is mutated in place by `add_error`.
    - errors:  list of (error_func, kwargs) tuples applied in order, mirroring
               `lens.add_error(error_func, **kwargs)` from `error_test.py`.
               A single (error_func, kwargs) tuple is also accepted.
    - ax:      optional matplotlib Axes to draw into.
    - savepath: optional path to save the figure.
    - title:   optional axis title.
    - R_min, R_max: optional x-axis limits in metres (default +/- lens.R).
    '''
    if isinstance(errors, tuple) and len(errors) == 2 and callable(errors[0]):
        errors = [errors]

    if lens.simulation.dim == 1:
        x = np.asarray(lens.grid)
    else:
        X, _ = lens.grid
        cy = X.shape[0] // 2
        x = np.asarray(X[cy, :])

    def _slice(profile):
        p = np.asarray(profile)
        return p if p.ndim == 1 else p[p.shape[0] // 2, :]

    original = _slice(lens.profile).copy()

    if errors is not None:
        for error_func, kwargs in errors:
            lens.add_error(error_func, **(kwargs or {}))

    perturbed = _slice(lens.profile)

    if R_min is None: R_min = -lens.R
    if R_max is None: R_max =  lens.R
    mask = (x >= R_min) & (x <= R_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        
    title=labels.get("title", None)
    x_scale_factor = labels.get("x_scale_factor", 1.0)
    y_scale_factor = labels.get("y_scale_factor", 1.0)
    xlabel = labels.get("xlabel", None)
    ylabel = labels.get("ylabel", None)
    

    ax.fill_between(x[mask]*x_scale_factor, 0, perturbed[mask]*y_scale_factor, color="steelblue", alpha=0.5)
    # ax.plot(x[mask], perturbed[mask], color="navy", lw=1, alpha=0.8, label="with error")
    ax.plot(x[mask]*x_scale_factor, original[mask]*y_scale_factor, color="black", lw=0.8,
            alpha=0.5, label="ideal")
    ax.axhline(0, color="black", lw=0.5)

    if show:
        ax.set(xlabel=xlabel, ylabel=ylabel,
            title=title)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
    ax.set_xlim(R_min*x_scale_factor, R_max*x_scale_factor)
    ax.set_ylim((0, None)) #type: ignore
    if savepath is not None:
        fig.savefig(savepath)
        print(f"Saved error-visualization figure to {savedir}.")

    return ax

def plot_full_lens_profile(lens):
    print("Plotting full lens profile...")

    x_sf = 1e6          # m -> µm on lateral axes
    z_sf = 1e6          # m -> nm on thickness axis

    labels = {
        "label":          type(lens).__name__,
        "xlabel":         r"x [$\mu$m]",
        "y_axis_label":   r"y [$\mu$m]",
        "x_scale_factor": x_sf,
        "y_scale_factor": z_sf,
        "ylabel": None,
        "title": None,
        "cmap":           "viridis",
        "elev":           30,
        "azim":           -55,
        "z_aspect":       0.5,
    }

    fig = plt.figure(figsize=(10, 3), layout="constrained")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.8])

    ax3d_placeholder = fig.add_subplot(gs[0, 0])
    ax2d             = fig.add_subplot(gs[0, 1])

    
    ## 3-D plot
    lens.plot_profile(ax=ax3d_placeholder, labels=labels, _3d=True)
    
    # 2-D copy
    lens_copy = lens.copy(dim=1)
    labels["ylabel"] = r"thickness [$\mu$m]"
    lens_copy.plot_profile(ax=ax2d,             labels=labels, _3d=False, R_min=-1.01*lens.R, R_max=1.01*lens.R)
    
    fig.suptitle(rf"{type(lens).__name__} profile (f={lens.f:.3g} m, R={lens.R*1e6:.3g} $\mu$m)",)

    out = savedir / "lens_profile.png"
    fig.savefig(out, dpi=150)
    print(f"Saved lens profile to {out}")
    

if __name__ == "__main__":
    savedir.mkdir(parents=True, exist_ok=True)

    E, f, R = 8e3, 0.1, 1e-4
    E, f, R = 8e3, 1.0, 5e-5
    Lx, N = 3e-4, 100000
    Lz = 2*f
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N, Ny=N)
    source = GaussianBeam(energy=E, simulation=sim, z=0, w0=R)
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