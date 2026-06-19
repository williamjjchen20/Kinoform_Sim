import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xraylib as xrl
import os, functools, inspect, copy
from pathlib import Path
import argparse

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
                            err_values, error_kwargs=None, savepath=None):
    '''
    Sweep an etch error magnitude and record focal-plane intensity stats.

    args
    - source_factory: callable() -> fresh Waveform (with its own SimulationObject)
    - lens_factory:   callable(source) -> fresh ThinLens bound to that source
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
    I_max_ref = float(np.max(ref_source.intensity()))

    I_max = np.zeros_like(err_values)
    I_avg = np.zeros_like(err_values)
    P_foc = np.zeros_like(err_values)
    fwhm  = np.zeros_like(err_values)
    strehl = np.zeros_like(err_values)

    for i, e in enumerate(err_values):
        source = source_factory()
        lens = lens_factory(source)
        lens.add_error(error_func, **{mag_name: float(e)}, **error_kwargs)
        lens.init_transmittance(source)
        lens.transform(source)
        source.propagate(lens.f, propagator)

        Imax, Iavg = intensity_stats(source)
        I_max[i] = Imax
        I_avg[i] = Iavg
        P_foc[i] = total_power(source)
        try:
            fwhm[i] = FWHM(source)
        except Exception:
            fwhm[i] = np.nan
        strehl[i] = Imax / I_max_ref if I_max_ref > 0 else np.nan

        print(f"[{i+1}/{len(err_values)}] {mag_name}={e:.3e}  "
              f"I_max={Imax:.3e}  Strehl={strehl[i]:.3f}  "
              f"FWHM={fwhm[i]:.3e}  P_focal={P_foc[i]:.3e}")

    if savepath is not None:
        _plot_sweep(err_values, I_max, P_foc, fwhm, strehl, mag_name,
                    error_func.__name__, savepath)

    return {
        "err": err_values,
        "I_max": I_max,
        "I_avg": I_avg,
        "P_focal": P_foc,
        "fwhm": fwhm,
        "strehl": strehl,
    }


def _plot_sweep(err, I_max, P_foc, fwhm, strehl, mag_name, error_name, savepath):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{error_name}: focal-plane response vs {mag_name}")

    ax[0, 0].plot(err, I_max, marker="o", color="navy")
    ax[0, 0].set(xlabel=mag_name, ylabel="I_max", yscale="log")

    ax[0, 1].plot(err, strehl, marker="o", color="darkred")
    ax[0, 1].set(xlabel=mag_name, ylabel="Strehl (I_max / I_max_ref)")
    ax[0, 1].axhline(1.0, color="black", lw=0.5, ls="--")

    ax[1, 0].plot(err, fwhm, marker="o", color="darkgreen")
    ax[1, 0].set(xlabel=mag_name, ylabel="FWHM [m]")

    ax[1, 1].plot(err, P_foc, marker="o", color="purple")
    ax[1, 1].set(xlabel=mag_name, ylabel="Focal-plane power")

    fig.savefig(savepath)
    plt.close(fig)
    print(f"Saved sweep figure to {savepath}.")


def _demo():
    Lx, Lz, N = 1.5e-4, 10000, 2048
    E, f, R = 8.5e3, 1.0, 5e-5
    dim = 2
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    propagator = functools.partial(angular_spectrum_method, dim=dim)

    def source_factory():
        sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Lx, Ny=N)
        return ConstantBeam(energy=E, simulation=sim, z=0)

    def lens_factory(src):
        return Kinoform(wavelength=src.wavelength, f=f, R=R, n=n,
                        simulation=src.simulation, z=0)

    err_values = np.linspace(0, 4e-6, 9)
    out = savedir / "etch_error_vs_intensity_random.png"
    etch_error_vs_intensity(
        source_factory, lens_factory, propagator,
        LensErrors.random_etch, err_values,
        error_kwargs={"count": N**dim // 3, "seed": 42},
        savepath=out,
    )


if __name__ == "__main__":
    _demo()