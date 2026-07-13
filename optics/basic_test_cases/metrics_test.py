import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xraylib as xrl
import os, functools
from pathlib import Path

from ..propagators import *
from ..metrics.metrics import *
from ..classes import *


script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/metrics").resolve()
savedir.mkdir(parents=True, exist_ok=True)

def run_lens(lens_cls, label, simulation, propagator, E, f, R, n, w0=None, init_lens=True):
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
    if init_lens: lens.init_transmittance(source)

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
    I_max, I_avg = max_intensity(focal_wave), mean_intensity(focal_wave)
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


def plot_comparison(parabolic, kinoform, savepath):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    pf_wave, pf_lens = parabolic
    kf_wave, kf_lens = kinoform

    Lx, Ly = pf_wave.simulation.Lx, pf_wave.simulation.Ly
    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]

    # Lens phase
    ax[0, 0].imshow(pf_lens.angle(), cmap="twilight", extent=extent)
    ax[0, 0].set(title="Parabolic Lens Phase", xlabel="x [m]", ylabel="y [m]")
    ax[1, 0].imshow(kf_lens.angle(), cmap="twilight", extent=extent)
    ax[1, 0].set(title="Kinoform Phase", xlabel="x [m]", ylabel="y [m]")

    # Focal-plane intensity (log scale)
    I_pf = pf_wave.intensity()
    I_kf = kf_wave.intensity()
    vmax = max(I_pf.max(), I_kf.max())
    vmin = max(vmax * 1e-6, 1e-20)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    im1 = ax[0, 1].imshow(I_pf, norm=norm, cmap="inferno", extent=extent)
    ax[0, 1].set(title="Parabolic Focal Intensity", xlabel="x [m]", ylabel="y [m]")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[1, 1].imshow(I_kf, norm=norm, cmap="inferno", extent=extent)
    ax[1, 1].set(title="Kinoform Focal Intensity", xlabel="x [m]", ylabel="y [m]")
    fig.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    # Central line cuts
    Nx = I_pf.shape[1]
    x = np.linspace(-Lx/2, Lx/2, Nx)
    cy_pf = I_pf.shape[0] // 2
    cy_kf = I_kf.shape[0] // 2
    ax[0, 2].plot(x, I_pf[cy_pf, :], color="navy")
    ax[0, 2].set(title="Parabolic Central Cut", xlabel="x [m]", ylabel="Intensity", yscale="log")
    ax[1, 2].plot(x, I_kf[cy_kf, :], color="darkred")
    ax[1, 2].set(title="Kinoform Central Cut", xlabel="x [m]", ylabel="Intensity", yscale="log")

    plt.savefig(savepath)
    plt.close(fig)


def test_compare_xray_lenses():
    print("Comparing X-ray Parabolic Lens vs. Kinoform...")

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

    # --- Parabolic ---
    sim_pf = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

    focal_pf, lens_pf, Pin_pf = run_lens(
        XrayParabolicLens, "Parabolic", sim_pf, AngularSpectrum(sim_pf), E, f, R, n
    )

    # --- Kinoform ---
    sim_kf = SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

    focal_kf, lens_kf, Pin_kf = run_lens(
        Kinoform, "Kinoform", sim_kf, AngularSpectrum(sim_kf), E, f, R, n
    )

    # Collect & print metrics
    m_pf = collect_metrics(focal_pf, Pin_pf, "Parabolic")
    m_kf = collect_metrics(focal_kf, Pin_kf, "Kinoform")
    print_comparison([m_pf, m_kf])

    # Strehl-like ratio: peak intensity ratio (kinoform / parabolic)
    print(f"\nPeak intensity ratio (Kinoform / Parabolic): {m_kf['I_max']/m_pf['I_max']:.4f}")
    print(f"FWHM ratio (Kinoform / Parabolic):           {m_kf['FWHM [m]']/m_pf['FWHM [m]']:.4f}")
    print(f"Efficiency ratio (Kinoform / Parabolic):     {m_kf['focusing_efficiency']/m_pf['focusing_efficiency']:.4f}")

    # Comparison figure
    out = os.path.join(savedir, "Metrics_Parabolic_vs_Kinoform.png")
    plot_comparison((focal_pf, lens_pf), (focal_kf, lens_kf), out)
    print(f"Saved comparison figure to {out}")

def test_FWHM():
    '''
    Validate the FWHM metric against closed-form values for:
      (1) 1D centered Gaussian beam       -> FWHM = w0 * sqrt(2*ln2)
      (2) 2D centered Gaussian beam       -> same
      (3) 2D off-center Gaussian beam     -> same; checks the 2D baseline fix
      (4) 2D Airy disk from a uniform-aperture parabolic lens at focus
                                          -> FWHM ~= 1.0289 * lambda*f / (2R)
    Per-case relative error is asserted within a tolerance loose enough to
    survive sample-spacing quantization but tight enough to catch regressions.
    '''
    print("Testing FWHM metric against theoretical values...")
    print("="*72)
    results = []
    tol = 5e-2   # 5% relative tolerance

    def record(name, measured, expected, atol=None):
        rel = abs(measured - expected) / expected if expected else np.nan
        ok = (rel <= tol) if atol is None else (abs(measured - expected) <= atol)
        print(f"  {name:<40}  measured={measured:.6e}  "
              f"expected={expected:.6e}  rel_err={rel:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
        results.append((name, measured, expected, rel, ok))
        return ok

    # --- (1) 1D centered Gaussian ---
    Lx, N, w0 = 1.0e-3, 1024, 5.0e-5
    sim1d = SimulationObject(Lx=Lx, Nx=N, Lz=1.0)
    g1d = GaussianBeam(energy=1.96, simulation=sim1d, z=0, w0=w0)
    expected_gauss = w0 * np.sqrt(2 * np.log(2))
    record("1D Gaussian (centered)", FWHM(g1d), expected_gauss)

    # --- (2) 2D centered Gaussian ---
    N2 = 1024
    sim2d = SimulationObject(Lx=Lx, Ly=Lx, Nx=N2, Ny=N2, Lz=1.0)
    g2d = GaussianBeam(energy=1.96, simulation=sim2d, z=0, w0=w0)
    record("2D Gaussian (centered)", FWHM(g2d), expected_gauss)

    # --- (3) 2D off-center Gaussian (checks baseline subtraction) ---
    g2d_off = GaussianBeam(energy=1.96, simulation=sim2d, z=0, w0=w0)
    X, Y = g2d_off.grid
    x0, y0 = 2.0e-4, -1.5e-4
    g2d_off.field = np.exp(-((X - x0)**2 + (Y - y0)**2) / w0**2)
    record("2D Gaussian (off-center)", FWHM(g2d_off), expected_gauss)

    # --- (4) 2D Airy disk via XrayParabolicLens at its focal plane ---
    Lx_a, N_a = 1.5e-4, 2048
    E, f, R = 8e3, 1., 5e-6
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)
    sim_airy = SimulationObject(Lx=Lx_a, Ly=Lx_a, Nx=N_a, Ny=N_a, Lz=10.0)
    propagator = AngularSpectrum(sim_airy)

    # sampling sanity
    wavelength = (const.h * const.c) / (E * const.e)
    airy_half_max = 1.029 * wavelength * f / (2 * R)
    dx = Lx_a / N_a
    print(f"  [Airy] lambda={wavelength:.3e} m, expected FWHM={airy_half_max:.3e} m, "
          f"dx={dx:.3e} m, samples_per_FWHM~{airy_half_max/dx:.2f}")

    src, lens, _ = run_lens(XrayParabolicLens, "Airy", sim_airy, propagator,
                            E, f, R, n, init_lens=True)
    # Tolerance: sample spacing alone limits accuracy to ~dx/expected
    airy_tol = 5e-2
    rel = abs(FWHM(src) - airy_half_max) / airy_half_max
    ok = rel <= airy_tol
    print(f"  {'2D Airy disk (parabolic lens)':<40}  measured={FWHM(src):.6e}  "
          f"expected={airy_half_max:.6e}  rel_err={rel:.3e}  "
          f"tol={airy_tol:.2e}  {'PASS' if ok else 'FAIL'}")
    results.append(("2D Airy disk", FWHM(src), airy_half_max, rel, ok))

    # --- diagnostic plot: line cuts with measured & expected half-max marked ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # 1D Gaussian cut
    x_g = g1d.grid
    I_g = g1d.intensity()
    ax[0].plot(x_g, I_g / I_g.max(), color="navy", label="|U|^2 / max")
    ax[0].axhline(0.5, color="gray", lw=0.5, ls="--")
    ax[0].axvspan(-expected_gauss/2, expected_gauss/2, color="crimson", alpha=0.15,
                  label=f"expected FWHM={expected_gauss:.2e}")
    ax[0].set(title="1D Gaussian", xlabel="x [m]", ylabel="I/I_max",
              xlim=(-3*w0, 3*w0))
    ax[0].legend(fontsize=8)

    # 2D Gaussian center row
    I2 = g2d.intensity()
    x2 = np.linspace(-Lx/2, Lx/2, N2)
    cy = I2.shape[0] // 2
    ax[1].plot(x2, I2[cy, :] / I2.max(), color="navy")
    ax[1].axhline(0.5, color="gray", lw=0.5, ls="--")
    ax[1].axvspan(-expected_gauss/2, expected_gauss/2, color="crimson", alpha=0.15)
    ax[1].set(title="2D Gaussian (center row)", xlabel="x [m]",
              xlim=(-3*w0, 3*w0))

    # Airy disk center row
    I_a = src.intensity()
    x_a = np.linspace(-Lx_a/2, Lx_a/2, N_a)
    cy_a = I_a.shape[0] // 2
    ax[2].plot(x_a, I_a[cy_a, :]/I_a.max(), color="navy")
    ax[2].axhline(0.5, color="gray", lw=0.5, ls="--")
    ax[2].axvspan(-airy_half_max/2, airy_half_max/2, color="crimson", alpha=0.15,
                  label=f"expected FWHM={airy_half_max:.2e}")
    ax[2].set(title="2D Airy disk (center row)", xlabel="x [m]",
              xlim=(-5*airy_half_max, 5*airy_half_max))
    ax[2].legend(fontsize=8)

    out = os.path.join(savedir, "FWHM_validation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved diagnostic plot to {out}")

    # summary
    n_pass = sum(1 for r in results if r[-1])
    print("="*72)
    print(f"FWHM test: {n_pass}/{len(results)} cases passed")
    assert n_pass == len(results), "FWHM test failures (see log above)"


if __name__ == "__main__":
    test_FWHM()
    # test_compare_xray_lenses()
