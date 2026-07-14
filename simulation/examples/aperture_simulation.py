import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, functools
import time
from pathlib import Path

import time

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/").resolve()

from ..setup import *

def save_snapshot(wave, filename, title=None):
    """
    Save a snapshot figure of the last simulation run's wavefield.

    Shows the 2D intensity heatmap and the central horizontal slice through
    the peak, with the FWHM region highlighted on the slice.
    """
    I = wave.intensity()
    fwhm = FWHM(wave)

    if wave.dim == 1:
        X = wave.grid * wave.Rx
        peak_idx = int(np.argmax(I))
        x_peak = X[peak_idx]
        slice_I = I
        slice_x = X
    else:
        Xg, Yg = wave.grid
        Xg, Yg = Xg * wave.Rx, Yg * wave.Ry
        peak_idx = np.unravel_index(np.argmax(I), I.shape)
        x_peak = Xg[peak_idx]
        slice_I = I[peak_idx[0], :]
        slice_x = Xg[peak_idx[0], :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if wave.dim == 2:
        extent = [Xg.min(), Xg.max(), Yg.min(), Yg.max()]
        im = axes[0].imshow(I, extent=extent, origin="lower", cmap="inferno")
        axes[0].axhline(Yg[peak_idx[0], 0], color="cyan", lw=0.8, ls="--") #type: ignore
        fig.colorbar(im, ax=axes[0], label="Intensity")
        axes[0].set(xlabel="x [m]", ylabel="y [m]", title="2D Intensity")
    else:
        axes[0].plot(X, I)
        axes[0].set(xlabel="x [m]", ylabel="Intensity", title="1D Intensity")

    axes[1].plot(slice_x, slice_I, color="k")
    axes[1].set(xlabel="x [m]", ylabel="Intensity",
                title=f"Central Slice (FWHM = {fwhm:.3e} m)")

    if np.isfinite(fwhm):
        x_left = x_peak - fwhm / 2
        x_right = x_peak + fwhm / 2
        I_half = np.max(slice_I) / 2
        axes[1].axhline(I_half, color="gray", ls=":", lw=0.8)
        axes[1].axvspan(x_left, x_right, color="orange", alpha=0.3,
                        label=f"FWHM = {fwhm:.3e} m")
        axes[1].axvline(x_left, color="orange", lw=1)
        axes[1].axvline(x_right, color="orange", lw=1)
        axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def main():
    N=1024
    Lx=Ly=7e-5
    E = 8e3
    n = 100
    
    aperture_width = 50e-6
    z_aperture = 0.0
    
    dz = 500e-3 # needs to sweep
    heights = np.array([0.25e-6])#np.linspace(0.1e-6, 50e-6, n)
    print(heights)
    fwhm = np.zeros_like(heights)
    
    start = time.time()
    for i, h in enumerate(heights):
        simulation = build_simulation(N=N, Lx=Lx, Ly=Ly, Lz=10)
        propagator = AngularSpectrum(simulation)
        source = ConstantBeam(energy=E, simulation=simulation, z=0)
            
        aperture = SingleSlit(simulation, z=z_aperture, width=aperture_width, height=h)
        
        
        # source.propagate(z_aperture, propagator)
        aperture.transform(source)
        source.propagate(dz, propagator)
        fwhm[i] = FWHM(source)
        
        if i == 0:
            save_snapshot(source, savedir / "aperture_snapshot.png",
                    title=f"Snapshot: h={heights[-1]:.3e} m, dz={dz} m")
            
        quit()
        
        if i % 10 == 0: print(f"Finished testing h={h} m...")
    end = time.time()
    print(f"Simulation took {np.abs(end-start):.3f} s")


    plt.figure()
    plt.plot(heights, fwhm)
    plt.title("Aperture Height vs. FWHM")
    plt.xlabel(r"Aperture Height $[\mu m]$")
    plt.ylabel(r"FWHM $[m]$")
    plt.savefig(savedir / "aperture.png")
    
if __name__ == "__main__":
    main()