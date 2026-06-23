import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, functools
import time
from pathlib import Path

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "test_figs/").resolve()


import xraylib as xrl
from ..optics import SimulationObject, Waveform, AngularSpectrum, ConstantBeam, Kinoform

def plot_sideview(simulation: SimulationObject, z_max, dz):
    propagator = simulation.propagator 
    source = simulation.objects["source"]
    lens = simulation.objects["lens"]
    
    Nz = int(z_max//dz)
    print("Nz:", Nz)
    print("Simulating ...")
    
    dim = simulation.dim
    view = np.zeros((simulation.Nx, Nz), dtype=np.complex128)
    
    lens_reached = False
    for i in range(Nz):
        source_z = np.round(source.z, 10) # nm accuracy
        lens_z = lens.z
        if i % 10 == 0: print(f"Source at z={source_z}", flush=True)
        if not lens_reached and source_z + dz >= lens_z:
            diff = lens_z - source_z
            source.propagate(diff, propagator.propagator)
            lens.transform(source)
            source.propagate(dz-diff, propagator.propagator)
            lens_reached = True
            print("Source has reached lens")
        else:
            source.propagate(dz, propagator.propagator)
        
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

if __name__ == "__main__":
    savedir.mkdir(parents=True, exist_ok=True)

    Lx, Lz, N = 1.5e-4, 1.5, 2048
    E, f, R = 8.5e3, 1.0, 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
    sim.add_propagator(AngularSpectrum(dim=2))

    source = ConstantBeam(energy=E, simulation=sim, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=0.1)

    dz = 0.01
    z_lim = f + lens.z
    assert (z_lim <= Lz)
    
    t_start = time.time()
    view = plot_sideview(sim, z_max=Lz, dz=dz)
    I = np.abs(view)**2
    t_end = time.time()
    print(f"Time Taken: {t_end-t_start} s")

    fig, ax = plt.subplots(figsize=(10, 4))

    norm = colors.LogNorm(vmin=1e-4, vmax=I.max())
    im = ax.imshow(I, norm=norm,
              extent=[0, Lz, -Lx*1e6 / 2, Lx*1e6 / 2], #type: ignore
              aspect="auto", origin="lower", cmap="inferno")
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.08)
    cbar.set_label("Intensity")
    
    ax.set(xlabel="z [m]", ylabel=r"x $[\mu m]$",
           title=rf"Kinoform (f={f} m, R={R*1e6} $\mu m$) intensity")
    out = savedir / "kinoform_sideview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved side-view to {out}")