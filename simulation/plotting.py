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

def plot_3D_Lens(lens):
    pass

# def test_focal_intensity_consistency():
#     """
#     Sanity check: peak intensity from plot_sideview at the focal column should
#     match a direct one-shot propagation over the same total distance.

#     Setup: source at z=0, lens at z=z_lens, focal spot at z = z_lens + f.
#     Direct path: propagate to lens -> transform -> propagate f to focal plane.
#     Sideview path: extract the column nearest z = z_lens + f from view.
#     """
#     print("\n=== test_focal_intensity_consistency ===")
#     Lx, Lz, N = 1.5e-4, 1.5, 2048
#     E, f, R = 8.0e3, 1.0, 5e-5
#     z_lens = 0.0
#     n = xrl.Refractive_Index("Si", E / 1000, 2.329)

#     dz = 0.01
    
#     sim_direct = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
#     source_direct = ConstantBeam(energy=E, simulation=sim_direct, z=0)
#     lens_direct = Kinoform(wavelength=source_direct.wavelength, f=f, R=R, n=n,
#                            simulation=sim_direct, z=z_lens)
#     lens_direct.init_transmittance(source_direct)
#     print(lens_direct.R)
#     propagator_direct = AngularSpectrum(simulation=sim_direct)
    
#     z_focal = z_lens + f
    
#     print("Running direct propagation...")
#     source_direct.propagate(z_lens, propagator_direct)
#     lens_direct.transform(source_direct)
#     source_direct.propagate(f, propagator_direct)
#     I_direct_peak = source_direct.intensity().max()
#     print(f"  Direct peak intensity at z={z_focal:.3f} m: {I_direct_peak:.6e}")
    

if __name__ == "__main__":
    savedir.mkdir(parents=True, exist_ok=True)

    # test_focal_intensity_consistency()
    # quit()
    E, f, R = 8.e3, 0.1, 1e-4
    Lx, N = 3e-4, 10000
    Lz = 2*f
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N, Ny=N)
    source = GaussianBeam(energy=E, simulation=sim, z=0, w0=R/2)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=0.05)
    # lens.init_transmittance(source)

    dz = 0.001
    z_lim = f + lens.z
    assert (z_lim <= Lz)
    
    t_start = time.time()
    view = simulation_sideview(sim, z_max=Lz, dz=dz)
    
    I = np.abs(view)**2
    t_end = time.time()
    print(f"Time Taken: {t_end-t_start} s")

    fig, ax = plt.subplots(figsize=(10, 4))
    print(I.max())
    norm = colors.LogNorm(vmin=1e-5, vmax=I.max())
    im = ax.imshow(I, norm=norm,
              extent=[0, Lz, -Lx*1e6 / 2, Lx*1e6 / 2], #type: ignore
              aspect="auto", origin="lower", cmap="inferno")
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.08)
    cbar.set_label("Intensity")
    
    ax.set(xlabel="z [m]", ylabel=r"x $[\mu m]$",
           title=rf"Kinoform (E={E/1000} keV, f={f} m, R={R*1e6} $\mu m$) intensity")
    out = savedir / "kinoform_sideview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved side-view to {out}")