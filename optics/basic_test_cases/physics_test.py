import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import xraylib as xrl
import os, sys
from pathlib import Path

from ..propagators import AngularSpectrum
from ..classes import *
from ..metrics import *

F = WaveFunctions()

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/physics_test").resolve()
savedir.mkdir(parents=True, exist_ok=True)


def _report(label, wave, P_in):
    P = total_power(wave)
    eff = P / P_in if P_in > 0 else np.nan
    print(f"{label:<30} P={P:.6e}   P/P_in={eff:.6f}")
    return P, eff


def test_power_conservation_freespace():
    print("=" * 60)
    print("Free-space propagation power conservation (no lens)")
    print("=" * 60)

    Lx, Lz, N = 1.5e-4, 10000, 2048
    E = 8.5e3
    f = 1.0

    sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
    propagator = AngularSpectrum(sim)

    source = ConstantBeam(energy=E, simulation=sim, z=0)
    P_in = total_power(source)
    _report("Initial (z=0)", source, P_in)

    source.propagate(f / 2, propagator)
    _report(f"After z = f/2 = {f/2} m", source, P_in)

    source.propagate(f / 2, propagator)
    _report(f"After z = f   = {f} m", source, P_in)


def test_power_conservation_kinoform():
    print("=" * 60)
    print("Kinoform lens power flow (incident -> post-lens -> focal)")
    print("=" * 60)

    Lx, Lz, N = 1.5e-4, 10000, 2048
    E = 8.5e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Ly=Lx, Lz=Lz, Nx=N, Ny=N)
    propagator = AngularSpectrum(sim)

    source = ConstantBeam(energy=E, simulation=sim, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=0)
    
    # mask out incident x-rays into aperture_field
    source.field *= lens.aperture_field
    
    P_in = total_power(source)
    _report("Incident (pre-lens)", source, P_in)


    lens.init_transmittance(source)
    lens.transform(source)
    _report("Post-lens (z=0+)", source, P_in)

    source.propagate(f / 2, propagator)
    _report(f"After z = f/2 = {f/2} m", source, P_in)

    source.propagate(f / 2, propagator)
    _report(f"At focal plane (z=f)", source, P_in)


if __name__ == "__main__":
    test_power_conservation_freespace()
    test_power_conservation_kinoform()
