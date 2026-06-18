import numpy as np
import xraylib as xrl
import matplotlib.pyplot as plt
from pathlib import Path

from ..classes import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "../test_figs/error_test").resolve()
savedir.mkdir(parents=True, exist_ok=True)


def test_kinoform_random_etch(max_err):
    print("Testing Kinoform with random etch error (1D)...")
    Lx, Lz = 1.5e-4, 10000
    N = 5000

    E = 8.5e3
    f = 1.0
    R = 5e-5
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    simulation = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)

    source = ConstantBeam(energy=E, simulation=simulation, z=0)
    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=simulation, z=0)

    lens.add_error(LensErrors.random_etch, max_err=max_err, count=N//3, seed=42)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = lens.grid
    mask = np.abs(x) <= lens.R
    ax.fill_between(x[mask], 0, lens.profile[mask], color="steelblue", alpha=0.6)
    ax.plot(x[mask], lens.profile[mask], color="navy", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="x [m]", ylabel="thickness [m]",
           title=f"Etched Kinoform profile (max_err={max_err:.1e} m)")
    ax.set_xlim(-lens.R, lens.R)

    out = savedir / "Kinoform_random_etch_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved etched kinoform profile to {out}.")


if __name__ == "__main__":
    max_err = 4e-7
    test_kinoform_random_etch(max_err)
