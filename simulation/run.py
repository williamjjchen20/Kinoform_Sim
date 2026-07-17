import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import xraylib
import functools, sys, os
import argparse
import json

from .setup import *
from .plotting import *

_CURR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(prog="Optics Simulation", description="")
parser.add_argument('-N', type=int, nargs='?', default=1024, 
                    help="Numerical resolution")
parser.add_argument('-Lx', type=float, nargs='?', required=True, 
                    help="Box size")
parser.add_argument('-Lz', type=float, nargs='?', required=True, 
                    help="Box length")
parser.add_argument('-Ly', type=float, nargs="?", default=None, 
                    help="Optional box length for higher dimension")
parser.add_argument("--params", type=str,
                    help="pre-saved parameters")
parser.add_argument("--save", type=bool, default=False, help="flag to save parameters")


def _prompt(label: str, cast, saved_val):
    # if saved_val is not None: return saved_val
    hint = f" [{saved_val}]"
    raw = input(f"  > {label}{hint}: ").strip()
    return cast(raw)

def __initialize_source(simulation: SimulationObject, saved: dict) -> tuple[Waveform | None, dict]:
    BAR  = "#" * 60
    RULE = "-" * 60
    print(BAR)
    print("  Source Setup")
    print(BAR)
    print("  Available types: Constant, Gaussian  (blank to skip)")

    src_type = _prompt("Source type", str, saved.get("type"))

    if not src_type:
        print("  No source.")
        print(RULE)
        return None, {}

    E  = _prompt("Energy [eV]", float, saved.get("energy"))
    U0 = _prompt("Amplitude U0", float, saved.get("U0"))
    
    assert E is not None and U0 is not None
    params: dict = {"type": src_type, "energy": E, "U0": U0}

    match src_type:
        case "Constant":
            source = ConstantBeam(energy=E, simulation=simulation, z=0, U0=U0)
        case "Gaussian":
            w0 = _prompt("Beam waist w0 [m]", float, saved.get("w0"))
            params["w0"] = w0
            source = GaussianBeam(energy=E, simulation=simulation, z=0, w0=w0, U0=U0)
        case _:
            raise Exception(f"Unknown source type: {src_type!r}")

    print(RULE)
    print(f"  Source: {src_type}, E={E} eV")
    print(RULE)
    return source, params


def __initialize_aperture(simulation: SimulationObject, saved: dict) -> tuple[Aperture | None, dict]:
    BAR  = "#" * 60
    RULE = "-" * 60
    print(BAR)
    print("  Aperture Setup")
    print(BAR)
    if simulation.dim == 1:
        print("  Available types: SingleSlit  (blank to skip)")
    else:
        print("  Available types: SingleSlit, Circular  (blank to skip)")

    apt_type = _prompt("Aperture type", str, saved.get("type"))

    if not apt_type:
        print("  No aperture.")
        print(RULE)
        return None, {}

    params: dict = {"type": apt_type}

    match apt_type:
        case "SingleSlit":
            width = _prompt("Slit width [m]", float, saved.get("width"))
            params["width"] = width
            height = None
            if simulation.dim == 2:
                height = _prompt("Slit height [m]", float, saved.get("height"))
                params["height"] = height
            aperture = SingleSlit(simulation=simulation, z=0, width=width, height=height)
        case "Circular":
            if simulation.dim != 2:
                raise Exception("Circular aperture requires a 2D simulation")
            radius = _prompt("Radius [m]", float, saved.get("radius"))
            params["radius"] = radius
            aperture = CircularAperture(simulation=simulation, z=0, radius=radius)
        case _:
            raise Exception(f"Unknown aperture type: {apt_type!r}")

    print(RULE)
    print(f"  Aperture: {apt_type}")
    print(RULE)
    return aperture, params

def __initialize_lens(simulation: SimulationObject, source: Waveform, saved: dict) -> tuple[ThinLens | None, dict]:
    BAR  = "#" * 60
    RULE = "-" * 60
    print(BAR)
    print("  Lens Setup")
    print(BAR)
    print("  Available types: Parabolic, FZP, Kinoform, Optical  (blank to skip)")

    lens_type = _prompt("Lens type", str, saved.get("type", ""))

    if not lens_type:
        print("  No lens.")
        print(RULE)
        return None, {}

    if source is None:
        raise Exception("A source must be defined before the lens.")

    f = _prompt("Focal length f [m]", float, saved.get("f"))
    R = _prompt("Aperture radius R [m]", float, saved.get("R"))

    material = _prompt("Lens material (chemical formula)", str, saved.get("material"))
    density  = _prompt("Material density [g/cm^3]", float, saved.get("density"))
    n = xrl.Refractive_Index(material, source.energy / 1000, density)

    params: dict = {"type": lens_type, "f": f, "R": R, "material": material, "density": density}

    match lens_type:
        case "Parabolic":
            lens = XrayParabolicLens(f=f, R=R, n=n, simulation=simulation, z=0)
        case "FZP":
            zone_height = _prompt("Zone height [m] (blank = default λ/2δ)", lambda s: float(s) if s else None, saved.get("zone_height"))
            p = _prompt("Diffraction order p", int, saved.get("p", 2))
            params.update({"zone_height": zone_height, "p": p})
            lens = FZP(wavelength=source.wavelength, f=f, R=R, n=n, simulation=simulation, z=0,
                       zone_height=zone_height, p=p)
        case "Kinoform":
            zone_height = _prompt("Zone height [m] (blank = default λ/δ)", lambda s: float(s) if s else None, saved.get("zone_height"))
            full = _prompt("Full zones? (True/False)", lambda s: s.lower() in ("1", "true", "t", "yes", "y"), saved.get("full", True))
            params.update({"zone_height": zone_height, "full": full})
            lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n, simulation=simulation, z=0,
                            zone_height=zone_height, full=full)
        case "Optical":
            t0 = _prompt("Central thickness t0 [m]", float, saved.get("t0"))
            R1 = _prompt("Radius of curvature R1 [m]", float, saved.get("R1"))
            R2 = _prompt("Radius of curvature R2 [m]", float, saved.get("R2"))
            params.update({"t0": t0, "R1": R1, "R2": R2})
            lens = OpticalLens(R=R, n=n, t0=t0, R1=R1, R2=R2, simulation=simulation, z=0)
        case _:
            raise Exception(f"Unknown lens type: {lens_type!r}")

    print(RULE)
    print(f"  Lens: {lens_type}, f={f} m, R={R} m, material={material}")
    print(RULE)
    return lens, params


def load_params(dir) -> dict:
    if dir:
        with open(_CURR / dir, "r") as file:
            params = json.load(file)
    else:
        params = {}
        
    return params

def save_params(dir, params):
    if not dir:
        dir = "saved_params.json"
        
    with open(_CURR / dir, "w") as file:
        json.dump(params, file)
    
    return

def main():
    args = parser.parse_args()
    simulation = build_simulation(N=args.N, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz)
    
    params = load_params(args.params)
    
    source, source_params = __initialize_source(simulation, params.get("source", dict()))
    if source is None: raise Exception("Source is not defined.")
    aperture, aperture_params = __initialize_aperture(simulation, params.get("aperture", dict()))
    lens, lens_params = __initialize_lens(simulation, source, params.get("lens", dict()))
    
    if args.save:
        new_params = {"source": source_params, "aperture": aperture_params, "lens": lens_params}
        save_params(args.params, new_params)

if __name__ == "__main__":
    main()