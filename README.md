# Kinoform_Sim

A scalar wave-optics simulation framework for X-ray diffractive lenses, with a focus on kinoform and Fresnel zone plate (FZP) fabrication error modelling.

---

## Overview

Kinoform_Sim propagates coherent scalar wavefields through optical systems composed of sources, apertures, and thin lenses. It is designed for X-ray optics research, where diffractive focusing elements such as kinoforms and FZPs are used to achieve tight focal spots at high photon energies. The package supports both 1-D (slit geometry) and 2-D (full transverse plane) simulations and provides tools to study how realistic fabrication imperfections degrade focal performance.

---

## Project Structure

```
kinoform_sim/
├── optics/
│   ├── classes/
│   │   ├── classes.py          # Base classes: SimulationObject, Waveform, Aperture, ThinLens, Propagator
│   │   ├── wave_classes.py     # GaussianBeam, ConstantBeam; analytic WaveFunctions
│   │   ├── aperture_classes.py # SingleSlit, CircularAperture; analytic DiffractionPatterns
│   │   └── lens_classes.py     # CircularLens, OpticalLens, XrayParabolicLens, FZP, Kinoform, LensErrors
│   ├── metrics/
│   │   ├── metrics.py          # FWHM, focal_efficiency, total_power, Strehl ratio
│   │   ├── error_metrics.py    # Error sweep utilities
│   │   └── lens_comparison.py  # Side-by-side lens metric collection and printing
│   └── propagators.py          # AngularSpectrum, ScaledAngularSpectrum (FFT + NUFFT)
├── simulation/
│   ├── setup.py                # build_simulation(), run() orchestration helper
│   ├── run.py                  # Interactive CLI for configuring and running simulations
│   ├── results.py              # Pre-built result scripts: sweeps, profiles, focal plots
│   └── plotting.py             # Visualisation helpers
├── environment.yml
└── notebook.ipynb
```

---

## Key Components

### Simulation Domain — `SimulationObject`

The central bookkeeping object. It holds the physical box dimensions (`Lx`, `Ly`, `Lz`), the grid resolution (`Nx`, `Ny`), the refractive index of the medium, and references to all placed optical objects (source, lenses, apertures) and the propagator.

```python
from kinoform_sim.optics import SimulationObject

sim = SimulationObject(Lx=3e-4, Nx=100_000, Lz=0.2)          # 1-D
sim = SimulationObject(Lx=3e-4, Nx=1024, Lz=0.2, Ly=3e-4, Ny=1024)  # 2-D
```

### Wave Sources

| Class | Description |
|---|---|
| `GaussianBeam` | TEM₀₀ Gaussian beam parameterised by waist `w0` and photon energy |
| `ConstantBeam` | Uniform plane wave with a plane-wave phase factor `exp(ikz)` |

Energy is supplied in eV; wavelength is derived automatically via `hc/E`.

### Propagators

| Class | Method |
|---|---|
| `AngularSpectrum` | Standard FFT-based angular spectrum method with evanescent filtering and Matsushima–Shimobaba bandwidth limiting |
| `ScaledAngularSpectrum` | Scaled angular spectrum via a type-2 NUFFT (finufft) for zoom-in propagation with magnification factors `Rx`, `Ry ≤ 1` |

Both propagators operate in 1-D or 2-D depending on the simulation dimension.

### Apertures

| Class | Description |
|---|---|
| `SingleSlit` | Rectangular transmission mask; 1-D or 2-D |
| `CircularAperture` | Circular binary mask; 2-D only |

The `DiffractionPatterns` class provides closed-form Fraunhofer patterns (single slit 1-D/2-D, Airy disk) for validation.

### Lenses

All lenses inherit from `ThinLens` (a phase-screen model) and use `xraylib` to look up the complex refractive index `n` for a given material and energy.

| Class | Description |
|---|---|
| `OpticalLens` | Biconvex/biconcave optical lens with lensmaker's equation and parabolic thickness |
| `XrayParabolicLens` | X-ray parabolic refractive lens (CRL element); thickness from `(sqrt(r² + f²) − f) / δ` |
| `FZP` | Binary Fresnel zone plate; alternating opaque/transparent zones, configurable diffraction order `p` |
| `Kinoform` | Blazed diffractive lens; continuous sawtooth phase profile wrapped at `2π`; full or partial zone support |

FZP and Kinoform both expose:
- `zone_locations`, `zone_left`, `zone_right`, `zone_widths`, `zone_heights`
- `mth_zone(m)` — radius of the m-th zone boundary
- `reshape(zone_left, zone_right)` — update zone boundaries after error application
- `copy(dim)` — clone into a different simulation dimension

### Fabrication Error Models — `LensErrors`

`LensErrors` provides a library of static methods that model common X-ray lens fabrication defects. Each method is compatible with `ThinLens.add_error()` and returns `(updated_profile, error_array)`.

| Method | Physical error modelled |
|---|---|
| `cap_height` | Incomplete etch — zones fail to reach full design height |
| `cap_floor` | Residual floor — etched valleys retain a non-zero base thickness |
| `periodic_etch` | Deterministic periodic pixel-level height perturbation |
| `random_etch` | Random roughness; supports uniform, Gaussian, Cauchy, and exponential distributions |
| `gaussian_etch` | Spatially Gaussian-weighted random roughness (centre-heavy or edge-heavy) |
| `zone_shift` | Lateral zone wall misplacement — each zone boundary displaced by a per-zone offset |
| `zone_removal` | Partial zone under/over-etch — a radial strip at the inner or outer zone edge is zeroed |
| `FZP_sidewall_taper` | Linear sidewall taper on FZP transparent zones |
| `zone_quantization` | Replace zone profile with a piecewise-linear anchor interpolation |
| `kinoform_sidewall_taper` | Quadratic taper on the rising edge of each kinoform zone |
| `kinoform_zone_shrink` | Radial zone shrink without taper |
| `kinoform_height_shrink` | Per-zone peak height reduction with preserved curvature shape |
| `kinoform_zone_warping` | Resolution-limited zone collapse toward FZP rectangles for thin outer zones |

### Metrics

| Function | Description |
|---|---|
| `FWHM` | Full-width at half-maximum of the focal intensity (1-D or 2-D radial slice) |
| `total_power` | Integrated intensity over the full simulation window |
| `focal_power` | Integrated intensity within a specified radius of the peak |
| `focal_efficiency` | Ratio of focal power to incident power |
| `strehl_ratio` | Peak intensity ratio of aberrated to ideal lens |
| `max_intensity` | Peak intensity |

---

## Installation

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate xray
```

Additional dependencies used at runtime (not in `environment.yml`):

```bash
pip install pyfftw finufft numba xraylib
```

---

## Usage

### Basic simulation

```python
import xraylib as xrl
from kinoform_sim.optics import (
    SimulationObject, AngularSpectrum,
    GaussianBeam, Kinoform, LensErrors
)

E, f, R = 8e3, 0.1, 1e-4          # 8 keV, f = 10 cm, R = 100 µm
Lx, N, Lz = 3e-4, 150_000, 0.2

n = xrl.Refractive_Index("Si", E / 1000, 2.329)

sim = SimulationObject(Lx=Lx, Nx=N, Lz=Lz)
propagator = AngularSpectrum(sim)

source = GaussianBeam(energy=E, simulation=sim, z=0, w0=R)
lens   = Kinoform(wavelength=source.wavelength, f=f, R=R,
                  n=n, simulation=sim, z=0)

lens.init_transmittance(source)
source.filter(lens)
lens.transform(source)
source.propagate(f, propagator)

source.view()
```

### Applying fabrication errors

```python
lens.add_error(LensErrors.kinoform_height_shrink, height=0.9, proportion=True)
lens.add_error(LensErrors.kinoform_sidewall_taper, err=1e-8, proportion=1.0)
lens.add_error(LensErrors.random_etch, max_err=5e-9, distribution="gaussian", seed=42)
```

### Interactive CLI

```bash
python -m kinoform_sim.simulation.run -N 1024 -Lx 3e-4 -Lz 0.2
```

Pre-saved parameter sets can be loaded with `--params saved_params.json`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array arithmetic |
| `scipy` | FFT, integration, special functions, constants |
| `matplotlib` | Plotting and visualisation |
| `xraylib` | X-ray refractive index lookup |
| `pyfftw` | Fast FFT with plan caching |
| `finufft` | Non-uniform FFT for scaled propagation |
| `numba` | JIT compilation for performance-critical loops |
