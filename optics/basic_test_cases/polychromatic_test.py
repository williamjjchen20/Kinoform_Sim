import numpy as np
import scipy.constants as const
import os, sys
from pathlib import Path

from ..classes import *
from ..metrics import *
from ..propagators import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./test_figs").resolve()

JOULE_TO_EV = 1/const.e

# X-ray parameters shared with physics_test / metrics_test
Lx, Lz, N = 1.5e-4, 10000, 2048
N_2D = 512          # 2-D grid kept small: a 7-sample 2048^2 Gaussian takes ~20 s to build
E0 = 8.5e3          # central energy [eV]
R = 5e-5            # beam waist for Gaussian sources [m]
N_SPEC = 7          # spectral samples

# Relative bandwidths (FWHM): Si(111) DCM and multilayer monochromator
BW_SI111 = 1.4e-4
BW_MULTILAYER = 1e-2


def gaussian_spectrum(E0, rel_bw, n_samples):
    '''
    Gauss-Hermite sampling of a Gaussian spectrum centered at E0 with FWHM rel_bw*E0.
    Returns energies [eV] and (unnormalized) quadrature weights.
    '''
    x, w = np.polynomial.hermite_e.hermegauss(n_samples)
    sigma = rel_bw*E0/(2*np.sqrt(2*np.log(2)))
    return E0 + sigma*x, w/np.sqrt(2*const.pi)


def _make_sim(dim, precision="double"):
    if dim == 1:
        return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, precision=precision)
    return SimulationObject(Lx=Lx, Ly=Lx, Nx=N_2D, Ny=N_2D, Lz=Lz, precision=precision)


def _make_mono(waveform_cls, E, sim, z, **kwargs):
    return waveform_cls(energy=E, simulation=sim, z=z, **kwargs)


def _make_poly(waveform_cls, energies, weights, sim, z, **kwargs):
    return PolychromaticWaveform(energies, weights, waveform_cls, sim, z, **kwargs)


def _beams():
    return [(ConstantBeam, {}), (GaussianBeam, {"w0": R})]


def test_init_shapes():
    print("="*60)
    print("Polychromatic initialization: shapes, dtypes, attributes")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    for dim in (1, 2):
        for precision in ("double", "single"):
            for waveform_cls, kwargs in _beams():
                sim = _make_sim(dim, precision)
                wave = _make_poly(waveform_cls, energies, weights, sim, z=0, **kwargs)
                spatial = (N,) if dim == 1 else (N_2D, N_2D)
                spec_shape = (N_SPEC,) + (1,)*dim

                assert wave.field.shape == (N_SPEC,) + spatial, wave.field.shape
                assert wave.field.dtype == sim.complex_dtype, wave.field.dtype
                assert wave.spec_shape == (-1,) + (1,)*dim
                assert np.shape(wave.wavelength) == spec_shape
                assert np.allclose(np.ravel(wave.wavelength), const.h*const.c*JOULE_TO_EV/energies, rtol=1e-14)
                assert np.isclose(np.sum(wave.weights), 1.0)
                assert np.all(wave.weights > 0)
                assert sim.objects["source"] is wave
                assert np.all(np.isfinite(wave.field))
                print(f"  [PASS] {dim}-D {precision:<6} {waveform_cls.__name__:<12} field {wave.field.shape} {wave.field.dtype}")


def test_component_equivalence():
    print("="*60)
    print("Each spectral slice == sqrt(w_i) * monochromatic waveform at E_i")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    w_norm = weights/np.sum(weights)
    for dim in (1, 2):
        for precision, rtol in (("double", 1e-12), ("single", 1e-5)):
            for z in (0., 0.5):
                for waveform_cls, kwargs in _beams():
                    poly = _make_poly(waveform_cls, energies, weights, _make_sim(dim, precision), z=z, **kwargs)
                    max_err = 0.
                    for i, (E, w) in enumerate(zip(energies, w_norm)):
                        mono = _make_mono(waveform_cls, E, _make_sim(dim, precision), z=z, **kwargs)
                        ref = np.sqrt(w)*mono.field
                        assert np.allclose(poly.field[i], ref, rtol=rtol, atol=rtol*np.abs(ref).max()), \
                            f"slice {i} (E={E:.3f} eV) mismatch"
                        max_err = max(max_err, np.abs(poly.field[i]-ref).max()/np.abs(ref).max())
                    print(f"  [PASS] {dim}-D {precision:<6} z={z:<4} {waveform_cls.__name__:<12} max rel err {max_err:.2e}")


def test_intensity():
    print("="*60)
    print("Spectrally summed intensity and power")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    w_norm = weights/np.sum(weights)
    for dim in (1, 2):
        spatial = (N,) if dim == 1 else (N_2D, N_2D)
        for waveform_cls, kwargs in _beams():
            poly = _make_poly(waveform_cls, energies, weights, _make_sim(dim), z=0.5, **kwargs)
            I = poly.intensity()
            I_spec = poly.spectral_intensity()
            assert I.shape == spatial, I.shape
            assert I_spec.shape == (N_SPEC,) + spatial, I_spec.shape
            assert np.allclose(I, I_spec.sum(axis=0))

            I_ref = np.zeros(spatial)
            for E, w in zip(energies, w_norm):
                I_ref += w*_make_mono(waveform_cls, E, _make_sim(dim), z=0.5, **kwargs).intensity()
            assert np.allclose(I, I_ref, rtol=1e-12)

            # normalized weights: a polychromatic beam carries the same power as one monochromatic beam
            P_poly = total_power(poly)
            P_mono = total_power(_make_mono(waveform_cls, E0, _make_sim(dim), z=0.5, **kwargs))
            assert np.isclose(P_poly, P_mono, rtol=1e-6), (P_poly, P_mono)
            print(f"  [PASS] {dim}-D {waveform_cls.__name__:<12} P_poly/P_mono = {P_poly/P_mono:.10f}")


def test_single_energy():
    print("="*60)
    print("N_spec = 1 reduces to the monochromatic waveform")
    print("="*60)

    for dim in (1, 2):
        for waveform_cls, kwargs in _beams():
            poly = _make_poly(waveform_cls, [E0], [1.0], _make_sim(dim), z=0.5, **kwargs)
            mono = _make_mono(waveform_cls, E0, _make_sim(dim), z=0.5, **kwargs)
            assert poly.field.shape == (1,) + mono.field.shape
            assert np.allclose(poly.field[0], mono.field, rtol=1e-14)
            assert np.allclose(poly.intensity(), mono.intensity(), rtol=1e-14)
            print(f"  [PASS] {dim}-D {waveform_cls.__name__}")


def test_weight_normalization():
    print("="*60)
    print("Weights are normalized regardless of input scale")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_SI111, N_SPEC)
    for dim in (1, 2):
        a = _make_poly(GaussianBeam, energies, weights, _make_sim(dim), z=0, w0=R)
        b = _make_poly(GaussianBeam, energies, 1e3*weights, _make_sim(dim), z=0, w0=R)
        assert np.allclose(a.weights, b.weights)
        assert np.allclose(a.field, b.field)
        print(f"  [PASS] {dim}-D weights sum = {np.sum(b.weights):.12f}")

    # flat spectrum: equal weights
    flat = _make_poly(ConstantBeam, energies, np.ones(N_SPEC), _make_sim(1), z=0)
    assert np.allclose(flat.weights, 1/N_SPEC)
    print(f"  [PASS] flat spectrum weights = 1/{N_SPEC}")


def test_mismatched_spectrum():
    print("="*60)
    print("Mismatched energies/weights lengths are rejected")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_SI111, N_SPEC)
    try:
        _make_poly(ConstantBeam, energies, weights[:-1], _make_sim(1), z=0)
    except ValueError as e:
        print(f"  [PASS] raised ValueError: {e}")
    else:
        raise AssertionError("len(weights) != len(energies) was accepted")


def test_repr_copy_phase():
    print("="*60)
    print("__repr__, copy, phase")
    print("="*60)

    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    for dim in (1, 2):
        wave = _make_poly(GaussianBeam, energies, weights, _make_sim(dim), z=0.5, w0=R)
        print(" ", wave)
        assert isinstance(repr(wave), str)

        dup = wave.copy()
        assert isinstance(dup, PolychromaticWaveform)
        assert dup.simulation is not wave.simulation
        assert dup.waveform_cls is wave.waveform_cls
        assert np.allclose(dup.energy, wave.energy)
        assert np.allclose(dup.weights, wave.weights)
        assert np.allclose(dup.field, wave.field)
        dup.field *= 0
        assert np.any(wave.field != 0), "copy shares field memory with original"

        assert wave.phase() is None
        print(f"  [PASS] {dim}-D")
        
def test_simple_propagation():
    print("="*60)
    print("Single-energy polychromatic propagation == monochromatic propagation")
    print("="*60)

    dz = 1.0

    for dim in (1, 2):
        for waveform_cls, kwargs in _beams():
            sim_poly, sim_mono = _make_sim(dim), _make_sim(dim)
            poly = _make_poly(waveform_cls, np.array([E0]), [1.0], sim_poly, z=0, **kwargs)
            mono = _make_mono(waveform_cls, E0, sim_mono, z=0, **kwargs)

            poly.propagate(dz, AngularSpectrum(simulation=sim_poly))
            mono.propagate(dz, AngularSpectrum(simulation=sim_mono))

            assert poly.z == mono.z == dz
            spatial = (N,) if dim == 1 else (N_2D, N_2D)
            assert mono.field.shape == spatial, mono.field.shape
            assert poly.field.shape == (1,) + spatial, poly.field.shape
            assert np.all(np.isfinite(poly.field))
            max_err = np.abs(poly.field[0]-mono.field).max()/np.abs(mono.field).max()
            assert np.allclose(poly.field[0], mono.field, rtol=1e-12, atol=1e-12*np.abs(mono.field).max()), \
                f"field mismatch, max rel err {max_err:.2e}"
            assert np.allclose(poly.intensity(), mono.intensity(), rtol=1e-12)
            print(f"  [PASS] {dim}-D {waveform_cls.__name__:<12} max rel err {max_err:.2e}")
            
def test_poly_propagation():
    print("="*60)
    print("Polychromatic propagation")
    print("="*60)
    
    dz = 1.0
    
    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    for dim in (1, 2):
        waveform_cls, kwargs = (GaussianBeam, {"w0": R})
        sim_poly, sim_mono = _make_sim(dim), _make_sim(dim)
        poly = _make_poly(waveform_cls, energies, weights, sim_poly, z=0, **kwargs)
        mono = _make_mono(waveform_cls, E0, sim_mono, z=0, **kwargs)

        poly.propagate(dz, AngularSpectrum(simulation=sim_poly))
        mono.propagate(dz, AngularSpectrum(simulation=sim_mono))
        
        assert poly.z == mono.z == dz #type: ignore 
        spatial = (N,) if dim == 1 else (N_2D, N_2D)
        assert mono.field.shape == spatial, mono.field.shape
        assert poly.field.shape != (1,) + spatial, poly.field.shape
        max_err = np.abs(poly.field[0]-mono.field).max()/np.abs(mono.field).max()
        print(f"field mismatch, max rel err {max_err:.2e}")
        assert not np.allclose(poly.intensity(), mono.intensity())
        save = savedir
        poly.view(savedir=str(save / f"{waveform_cls.__name__}_Polychromatic_{dim}D"))
        
def test_poly_scaled_propagation():
    print("="*60)
    print("Polychromatic scaled propagation")
    print("="*60)

    dz = 1.0
    Rx, Ry = 0.5, 0.5   # for even N, every 2nd output sample lands on the unscaled ASM grid

    energies, weights = gaussian_spectrum(E0, BW_MULTILAYER, N_SPEC)
    for dim in (1, 2):
        waveform_cls, kwargs = (GaussianBeam, {"w0": R})
        sim_poly, sim_mono, sim_ref = _make_sim(dim), _make_sim(dim), _make_sim(dim)
        poly = _make_poly(waveform_cls, energies, weights, sim_poly, z=0, **kwargs)
        mono = _make_mono(waveform_cls, E0, sim_mono, z=0, **kwargs)
        ref = _make_poly(waveform_cls, energies, weights, sim_ref, z=0, **kwargs)

        poly.propagate(dz, ScaledAngularSpectrum(sim_poly, Rx=Rx, Ry=Ry))
        mono.propagate(dz, ScaledAngularSpectrum(sim_mono, Rx=Rx, Ry=Ry))
        ref.propagate(dz, AngularSpectrum(simulation=sim_ref))

        assert poly.z == mono.z == dz #type: ignore
        spatial = (N,) if dim == 1 else (N_2D, N_2D)
        assert mono.field.shape == spatial, mono.field.shape
        assert poly.field.shape == (N_SPEC,) + spatial, poly.field.shape
        assert poly.Rx == Rx and poly.Ry == (Ry if dim == 2 else 1.)

        # Each spectral slice must match unscaled ASM over the zoomed window
        if dim == 1:
            zoomed, expected = poly.field[:, ::2], ref.field[:, N//4:3*N//4]
        else:
            window = slice(N_2D//4, 3*N_2D//4)
            zoomed, expected = poly.field[:, ::2, ::2], ref.field[:, window, window]
        spatial_axes = tuple(range(1, dim+1))
        slice_errs = np.abs(zoomed-expected).max(axis=spatial_axes)/np.abs(expected).max(axis=spatial_axes)
        max_err = slice_errs.max()
        assert max_err < 1e-6, f"zoomed field mismatch, rel err by slice {slice_errs}"
        print(f"  [PASS] {dim}-D {waveform_cls.__name__:<12} max rel err vs unscaled ASM {max_err:.2e}")

        save = savedir
        extent = [-Rx*Lx/2, Rx*Lx/2, -Ry*Lx/2, Ry*Lx/2]   # used by 2-D view only; 1-D scales by wave.Rx
        poly.view(savedir=str(save / f"{waveform_cls.__name__}_Polychromatic_Scaled_{dim}D"), labels={"extent": extent})


if __name__ == "__main__":
    test_init_shapes()
    test_component_equivalence()
    test_intensity()
    test_single_energy()
    test_weight_normalization()
    test_mismatched_spectrum()
    test_repr_copy_phase()
    test_simple_propagation()
    test_poly_propagation()
    test_poly_scaled_propagation()