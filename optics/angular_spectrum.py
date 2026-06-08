import xraylib
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

from classes.classes import Waveform, SimulationObject

def angular_spectrum_method(U: np.ndarray, z: float, simulation: SimulationObject, wavelength: float, n=1, dim=1) -> np.ndarray:
    '''
    Calculates the propagation of disturbance from initial wavefunction U(0) to U(z) in the specified dimensions
    
    Arguments
    - U: Initial wavefunction 
    - z: Propagation length
    - lambda_: wavelength
    
    Optional Arguments
    - n: refractive index
    - dim: spatial dimension count (1, 2)
    
    Return
    - Uz: propagated wavefunction at distance z
    '''
    # Calculate the intiial angular spectrum as a FT of the initial wavefunction
    if dim == 1:
        fft = np.fft.fft
        ifft = np.fft.ifft
    else: # dim = 2  
        fft = np.fft.fft2
        ifft = np.fft.ifft2
    
    A0 = fft(U)
    K = 2*const.pi*n/wavelength
    if dim == 1:
        kx = 2*const.pi*(np.fft.fftfreq(simulation.Nx, simulation.dx))
        kz = np.sqrt((K**2 - kx**2).astype(complex))
        K_c = kx**2 
    else: # dim == 2: 
        kx = 2*const.pi*np.fft.fftfreq(simulation.Nx, simulation.dx)
        assert simulation.Ny is not None and simulation.dy is not None
        ky = 2*const.pi*np.fft.fftfreq(simulation.Ny, simulation.dy)
        kx, ky = np.meshgrid(kx, ky)
        kz = np.sqrt((K**2 - kx**2 - ky**2).astype(complex))
        K_c = kx**2 + ky**2

    # Transfer function H
    H = np.exp(1j*kz*z)
    H[K_c > K**2] = 0 # evanescent waves filtered out 
    
    # Calculate the propagated waveform via an inverse FT 
    Uz = ifft(A0 * H)
    return Uz
    