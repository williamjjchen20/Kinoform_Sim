import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import finufft as nufft

from .classes import SimulationObject, Waveform, Propagator

class AngularSpectrum(Propagator):
    def __init__(self, simulation):
        super().__init__(angular_spectrum_method, simulation=simulation)
    
    def _conditions(self, wave: Waveform, z, D):
        '''
        Nyquist sampling limit must be met.
        
        '''
        wavelength = wave.wavelength
        sim = wave.simulation
        Lx, Ly, Nx, Ny = sim.Lx, sim.Ly, sim.Nx, sim.Ny
        dim = self.dim
        assert dim == sim.dim
        
        f_s = wavelength*np.abs(z)/D
        if Lx/Nx > f_s: 
            raise Exception(f"Nyquist criterion not met! Use Nx >= {np.round(Lx/f_s)}")
        if dim == 2 and Ly/Ny > f_s: #type: ignore
            raise Exception(f"Nyquist criterion not met! Use Ny >= {np.round(Ly/f_s)}")
        
        self.validated = True
        
        return
    
class ScaledAngularSpectrum(Propagator):
    def __init__(self, simulation, Rx=1., Ry=1.):
        super().__init__(scaled_angular_spectrum_method, simulation=simulation, Rx=Rx, Ry=Ry)
    

def angular_spectrum_method(wave: Waveform, dz: float, n:float=1.) -> np.ndarray:
    '''
    Calculates the propagation of disturbance from initial wavefunction U(0) to U(z) in the specified dimensions. 
    
    This function implements the ASM with a FFT and IFFT. 
    It is better suited for near-field propagation in the Rayleight limit of z < 2*D^2/lambda but adopts the bandwidth limits from
    (Matsushima and Shimobaba, 2009) to adopt greater paraxial far field accuracy.

    
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
    
    # Parameters
    simulation = wave.simulation
    U = wave.field
    wavelength = wave.wavelength
    dim = wave.dim
    
    Lx, Ly = simulation.Lx, simulation.Ly
    Nx, Ny = simulation.Nx, simulation.Ny
    dx, dy = simulation.dx, simulation.dy
    
    # Set input waveform to nyquist ordering (0-centered)
    U0 = np.fft.fftshift(U)
    # Calculate the intiial angular spectrum as a FT of the initial wavefunction
    if dim == 1:
        fft = np.fft.fft
        ifft = np.fft.ifft
    else: # dim = 2  
        fft = np.fft.fft2
        ifft = np.fft.ifft2
    
    A0 = fft(U0)
    K = 2*const.pi*n/wavelength
    if dim == 1:
        # nyquist ordering
        kx = 2*const.pi*(np.fft.fftfreq(Nx, dx))
        kz = np.sqrt((K**2 - kx**2).astype(complex))
        
        # Evanescent condition
        K_c = kx**2 
        # Bandwidth limit
        kx_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Lx)**2+1)*wavelength)
        band_mask = np.abs(kx) <= kx_max
        
    else: # dim == 2: 
        kx = 2*const.pi*np.fft.fftfreq(Nx, dx)
        assert Ly is not None and Ny is not None and dy is not None
        ky = 2*const.pi*np.fft.fftfreq(Ny, dy)
        kx, ky = np.meshgrid(kx, ky)
        kz = np.sqrt((K**2 - kx**2 - ky**2).astype(complex))
        
        # Evanescent condition
        K_c = kx**2 + ky**2
        # Bandwidth limit
        kx_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Lx)**2+1)*wavelength)
        ky_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Ly)**2+1)*wavelength)
        band_mask = np.abs(kx) <= kx_max
        band_mask &= np.abs(ky) <= ky_max
        
    # Transfer function H
    H = np.exp(1j*kz*dz)
    H[K_c > K**2] = 0 # evanescent waves filtered out 
    H[~band_mask] = 0 # aliasing removed 
    
    # Calculate the propagated waveform via an inverse FT and shift back to standard ordering
    Uz = np.fft.ifftshift(ifft(A0 * H))
    return Uz
    
def scaled_angular_spectrum_method(wave: Waveform, dz: float, n: float=1., Rx=1., Ry=1.) -> np.ndarray:
    '''
    Scaled angular spectrum method based on Shimobaba et al. (2012), Eq. (3).
    
    Uses a standard FFT for the forward spectrum, applies the free-space
    transfer function, then evaluates the inverse transform on a scaled
    output grid via a type-2 NUFFT.

    Arguments
    - wave: Waveform object
    - dz: propagation distance
    - n: refractive index
    - Rx, Ry: magnification factors (output grid = R * input grid)

    Return
    - Uz: propagated wavefunction at distance z
    '''
    
    simulation = wave.simulation
    U = wave.field
    wavelength = wave.wavelength
    dim = wave.dim
    
    Lx, Ly = simulation.Lx, simulation.Ly
    Nx, Ny = simulation.Nx, simulation.Ny
    dx, dy = simulation.dx, simulation.dy
    
    assert Rx <= 1 and Ry <= 1 # Zoom in condition
    
    # Set input waveform to nyquist ordering (0-centered)
    U0 = np.fft.ifftshift(U)
    K = 2*const.pi*n/wavelength
    if dim == 1:
        fft = np.fft.fft
        
        #center shift correction
        shiftx = np.abs(Lx-Rx*Lx)/2
        phi_c = shiftx

        mf = np.arange(Nx) - Nx//2
        kx = 2*const.pi * mf / (Nx*dx)
        # standard array ordering
        kz = np.sqrt((K**2 - kx**2 - (2*const.pi)**2*phi_c).astype(complex))
        
        # Evanescent condition
        K_c = kx**2
        # Bandwidth limit
        kx_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Lx)**2+1)*wavelength)
        band_mask = np.abs(kx) <= kx_max
    
    else: # dim = 2  
        assert Ly is not None and Ny is not None and dy is not None
        fft = np.fft.fft2
        
        # center shift correction
        shiftx, shifty = np.abs(Lx-Rx*Lx)/2, np.abs(Ly-Ry*Ly)/2
        phi_c = shiftx+shifty
        
        mfx, mfy = np.arange(Nx) - Nx//2, np.arange(Ny) - Ny//2
        # standard array ordering
        kx, ky = 2*const.pi * mfx / (Nx * dx), 2*const.pi * mfy / (Ny * dy)
    
        kx, ky = np.meshgrid(kx, ky)
        mfx, mfy = np.meshgrid(mfx, mfy)
        
        kz = np.sqrt((K**2 - kx**2 - ky**2-(2*const.pi)**2*phi_c).astype(complex))
        
        # Evanescent Condition
        K_c = kx**2 + ky**2
        
        # Bandwidth Limit
        kx_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Lx)**2+1)*wavelength)
        ky_max = 2*const.pi * 1.0/(np.sqrt((2*dz/Ly)**2+1)*wavelength)
        band_mask = np.abs(kx) <= kx_max
        band_mask &= np.abs(ky) <= ky_max
    
        
    A0 = np.fft.ifftshift(fft(U0))
    H = np.exp(1j*kz*dz)
    H[K_c > K**2] = 0
    H[~band_mask] = 0
    Az = A0 * H
    
    if dim == 1:
        x_out = 2*const.pi*Rx*mf/Nx
        Uz = nufft.nufft1d2(x_out, Az, eps=1e-8, isign=1)/Nx
        wave.Rx = Rx
    else: # dim == 2:
        assert Ny is not None
        x_out = 2*const.pi * Rx * mfx.ravel() / Nx
        y_out = 2*const.pi * Ry * mfy.ravel() / Ny
        Uz = nufft.nufft2d2(x_out, y_out, Az, eps=1e-8, isign=1).reshape(Ny, Nx)/(Nx * Ny)
        wave.Rx, wave.Ry = Rx, Ry
        
    return Uz
    