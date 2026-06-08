import numpy as np
import scipy.constants as const

from classes import Waveform, SimulationObject

class WaveFunctions():
    
    def gaussian_beam_1D(self, x, z=0., wavelength=6.326e-7, w0=0.5e-3, U0=1.0, n=1.0):
        
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
        U = U0*np.sqrt(w0/wz)*np.exp(-x**2/wz**2)*np.exp(-1j*(k*z+k*x**2/(2*Rz)-np.arctan(z/zR)))

        return U

    def gaussian_beam_2D(self, X, Y, z=0., wavelength=6.326e-7, w0=0.5e-3, U0=1.0, n=1.0):
        
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
        U = U0*(w0/wz)*np.exp(-(X**2+Y**2)/wz**2)*np.exp(-1j*(k*z+k*(X**2+Y**2)/(2*Rz)-np.arctan(z/zR)))
        
        return U

    def const_wave_1D(self, X, z=0., wavelength=656.e-9, U0=1.0, n=1.0):
        U = np.ones_like(X)*U0
        return U

    def const_wave_2D(self, X, Y, z=0., wavelength=656.e-9, U0=1.0, n=1.0):
        U = np.ones_like(X)*U0
        return U


class GaussianBeam(Waveform):
    def __init__(self, energy:float, simulation: SimulationObject, z:float):
        super().__init__(energy, simulation, z, self.wave_function)
        
    def wave_function(self, *args, **kwargs):
        z = kwargs.get("z", 0.)
        U0 = kwargs.get("U0", 1.0)
        w0 = kwargs.get("w0", 0.5e-3)
        wavelength = self.wavelength
        n = self.simulation.n
        
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
            
        X = args[0]
        U = np.zeros_like(X)
        if self.simulation.dim == 2:
            Y = args[1]
            U = U0*(w0/wz)*np.exp(-(X**2+Y**2)/wz**2)*np.exp(-1j*(k*z+k*(X**2+Y**2)/(2*Rz)-np.arctan(z/zR)))
        else: # dim == 1
            U = U0*np.sqrt(w0/wz)*np.exp(-X**2/wz**2)*np.exp(-1j*(k*z+k*X**2/(2*Rz)-np.arctan(z/zR)))
            
        return U

class ConstantBeam(Waveform):
    def __init__(self, energy:float, simulation: SimulationObject, z:float):
        super().__init__(energy, simulation, z, self.wave_function)
        
    def wave_function(self, *args, **kwargs):
        X = args[0]
        U0 = kwargs.get("U0", 1.0)
        U = np.ones_like(X)*U0
        return U