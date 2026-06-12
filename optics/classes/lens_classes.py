import xraylib
import numpy as np
import scipy.constants as const

from classes import SimulationObject, Lens
from .aperture_classes import *

class ThicknessFunctions():
    @staticmethod
    def concave_plano_1D(X, f: float, n: float | complex):
        l = np.sqrt(X**2 + f**2) - f
        return l/n.real
    
    def concave_plano_2D(self):
        pass
    
    @staticmethod
    def short_kl_1D(X, j: int | np.ndarray, f: float, m: int, wavelength: float, n: float | complex):
        xj_sq = 2*j*(m*wavelength)*f + j**2*(m*wavelength)**2
        l = np.sqrt(X**2 + f**2) - np.sqrt(xj_sq+f**2)
        return l/n.real
    
class ThinLens(Lens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        # assert np.isclose(d/np.abs(R1), 0.) and np.isclose(d/np.abs(R2), 0.)
        self.R = R
        F = ApertureFunctions()
        if simulation.dim == 2:
            aperture_func = lambda X, Y, z=z, r=R: F.circular_mask(X, Y, r=r)
        else:
            aperture_func = lambda X, r=R: F.single_slit_1D(X, r=r)
        super().__init__(f, aperture_func, simulation, z, thickness_func=None, n=n, **kwargs)
        
    def transmittance(self, *args, wavelength, **kwargs):
        X = args[0]
        r_squared = X**2
        k = 2*const.pi/wavelength
        
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2

        t =  np.exp(-1j*k*r_squared/(2*self.f))
        return t
    
### implement later
class ThickLens(Lens):
    ''' 
    Model the thick lens as a dim+1 dimensional object that requires specific propagation through thickness
    Includes both refraction and diffraction effects
    
    '''
    
    pass
        
class Kinoform(Lens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        # assert np.isclose(d/np.abs(R1), 0.) and np.isclose(d/np.abs(R2), 0.)
        self.R = R
        self.delta = (n-1.).real

        F = ApertureFunctions()
        if simulation.dim == 2:
            aperture_func = lambda X, Y, z=z, r=self.R: F.circular_mask(X, Y, r=r)
        else:
            aperture_func = lambda X, r=self.R: F.single_slit_1D(X, r=r)
        super().__init__(f, aperture_func, simulation, z, thickness_func=None, n=n, **kwargs)
        
    def thickness(self, *args, wavelength, **kwargs):
        X = args[0]
        r_squared = X**2
        t_2pi = wavelength/self.delta
        if self.dim == 2:
            Y = args[1]
            r_squared += Y*2
        t_parabolic = r_squared/(2*self.f*self.delta)
        return t_parabolic % t_2pi

        