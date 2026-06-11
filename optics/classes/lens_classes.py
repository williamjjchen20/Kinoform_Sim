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
    
    @staticmethod
    def short_kl_1D(X, j: int | np.ndarray, f: float, m: int, wavelength: float, n: float | complex):
        xj_sq = 2*j*(m*wavelength)*f + j**2*(m*wavelength)**2
        l = np.sqrt(X**2 + f**2) - np.sqrt(xj_sq+f**2)
        return l/n.real
    
class ThinLens(Lens):
    def __init__(self, f, R, simulation: SimulationObject, z, **kwargs):
        # assert np.isclose(d/np.abs(R1), 0.) and np.isclose(d/np.abs(R2), 0.)
        self.R = R
        F = ApertureFunctions()
        aperture_func = lambda X, Y, z=z, r=R: F.circular_mask(X, Y, z=z, r=R)
        super().__init__(f, aperture_func, simulation, z, t=0., thickness_func=None, **kwargs)
        
    def func(self, *args, wavelength=6.326e-7):
        X = args[0]
        r_squared = X**2
        k = 2*const.pi/wavelength
        
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2

        t =  np.exp(-1j*k*r_squared/(2*self.f))
        return t
    
        
class Kinoform(Lens):
    def __init__(self):
        pass