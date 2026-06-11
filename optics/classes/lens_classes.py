import xraylib
import numpy as np
import scipy.constants as const

from classes import SimulationObject, Lens
from .aperture_classes import *

class ThinLens(Lens):
    def __init__(self, f, R, simulation: SimulationObject, z, **kwargs):
        # assert np.isclose(d/np.abs(R1), 0.) and np.isclose(d/np.abs(R2), 0.)
        F = ApertureFunctions()
        aperture_func = lambda X, Y, z=z, r=R: F.circular_mask(X, Y, z=z, r=R)
        
        super().__init__(f, aperture_func, simulation, z, **kwargs)
        
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
    pass