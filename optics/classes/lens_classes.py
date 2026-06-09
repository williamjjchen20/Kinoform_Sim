import xraylib
import numpy as np
import scipy.constants as const

from classes import SimulationObject, Lens

class ThinLens(Lens):
    def __init__(self, simulation: SimulationObject, z, aperture_func, R1, R2, d=0, n=1):
        
        assert np.isclose(d/np.abs(R1), 0) and np.isclose(d/np.abs(R2), 0)
        
        super().__init__(simulation, z, aperture_func, R1, R2, d, transmittance_func=self.transmittance_func, n=n)
        
    def transmittance_func(self, *args, **kwargs):
        X = args[0]
        R2 = X**2
        wavelength = kwargs["wavelength"]
        k = 2*const.pi/wavelength
        
        if self.simulation.dim == 2:
            Y = args[1]
            R2 += Y**2

        t =  np.exp(1j*k*self.d)*np.exp(-1j*k*R2/(2*self.f))
        return t
    
        
class Kinoform(Lens):
    pass