import numpy as np
import scipy.constants as const

from classes import Aperture, SimulationObject

class ApertureFunctions():
    def single_slit_1D(self, x, z=0., wavelength=6.326e-7, width=0.5e-3):
        A = width
        k = 2*const.pi/wavelength
        U0 = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(x**2))/(1j*wavelength*z)*A if z != 0 else np.inf
        U = U0*np.sinc(width*x/(wavelength*z))
        return U 
        
    def single_slit_2D(self, X, Y, z=0., wavelength=6.326e-7, width=0.5e-3, height=0.5e-3):
        A = width*height
        k = 2*const.pi/wavelength
        U0 = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(X**2+Y**2))/(1j*wavelength*z)*A
        U = U0*np.sinc(width*X/(wavelength*z))*np.sinc(height*Y/(wavelength*z))
        return U 

class SingleSlit(Aperture):
    def __init__(self, simulation: SimulationObject, z: float, width:float, height:float | None =None):
        self.simulation = simulation
        self.width = width
        if simulation.dim == 2:
            if height is None: raise Exception("Height must be well-defined!")
            self.height = height
            
        super().__init__(simulation, z, self.transmittance_func)
            
    def transmittance_func(self, *args, **kwargs):
        X = args[0]
        mask = np.abs(X) <= self.width/2
        field = np.zeros_like(X)
        
        if self.simulation.dim == 2:
            assert self.height is not None
            Y = args[1]
            mask &= np.abs(Y) <= self.height/2
        
        field[mask] = 1.0
        return field 
    
class Circular(Aperture):
    pass
            
class DoubleSlit(Aperture):
    def __init__(self, simulation: SimulationObject, z: float, separation:float, width1:float, width2:float | None=None, height1=None, height2=None):
        self.simulation = simulation
        self.width1 = width1
        self.width2 = width2 if width2 is not None else width1
        if simulation.dim == 2:
            if height1 is None: raise Exception("Height must be well-defined!")
            self.height1 = height2 if height2 is not None else height1
        
        super().__init__(simulation, z, self.transmittance_func)
            
    def transmittance_func(self, *args, **kwargs):
        pass
        # X = args[0]
        # mask = np.abs(X) <= self.width1/2
        
        # field = np.zeros_like(X)
        
        # if self.simulation.dim == 2:
        #     assert self.height1 is not None
        #     Y = args[1]
        #     mask &= np.abs(Y) <= self.height/2
        
        # field[mask] = 1.0
        # return field 



        