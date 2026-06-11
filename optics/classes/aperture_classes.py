import numpy as np
import scipy.constants as const

from classes import Aperture, SimulationObject

class DiffractionPatterns():
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
    
class ApertureFunctions():
    def circular_mask(self, X, Y, z=0., r=1.0):
        field = np.zeros_like(X)
        mask = np.sqrt(X**2+Y**2) <= r
        field[mask] = 1.0
        return field
        

class SingleSlit(Aperture):
    def __init__(self, simulation: SimulationObject, z: float, width:float, height:float | None =None):
        self.width = width
        if simulation.dim == 2:
            if height is None: raise Exception("Height must be well-defined!")
            self.height = height
            
        super().__init__(simulation, z)
            
    def func(self, *args):
        X = args[0]
        mask = np.abs(X) <= self.width/2
        field = np.zeros_like(X)
        
        if self.simulation.dim == 2:
            assert self.height is not None
            Y = args[1]
            mask &= np.abs(Y) <= self.height/2
        
        field[mask] = 1.0
        return field 
    
class CircularAperture(Aperture):
    def __init__(self, simulation: SimulationObject, z: float, radius: float):
        if simulation.dim == 1: raise Exception("Check simulation dimensions!")
        self.radius = radius
            
        super().__init__(simulation, z)

    def func(self, *args):
        X = args[0]
        Y = args[1]
        mask = np.sqrt(X**2 + Y**2) <= self.radius
        field = np.zeros_like(X)
        
        field[mask] = 1.0
        return field