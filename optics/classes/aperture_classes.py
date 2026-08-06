import numpy as np
import scipy.constants as const
import scipy.special as special

from .classes import Aperture, SimulationObject

class DiffractionPatterns():
    """Analytical Fraunhofer diffraction patterns for common aperture geometries.

    All methods assume the far-field (Fraunhofer) regime and raise an exception
    if the propagation distance `z` violates the Fraunhofer condition
    ``z > 0.1 * D**2 / wavelength``.
    """

    @staticmethod
    def fraunhofer_condition(D, wavelength, L):
        """Raise an exception if the Fraunhofer far-field condition is not satisfied.

        Parameters
        ----------
        D : float
            Characteristic aperture size [m].
        wavelength : float
            Optical wavelength [m].
        L : float
            Propagation distance from aperture to observation plane [m].

        Raises
        ------
        Exception
            If ``L <= 0.1 * D**2 / wavelength``.
        """
        if L <= 0.1*D**2/wavelength: raise Exception("Fraunhofer condition violated!")
        return
    
    @staticmethod
    def single_slit_1D(x, z=0., wavelength=6.326e-7, width=0.5e-3):
        """Compute the 1-D Fraunhofer diffraction pattern for a single slit.

        Uses the analytic sinc-based far-field result for a rectangular slit of
        finite ``width`` and infinite height.

        Parameters
        ----------
        x : array_like
            Transverse observation coordinates [m].
        z : float, optional
            Propagation distance [m]. Default is 0.
        wavelength : float, optional
            Optical wavelength [m]. Default is 632.6 nm.
        width : float, optional
            Slit width [m]. Default is 0.5 mm.

        Returns
        -------
        U : ndarray
            Complex field amplitude at the observation plane.
        """
        DiffractionPatterns.fraunhofer_condition(width, wavelength, z)
        A = width
        k = 2*const.pi/wavelength
        U0 = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(x**2))/(1j*wavelength*z)*A if z != 0 else np.inf
        U = U0*np.sinc(width*x/(wavelength*z))
        return U 
    
    @staticmethod  
    def single_slit_2D(X, Y, z=0., wavelength=6.326e-7, width=0.5e-3, height=0.5e-3):
        """Compute the 2-D Fraunhofer diffraction pattern for a rectangular aperture.

        Uses the separable sinc product for a slit of given ``width`` (x) and
        ``height`` (y).

        Parameters
        ----------
        X : array_like
            2-D array of transverse x observation coordinates [m].
        Y : array_like
            2-D array of transverse y observation coordinates [m].
        z : float, optional
            Propagation distance [m]. Default is 0.
        wavelength : float, optional
            Optical wavelength [m]. Default is 632.6 nm.
        width : float, optional
            Aperture width along x [m]. Default is 0.5 mm.
        height : float, optional
            Aperture height along y [m]. Default is 0.5 mm.

        Returns
        -------
        U : ndarray
            Complex field amplitude at the observation plane.
        """
        DiffractionPatterns.fraunhofer_condition(max(width, height), wavelength, z)
        A = width*height
        k = 2*const.pi/wavelength
        U0 = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(X**2+Y**2))/(1j*wavelength*z)*A
        U = U0*np.sinc(width*X/(wavelength*z))*np.sinc(height*Y/(wavelength*z))
        return U 
    
    @staticmethod
    def circular(X, Y, z=0., wavelength=6.326e-7, radius=0.5e-3):
        """Compute the 2-D Fraunhofer diffraction pattern for a circular aperture (Airy pattern).

        Uses the first-order Bessel function ``J1`` to produce the Airy disk
        intensity profile.

        Parameters
        ----------
        X : array_like
            2-D array of transverse x observation coordinates [m].
        Y : array_like
            2-D array of transverse y observation coordinates [m].
        z : float, optional
            Propagation distance [m]. Default is 0.
        wavelength : float, optional
            Optical wavelength [m]. Default is 632.6 nm.
        radius : float, optional
            Aperture radius [m]. Default is 0.5 mm.

        Returns
        -------
        U : ndarray
            Complex field amplitude at the observation plane.
        """
        DiffractionPatterns.fraunhofer_condition(2*radius, wavelength, z)
        A = const.pi*radius**2
        k = 2*const.pi/wavelength
        r = np.sqrt(X**2+Y**2)
        arg = k*radius*r/z

        U0 = np.exp(1j*k*z)*np.exp(1j*k*r**2/(2*z))/(1j*wavelength*z)*A
        U = U0*(2*special.jv(1, arg)/(arg))
        return U
    
class ApertureFunctions():
    """Static utility functions that return binary transmission masks over a grid."""

    @staticmethod
    def circular_mask(X, Y, r=1.0):
        """Return a 2-D circular binary mask.

        Parameters
        ----------
        X : array_like
            2-D array of x coordinates.
        Y : array_like
            2-D array of y coordinates.
        r : float, optional
            Mask radius (in the same units as ``X`` and ``Y``). Default is 1.0.

        Returns
        -------
        field : ndarray
            Array of the same shape as ``X`` with 1.0 inside the circle and
            0.0 outside.
        """
        field = np.zeros_like(X)
        mask = np.sqrt(X**2+Y**2) <= r
        field[mask] = 1.0
        return field
    
    @staticmethod
    def single_slit_1D(X, r=1.0):
        """Return a 1-D single-slit binary mask.

        Parameters
        ----------
        X : array_like
            1-D array of x coordinates.
        r : float, optional
            Half-width of the slit (in the same units as ``X``). Default is 1.0.

        Returns
        -------
        field : ndarray
            Array of the same shape as ``X`` with 1.0 within ``|X| <= r`` and
            0.0 elsewhere.
        """
        field = np.zeros_like(X)
        mask = np.abs(X) <= r
        field[mask] = 1.0
        return field
        
class SingleSlit(Aperture):
    """A rectangular single-slit aperture compatible with 1-D and 2-D simulations.

    For 1-D simulations only ``width`` is required. For 2-D simulations both
    ``width`` and ``height`` must be provided.

    Parameters
    ----------
    simulation : SimulationObject
        The simulation domain to which this aperture belongs.
    z : float
        Axial position of the aperture [m].
    width : float
        Full width of the slit along x [m].
    height : float or None, optional
        Full height of the slit along y [m]. Required for 2-D simulations.

    Raises
    ------
    Exception
        If ``simulation.dim == 2`` and ``height`` is ``None``.
    """

    def __init__(self, simulation: SimulationObject, z: float, width:float, height:float | None =None):
        self.width = width
        if simulation.dim == 2:
            if height is None: raise Exception("Height must be well-defined!")
            self.height = height
            
        super().__init__(simulation, z)
            
    def func(self, *args, **kwargs):
        """Evaluate the slit transmission mask on the simulation grid.

        Parameters
        ----------
        *args : array_like
            ``args[0]`` is the x coordinate array; ``args[1]`` is the y
            coordinate array (required for 2-D simulations).

        Returns
        -------
        field : ndarray
            Binary transmission mask (0 or 1).
        """
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
    """A circular aperture for 2-D simulations.

    Parameters
    ----------
    simulation : SimulationObject
        The simulation domain to which this aperture belongs. Must be 2-D.
    z : float
        Axial position of the aperture [m].
    radius : float
        Radius of the aperture [m].

    Raises
    ------
    Exception
        If ``simulation.dim == 1`` (circular apertures require a 2-D grid).
    """

    def __init__(self, simulation: SimulationObject, z: float, radius: float):
        if simulation.dim == 1: raise Exception("Check simulation dimensions!")
        self.radius = radius
            
        super().__init__(simulation, z)

    def func(self, *args, **kwargs):
        """Evaluate the circular aperture transmission mask on the simulation grid.

        Parameters
        ----------
        *args : array_like
            ``args[0]`` is the 2-D x coordinate array; ``args[1]`` is the 2-D
            y coordinate array.

        Returns
        -------
        field : ndarray
            Binary transmission mask (0 or 1).
        """
        X = args[0]
        Y = args[1]
        mask = np.sqrt(X**2 + Y**2) <= self.radius
        field = np.zeros_like(X)
        
        field[mask] = 1.0
        return field
