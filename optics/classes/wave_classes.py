import numpy as np
import scipy.constants as const

from .classes import Waveform, SimulationObject

class WaveFunctions():
    """Static analytic waveform functions for common beam profiles.

    These are standalone functions that can be evaluated on an arbitrary grid
    without requiring a :class:`~.classes.SimulationObject`. For simulation-
    integrated waveforms use :class:`GaussianBeam` or :class:`ConstantBeam`.
    """

    @staticmethod
    def gaussian_beam_1D(x, z=0., wavelength=6.326e-7, w0=0.5e-3, U0=1.0, n=1.0):
        """Evaluate a 1-D Gaussian beam profile at propagation distance ``z``.

        Parameters
        ----------
        x : array_like
            Transverse coordinate array [m].
        z : float, optional
            Propagation distance from the beam waist [m]. Default is 0.
        wavelength : float, optional
            Optical wavelength in vacuum [m]. Default is 632.6 nm.
        w0 : float, optional
            Beam waist radius (1/e field radius at z=0) [m]. Default is 0.5 mm.
        U0 : float, optional
            Peak field amplitude at the waist. Default is 1.0.
        n : float, optional
            Refractive index of the medium. Default is 1.0.

        Returns
        -------
        U : ndarray
            Complex field amplitude of the Gaussian beam.
        """
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
        U = U0*np.sqrt(w0/wz)*np.exp(-x**2/wz**2)*np.exp(-1j*(k*z+k*x**2/(2*Rz)-np.arctan(z/zR)))
        return U
    
    @staticmethod
    def gaussian_beam_2D(X, Y, z=0., wavelength=6.326e-7, w0=0.5e-3, U0=1.0, n=1.0):
        """Evaluate a 2-D Gaussian beam profile at propagation distance ``z``.

        Parameters
        ----------
        X : array_like
            2-D array of transverse x coordinates [m].
        Y : array_like
            2-D array of transverse y coordinates [m].
        z : float, optional
            Propagation distance from the beam waist [m]. Default is 0.
        wavelength : float, optional
            Optical wavelength in vacuum [m]. Default is 632.6 nm.
        w0 : float, optional
            Beam waist radius (1/e field radius at z=0) [m]. Default is 0.5 mm.
        U0 : float, optional
            Peak field amplitude at the waist. Default is 1.0.
        n : float, optional
            Refractive index of the medium. Default is 1.0.

        Returns
        -------
        U : ndarray
            Complex field amplitude of the 2-D Gaussian beam.
        """
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
        U = U0*(w0/wz)*np.exp(-(X**2+Y**2)/wz**2)*np.exp(-1j*(k*z+k*(X**2+Y**2)/(2*Rz)-np.arctan(z/zR)))
        return U
    
    @staticmethod
    def const_wave_1D(X, z=0., wavelength=656.e-9, U0=1.0, n=1.0):
        """Return a uniform 1-D plane wave with constant amplitude ``U0``.

        Parameters
        ----------
        X : array_like
            1-D transverse coordinate array [m].
        z : float, optional
            Propagation distance [m]. Unused; included for API consistency.
        wavelength : float, optional
            Optical wavelength [m]. Unused; included for API consistency.
        U0 : float, optional
            Field amplitude. Default is 1.0.
        n : float, optional
            Refractive index. Unused; included for API consistency.

        Returns
        -------
        U : ndarray
            Array of the same shape as ``X`` filled with ``U0``.
        """
        U = np.ones_like(X)*U0
        return U
    
    @staticmethod
    def const_wave_2D(X, Y, z=0., wavelength=656.e-9, U0=1.0, n=1.0):
        """Return a uniform 2-D plane wave with constant amplitude ``U0``.

        Parameters
        ----------
        X : array_like
            2-D array of transverse x coordinates [m].
        Y : array_like
            2-D array of transverse y coordinates [m]. Unused; included for API
            consistency.
        z : float, optional
            Propagation distance [m]. Unused; included for API consistency.
        wavelength : float, optional
            Optical wavelength [m]. Unused; included for API consistency.
        U0 : float, optional
            Field amplitude. Default is 1.0.
        n : float, optional
            Refractive index. Unused; included for API consistency.

        Returns
        -------
        U : ndarray
            Array of the same shape as ``X`` filled with ``U0``.
        """
        U = np.ones_like(X)*U0
        return U


class GaussianBeam(Waveform):
    """A Gaussian beam waveform integrated with a :class:`~.classes.SimulationObject`.

    Supports both 1-D and 2-D simulation domains. The beam is initialized at
    axial position ``z`` and can be propagated through the simulation using a
    compatible :class:`~.classes.Propagator`.

    Parameters
    ----------
    energy : float
        Photon energy [eV]. Determines the wavelength via ``hc / energy``.
    simulation : SimulationObject
        The simulation domain to which this waveform belongs.
    z : float
        Initial axial position of the beam waist [m].
    **kwargs
        Additional keyword arguments passed to :meth:`func`:

        - **U0** (*float*) -- Peak field amplitude at the waist. Default is 1.0.
        - **w0** (*float*) -- Beam waist radius [m]. Default is 1.0.
    """
    
    def __init__(self, energy:float, simulation: SimulationObject, z:float, **kwargs):
        super().__init__(energy, simulation, z, **kwargs)
        
    def func(self, *args, z=0., U0=1.0, w0=1.0, **kwargs):
        """Evaluate the Gaussian beam field on the simulation grid.

        Handles both 1-D (scalar ``args[0]``) and 2-D (``args[0]``, ``args[1]``)
        grids automatically based on ``self.simulation.dim``.

        Parameters
        ----------
        *args : array_like
            Grid coordinates. ``args[0]`` is always the x array; ``args[1]``
            is the y array for 2-D simulations.
        z : float, optional
            Propagation distance from the beam waist [m]. Default is 0.
        U0 : float, optional
            Peak field amplitude. Default is 1.0.
        w0 : float, optional
            Beam waist radius [m]. Default is 1.0.

        Returns
        -------
        U : ndarray
            Complex field amplitude of the Gaussian beam on the grid.
        """
        wavelength = self.wavelength
        n = self.simulation.n
        
        zR = const.pi*w0**2*n/wavelength
        wz = w0*np.sqrt(1+(z/zR)**2)
        k = 2*const.pi/wavelength
        Rz = np.inf if z == 0 else z*(1+(zR/z)**2)
            
        X = args[0]
        
        if self.simulation.dim == 2:
            Y = args[1]
            U = U0*(w0/wz)*np.exp(-(X**2+Y**2)/wz**2)*np.exp(1j*(k*z+k*(X**2+Y**2)/(2*Rz)-np.arctan(z/zR)))
        else: # dim == 1
            U = U0*np.sqrt(w0/wz)*np.exp(-X**2/wz**2)*np.exp(1j*(k*z+k*X**2/(2*Rz)-0.5*np.arctan(z/zR)))
            
        return U

class ConstantBeam(Waveform):
    """A uniform plane wave waveform integrated with a :class:`~.classes.SimulationObject`.

    Produces a spatially constant amplitude field modulated by a plane-wave
    phase factor ``exp(i k z)``. Supports both 1-D and 2-D simulation domains.

    Parameters
    ----------
    energy : float
        Photon energy [eV]. Determines the wavelength via ``hc / energy``.
    simulation : SimulationObject
        The simulation domain to which this waveform belongs.
    z : float
        Initial axial position of the beam [m].
    **kwargs
        Additional keyword arguments passed to :meth:`func`:

        - **U0** (*float*) -- Field amplitude. Default is 1.0.
    """

    def __init__(self, energy:float, simulation: SimulationObject, z:float, **kwargs):
        super().__init__(energy, simulation, z, **kwargs)
       
    def func(self, *args, z=0, U0=1.0, **kwargs):
        """Evaluate the constant plane wave field on the simulation grid.

        Parameters
        ----------
        *args : array_like
            Grid coordinates. ``args[0]`` is the x coordinate array (shape
            determines the output shape).
        z : float, optional
            Axial position used to compute the plane-wave phase ``exp(i k z)``
            [m]. Default is 0.
        U0 : float, optional
            Field amplitude. Default is 1.0.

        Returns
        -------
        U : ndarray
            Complex field array of the same shape as ``args[0]``, equal to
            ``U0 * exp(i k z)`` everywhere.
        """
        X = args[0]
        wavelength = self.wavelength
        k = 2*const.pi/wavelength
        U = np.ones_like(X)*U0*np.exp(1j*k*z)
        return U
