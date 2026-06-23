import xraylib
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import os

from .classes import SimulationObject, ThinLens
from .aperture_classes import ApertureFunctions

class CircularLens(ThinLens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        # assert np.isclose(d/np.abs(R1), 0.) and np.isclose(d/np.abs(R2), 0.)
        self.R = R
        self.delta = (1.-n).real
        
        F = ApertureFunctions()
        if simulation.dim == 2:
            aperture_func = lambda X, Y, r=R, **kw: F.circular_mask(X, Y, r=r)
        else:
            aperture_func = lambda X, r=R, **kw: F.single_slit_1D(X, r=r)
        super().__init__(f, aperture_func, simulation, z, thickness_func=None, n=n, **kwargs)
        
    def plot_profile(self, ax=None, savedir="", label=None):
        '''
        Plot the side profile (thickness vs. x) of the lens from
        `self.profile`. For 2D simulations, takes the central row.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if self.simulation.dim == 1:
            x = self.grid
            t = self.profile
        else:
            X, _ = self.grid
            x = X[0, :]
            cy = self.profile.shape[0] // 2
            t = self.profile[cy, :]

        mask = np.abs(x) <= self.R
        ax.fill_between(x[mask], 0, t[mask], color="steelblue", alpha=0.6)
        ax.plot(x[mask], t[mask], color="navy", lw=1)
        ax.set(xlabel="x [m]", ylabel="thickness [m]",
               title=f"{label} profile (f={self.f:.3g} m, R={self.R:.3g} m)")
        ax.set_xlim(-self.R, self.R)
        ax.axhline(0, color="black", lw=0.5)
        
        if savedir:
            if label is None: label = type(self).__name__
            out = os.path.join(savedir, f"{label}_profile_z={self.center[-1]}.png")
            fig.savefig(out)
            print(f"Saved lens profile to {out}.")
        return ax
    
class OpticalLens(CircularLens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        super().__init__(f, R, n, simulation, z,**kwargs)
        
    def transmittance(self, *args, wavelength, **kwargs):
        X = args[0]
        r_squared = X**2
        k = 2*const.pi/wavelength
        
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2

        t =  np.exp(-1j*k*r_squared/(2*self.f))
        return t

class XrayParabolicLens(CircularLens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        super().__init__(f, R, n, simulation, z,**kwargs)
        
    def thickness(self, *args, **kwargs):
        X = args[0]
        r_squared = X**2
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
        # Elements of modern x-ray physics (2001)
        # t_parabolic = r_squared/(2*self.f*self.delta) # paraxial approximation
        t_parabolic = (np.sqrt(r_squared+self.f**2)-self.f)/self.delta
    
        return t_parabolic
        
class Kinoform(CircularLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, **kwargs):
        self.wavelength = wavelength
        super().__init__(f, R, n, simulation, z,**kwargs)
        self.zones = (np.sqrt(f**2+R**2)-f)/wavelength
        
    def thickness(self, *args, **kwargs):
        ## Note: Bandwidth limited by requiring wavelength for a specific energy of x-ray
        X = args[0]
        r_squared = X**2
        t_2pi = self.wavelength/self.delta
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
        # t_parabolic = r_squared/(2*self.f*self.delta)
        t_parabolic = (np.sqrt(r_squared+self.f**2)-self.f)/self.delta
        return t_parabolic % t_2pi
    
    def r_m(self, m):
        return 2*m*self.f*self.wavelength + (m*self.wavelength)*2
    
class LensErrors():
    '''
    Takes in a lens of 'Lens' class and returns the lens profile including the error added (not mutable)
    Returns updated lens profile and the errors
    '''
    @staticmethod
    def periodic_etch(lens: ThinLens, err: float, interval: int = 1):
        errors = np.zeros_like(lens.profile)
        aperture_mask = lens.aperture_field > 0
        aperture_idx = np.flatnonzero(aperture_mask.ravel())
        
        # interval = len(aperture_idx)//count if count != 0 else 1
        etched_idx = aperture_idx[::interval]
        errors.flat[etched_idx] = err
       
        profile = lens.profile + errors
        
        # prevents errors from breaking plano surface
        if np.all(lens.profile >= 0): profile[profile < 0] = 0
        elif np.all(lens.profile <= 0): profile[profile > 0] = 0
        else: pass
        return profile , errors
    
    @staticmethod
    def random_etch(lens: ThinLens, max_err: float, interval: int = 0, seed=None):
        if seed is None: seed = 0
        rng = np.random.default_rng(seed)
        
        errors = np.zeros_like(lens.profile)
        aperture_mask = lens.aperture_field > 0
        aperture_idx = np.flatnonzero(aperture_mask.ravel())
        random_vals = rng.uniform(low=-max_err, high=max_err, size=lens.profile.size)
        
        count = len(aperture_idx)//interval
        etched_idx = rng.choice(aperture_idx, size=count, replace=False)
        vals = random_vals[etched_idx]
        errors.flat[etched_idx] = vals
        
        profile = lens.profile + errors
        
        # prevents errors from breaking plano surface
        if np.all(lens.profile >= 0): profile[profile < 0] = 0
        elif np.all(lens.profile <= 0): profile[profile > 0] = 0
        else: pass
        return profile, errors
    
    @staticmethod
    def kinoform_taper():
        '''
        
        '''
        
        pass