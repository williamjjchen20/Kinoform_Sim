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
        
    def plot_profile(self, wavelength, ax=None, savedir="", label=None, y=0.0):
        '''
        Plot the side profile (thickness vs. x) of the kinoform.
        For 2D simulations, takes a slice at y = `y`.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        if self.simulation.dim == 1:
            x = self.grid
            t = self.thickness(x)
        else:
            X, Y = self.grid
            x = X[0, :]
            y_row = np.full_like(x, y)
            t = self.thickness(x, y_row)
        
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
        t_parabolic = r_squared/(2*self.f*self.delta)
    
        return t_parabolic
        
class Kinoform(CircularLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, **kwargs):
        self.wavelength = wavelength
        super().__init__(f, R, n, simulation, z,**kwargs)
        
    def thickness(self, *args, **kwargs):
        ## Note: Bandwidth limited by requiring wavelength for a specific energy of x-ray
        X = args[0]
        r_squared = X**2
        t_2pi = self.wavelength/self.delta
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
        t_parabolic = r_squared/(2*self.f*self.delta)
        return t_parabolic % t_2pi
    
class LensErrors():
    '''
    Takes in a lens of 'Lens' class and returns the lens profile including the error added (not mutable)
    '''
    @staticmethod
    def etch(lens: ThinLens, err: float, count: int | None = None, seed=None):
        if seed is None: seed = 0
        errors = np.ones_like(lens.profile)*err
        errors = errors * lens.aperture_field
        if count is not None:
            N = lens.profile.size//count
            mask = np.arange(0, lens.profile.size, N)
            errors.flat[mask] = 0

        profile = lens.profile + errors
        
        # prevents errors from breaking plano surface
        if np.all(lens.profile >= 0): profile[profile < 0] = 0
        elif np.all(lens.profile <= 0): profile[profile > 0] = 0
        else: pass
        return profile 
    
    @staticmethod
    def random_etch(lens: ThinLens, max_err: float, count:int | None = None, seed=None):
        if seed is None: seed = 0
        rng = np.random.default_rng(seed)
        
        errors = rng.uniform(low=-max_err, high=max_err, size=np.shape(lens.profile))
        errors = errors * lens.aperture_field
        if count is not None: 
            num = lens.profile.size - count
            if num < 0: raise Exception("Cannot go higher than resolution.")
            indices = rng.choice(lens.profile.size, size=num, replace=False)
            errors.flat[indices] = 0
        
        profile = lens.profile + errors
        
        # prevents errors from breaking plano surface
        if np.all(lens.profile >= 0): profile[profile < 0] = 0
        elif np.all(lens.profile <= 0): profile[profile > 0] = 0
        else: pass
        return profile
    