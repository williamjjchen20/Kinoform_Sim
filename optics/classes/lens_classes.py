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
        
    def plot_profile(self, ax=None, savedir="", labels=None):
        '''
        Plot the side profile (thickness vs. x) of the lens from
        `self.profile`. For 2D simulations, takes the central row.
        
        `labels` is a dict supporting keys:
          - "label": series label used for the saved filename / title
          - "xlabel", "ylabel": axis labels
          - "title": axis title (overrides default)
          - "xscale", "yscale": matplotlib scales (e.g. "linear", "log")
          - "x_scale_factor", "y_scale_factor": multiplicative factors
        A bare string is accepted for backward compatibility and treated as
        {"label": labels}.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if isinstance(labels, str) or labels is None:
            labels = {"label": labels}

        label = labels.get("label", None)
        x_scale_factor = labels.get("x_scale_factor", 1.0)
        y_scale_factor = labels.get("y_scale_factor", 1.0)
        xlabel = labels.get("xlabel", "x [m]")
        ylabel = labels.get("ylabel", "thickness [m]")
        title  = labels.get("title", f"{label} profile (f={self.f:.3g} m, R={self.R:.3g} m)")
        xscale = labels.get("xscale", "linear")
        yscale = labels.get("yscale", "linear")

        if self.simulation.dim == 1:
            x = self.grid
            t = self.profile
        else:
            X, _ = self.grid
            x = X[0,:]
            cy = self.profile.shape[0] // 2
            t = self.profile[cy,:]

        mask = np.abs(x) <= self.R
        x_plot = x[mask] * x_scale_factor
        t_plot = t[mask] * y_scale_factor
        ax.fill_between(x_plot, 0, t_plot, color="steelblue", alpha=0.6)
        ax.plot(x_plot, t_plot, color="navy", lw=1)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title,
               xscale=xscale, yscale=yscale)
        ax.set_xlim(-self.R * x_scale_factor, self.R * x_scale_factor)
        ax.axhline(0, color="black", lw=0.5)
        
        if savedir:
            if label is None: label = type(self).__name__
            out = os.path.join(savedir, f"{label}_profile_z={self.center[-1]}.png")
            fig.savefig(out)
            print(f"Saved lens profile to {out}.")
        return ax
    
class OpticalLens(CircularLens):
    def __init__(self, R, n, t0, R1, R2, simulation: SimulationObject, z, **kwargs):
        self.t0 = t0
        self.R1 = R1
        self.R2 = R2
        assert (R1 != R2)
        self.f = 1/((n-1)*(1/R1-1/R2))
        print(self.f)
        super().__init__(self.f, R, n, simulation, z,**kwargs)
        
    def thickness(self, *args, **kwargs):
        X = args[0]
        r_squared = X**2
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
        
        t_parabolic = self.t0-r_squared/2*(1/self.R1 - 1/self.R2)
    
        return t_parabolic


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
        t_parabolic = (np.sqrt(r_squared+self.f**2)-self.f)/self.delta
    
        return t_parabolic
        
class Kinoform(CircularLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, **kwargs):
        self.wavelength = wavelength
        super().__init__(f, R, n, simulation, z,**kwargs)
        self.height = wavelength/self.delta
        self.zones = (np.sqrt(f**2+R**2)-f)/wavelength
        
    def thickness(self, *args, **kwargs):
        ## Note: Bandwidth limited by requiring wavelength for a specific energy of x-ray
        X = args[0]
        r_squared = X**2
        t_2pi = self.wavelength/self.delta
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
            
        t_parabolic = (np.sqrt(r_squared+self.f**2)-self.f)/self.delta
        
        return t_parabolic % t_2pi
    
    def zone_location(self, m):
        return np.sqrt(2*m*self.f*self.wavelength + (m*self.wavelength)**2)
    
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
        return profile, errors
    
    @staticmethod
    def random_etch(lens: ThinLens, max_err: float, interval: int = 0, distribution_func=None, seed=None):
        if seed is None: seed = 0
        rng = np.random.default_rng(seed)
        
        errors = np.zeros_like(lens.profile)
        aperture_mask = lens.aperture_field > 0
        aperture_idx = np.flatnonzero(aperture_mask.ravel())
        
        match distribution_func:
            case "gaussian":
                random_vals = max_err*rng.normal(size=lens.profile.size)
            case "cauchy":
                random_vals = max_err*rng.standard_cauchy(size=lens.profile.size)
            case "exponential":
                random_vals = max_err*rng.standard_exponential(size=lens.profile.size)
            case _:
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
    def gaussian_etch(lens: CircularLens, max_err:float, invert=False, seed=None):
        '''
        Generates error distributed over circular lens aperture according to a Gaussian distribution
        '''
        
        if seed is None: seed = 0
        rng = np.random.default_rng(seed)
        
        ## 3 sigma within aperture
        sigma = lens.R/3
        if lens.dim == 1:
            x = lens.grid
            r_squared = x**2
        else:
            X, Y = lens.grid
            r_squared = X**2+Y**2
       
        distribution = np.exp(-r_squared/(2*sigma**2))
        if invert: distribution = 1.-distribution
        err = rng.uniform(low=-max_err, high=max_err, size=np.shape(lens.profile))
        errors = distribution*err
        
        profile = lens.profile + errors
        if np.all(lens.profile >= 0): profile[profile < 0] = 0
        elif np.all(lens.profile <= 0): profile[profile > 0] = 0
        else: pass
        return profile, errors
    
    @staticmethod
    def zone_placement(kinoform: Kinoform, err: float | np.ndarray, gap=False, seed=None):
        '''
        Shift zone boundaries by a cumulative placement error of size `err` per
        zone, rebuild the parabolic profile so each sample's thickness is
        measured from its (shifted) zone's inner radius.
        '''              
        f = kinoform.f
        delta = kinoform.delta
        
        ###
        m_total = int(np.ceil(kinoform.zones))
        # cumulative per-zone shift: eps[m] is applied to outer boundary r_m[m]
        eps = np.arange(m_total) * err # for zones 1 to m
        print(eps)

        r_m = kinoform.zone_location(np.arange(m_total+1))
        # r_m[1:] += eps
        r_in, r_out = r_m[:-1], r_m[1:]
        
        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)
        
        shifted_outer = r_out + eps # cumulative errors
        zone_idx = np.clip(np.searchsorted(shifted_outer, r, side="right"), 0, m_total - 1)
        h_in  = (np.sqrt(r_in[zone_idx]**2 + f**2) - f) / delta
        
        if gap:            
            r_effective = r - eps[zone_idx]
            t_eff = (np.sqrt(r_effective**2 + f**2) - f) / delta
            t = t_eff - h_in
            t[t < 0] = 0
            profile = t * (kinoform.aperture_field > 0)

        else:
            t_parabolic = (np.sqrt(r**2 + f**2) - f) / delta
            profile = (t_parabolic - h_in) * (kinoform.aperture_field > 0)

        return profile, eps
    
    @staticmethod
    def sidewall_tapering(kinoform):
        pass
    
    
    
    @staticmethod
    def zone_removal(kinoform: Kinoform, m: int | np.ndarray, proportion: float | np.ndarray, direction: str ="out", extend=False, remove_last=False):
        '''
        Adds a taper by a specified percentage on a specified lateral zone

        args
        - kinoform: Kinoform lens
        - m: lateral zone number (negative indices count back from the last band)
        - proportion: radius proportion tapered off (>=0. & <= 1.)

        kwargs
        - direction: inward "in" or outward "out" from the specified zone
        - extend: taper all zones m' >= m
        - remove_last: also remove the trailing partial zone (band that the
          aperture clips). Equivalent to extending the sweep through m_total-1
          with proportion=1.
        '''
        zones = kinoform.zones
        # number of zone bands, including the trailing partial one inside R
        m_total = int(np.ceil(zones))
        if isinstance(m, (int, float)): ms = np.array([m])
        else: ms = np.asarray(m)
        if isinstance(proportion, (int, float)): proportions = np.array([proportion])
        else: proportions = np.asarray(proportion)
        
        # convert negative indices to positive equivalents wrt zone bands
        ms = np.where(ms < 0, m_total + ms, ms)
        assert np.all((ms >= 0) & (ms < m_total)), f"m must be in [0, {m_total})"

        if extend:
            m_min = int(np.min(ms))
            ms = np.arange(m_min, m_total)
            if proportions.size != ms.size:
                proportions = np.append(proportions,
                                        np.full(ms.size - proportions.size, proportions[-1]))

        # ensure the partial last band is fully removed if requested
        if remove_last and not extend:
            ms = np.append(ms, m_total - 1)
            proportions = np.append(proportions, 1.0)

        print(ms, proportions)
        assert len(proportions) == len(ms), "proportions must match m in length (or be scalar)"

        # band radii, clipped at the physical aperture so the partial zone is handled correctly
        r_m_in = kinoform.zone_location(ms)
        r_m_out = np.minimum(kinoform.zone_location(ms + 1), kinoform.R)

        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)

        profile = np.array(kinoform.profile)
        for r_in, r_out, p in zip(r_m_in, r_m_out, proportions):
            width = r_out - r_in
            if width <= 0: continue   # band lies entirely outside the aperture
            if direction.lower() == "out":
                r_cut = r_in + p*width
                mask = (r < r_cut) & (r >= r_in)
            elif direction.lower() == "in":
                r_cut = r_out - p*width
                mask = (r < r_out) & (r >= r_cut)
            else:
                raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
            profile[mask] = 0.

        return profile, None