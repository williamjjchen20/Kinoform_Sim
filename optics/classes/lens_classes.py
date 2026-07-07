import xraylib
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import os

from .classes import SimulationObject, ThinLens
from .aperture_classes import ApertureFunctions

class CircularLens(ThinLens):
    def __init__(self, f, R, n, simulation: SimulationObject, z, **kwargs):
        self.R = R
        self.R_orig = R
        
        F = ApertureFunctions()
        if simulation.dim == 2:
            assert R <= simulation.Lx/2 and R <= simulation.Ly/2 #type: ignore
            aperture_func = lambda X, Y, r=R, **kw: F.circular_mask(X, Y, r=r)
        else:
            assert R <= simulation.Lx/2
            aperture_func = lambda X, r=R, **kw: F.single_slit_1D(X, r=r)
        super().__init__(f, aperture_func, simulation, z, thickness_func=None, n=n, **kwargs)
        
    def reset(self):
        self.R = self.R_orig
        super().reset()
            
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
        self.delta = (1.-n).real
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
    
class FZP(CircularLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, zone_height: float | None = None, p=2, positive=True, **kwargs):
        self.wavelength = wavelength
        self.delta = (1.-n).real
        self.height = zone_height if zone_height is not None else wavelength/(p*self.delta)
        self.p = p
        self.positive = positive
        
        self.zones = (np.sqrt(f**2+R**2)-f)/(self.wavelength/p)
        zone_locations = FZP.calc_zone_locations(self.wavelength/p, f, R, np.arange(int(np.ceil(self.zones))+1))
        self.zone_locations = zone_locations
        self.zone_widths = FZP.calc_zone_widths(zone_locations)
        super().__init__(f, R, n, simulation, z,**kwargs)
        
    def thickness(self, *args, **kwargs):
        X = args[0]
        r_squared = X**2
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
            
        t_fzp = np.full(r_squared.shape, self.height)
        # indices start at 1 (between 0 and r1)
        r_z = np.clip(np.searchsorted(self.zone_locations, np.sqrt(r_squared), side="right")-1, 0, len(self.zone_locations)-1)

        if self.positive:
            t_fzp[r_z % 2 == 0] = 0
        else:
            t_fzp[r_z % 2 == 1] = 0
            
        return t_fzp
    
    def mth_zone(self, m):
        return self.zone_locations[m]

    @staticmethod
    def calc_zone_locations(wavelength, f, R, m):
        zone_locations = np.sqrt(2*m*f*wavelength + (m*wavelength)**2)
        zone_locations = np.clip(zone_locations, 0, R)
        return zone_locations
    
    @staticmethod
    def calc_zone_widths(zone_locations):
        assert len(zone_locations) > 1
        zone_widths = zone_locations[1:] - zone_locations[:-1]
        return zone_widths
        
class Kinoform(CircularLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, zone_height: float | None = None, **kwargs):
        self.wavelength = wavelength
        self.delta = (1.-n).real
        self.height = self.wavelength/self.delta if zone_height is None else zone_height
        self.effective_wavelength = wavelength if zone_height is None else self.delta*self.height
        
        self.zones = (np.sqrt(f**2+R**2)-f)/self.effective_wavelength
        zone_locations = Kinoform.calc_zone_locations(self.effective_wavelength, f, R, np.arange(int(np.ceil(self.zones))+1))
        self.zone_locations = zone_locations
        self.zone_widths = Kinoform.calc_zone_widths(zone_locations)
        super().__init__(f, R, n, simulation, z,**kwargs)
        
    def thickness(self, *args, **kwargs):
        ## Note: Bandwidth limited by requiring wavelength for a specific energy of x-ray
        X = args[0]
        r_squared = X**2
        t_2pi = self.height
        if self.simulation.dim == 2:
            Y = args[1]
            r_squared += Y**2
            
        t_parabolic = (np.sqrt(r_squared+self.f**2)-self.f)/self.delta
        
        return t_parabolic % t_2pi
    
    def mth_zone(self, m):
        return self.zone_locations[m]
    
    @staticmethod
    def calc_zone_locations(wavelength, f, R, m):
        zone_locations = np.sqrt(2*m*f*wavelength + (m*wavelength)**2)
        zone_locations = np.clip(zone_locations, 0, R)
        return zone_locations
    
    @staticmethod
    def calc_zone_widths(zone_locations):
        assert len(zone_locations) > 1
        zone_widths = zone_locations[1:] - zone_locations[:-1]
        return zone_widths
    
class LensErrors():
    '''
    Takes in a lens of 'Lens' class and returns the lens profile including the error added
    Returns updated lens and errors
    '''
    @staticmethod 
    def cap_height(lens: Kinoform | FZP, h: float, proportion=False) -> tuple[np.ndarray, np.ndarray | None]:
        assert h > 0
        height = lens.height
        if proportion:
            assert h <= 1
            h_max = h*height
        else:
            h_max = h
        
        mask = lens.profile > h_max
        profile = lens.profile
        profile[mask] = h_max
        return profile, None
    
    @staticmethod
    def cap_floor(lens: Kinoform | FZP, h: float, proportion = False) -> tuple[np.ndarray, np.ndarray | None]:
        assert h > 0
        height = lens.height
        if proportion:
            assert h <= 1
            h_min = h*height
        else:
            h_min = h
            
        mask = lens.profile < h_min
        profile = lens.profile
        profile[mask] = h_min
        return profile, None
    
    @staticmethod
    def periodic_etch(lens: ThinLens, err: float, interval: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
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
    def random_etch(lens: ThinLens, max_err: float, interval: int = 1, distribution=None, seed=None) -> tuple[np.ndarray, np.ndarray | None]:
        if seed is None: seed = 0
        rng = np.random.default_rng(seed)
        
        errors = np.zeros_like(lens.profile)
        aperture_mask = lens.aperture_field > 0
        aperture_idx = np.flatnonzero(aperture_mask.ravel())
        
        match distribution:
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
    def gaussian_etch(lens: CircularLens, max_err:float, invert=False, seed=None) -> tuple[np.ndarray, np.ndarray | None]:
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
    def kinoform_zone_placement(kinoform: Kinoform, err: float | np.ndarray, gap=True, mutable=True) -> tuple[np.ndarray, np.ndarray | None]:
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
        if isinstance(err, (int, float)): 
            err = np.full(m_total, err)
        else: 
            assert (len(err) < m_total)
            err = np.asarray(err)
        eps = np.cumsum(err)

        assert np.all(eps[:-1] <= eps[1:]) 
        
        r_m = kinoform.zone_locations
        r_in, r_out = r_m[:-1], r_m[1:]
        
        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)
        
        shifted_outer = r_out + eps # cumulative errors
        zone_idx = np.clip(np.searchsorted(shifted_outer, r, side="right"), 0, m_total - 1)
        h_in  = (np.sqrt(r_in[zone_idx]**2 + f**2) - f) / delta #advanced indexing
        
        ### New kinoform features
        if mutable:
            print("Warning: Mutating original aperture profile...")
            R_new = kinoform.R + eps[-1]
            kinoform_new = Kinoform(kinoform.wavelength, kinoform.f, R_new, n=kinoform.n, simulation=kinoform.simulation.copy(), z=kinoform.z, **kinoform.kwargs)    
            kinoform.R = R_new
            # kinoform.aperture = kinoform_new.aperture
            kinoform.aperture_field = kinoform_new.aperture_field
            kinoform.zone_locations = np.insert(shifted_outer, 0, 0.)
            kinoform.zone_widths = Kinoform.calc_zone_widths(kinoform.zone_locations)
                  
        r_effective = r - eps[zone_idx]
        t_eff = (np.sqrt(r_effective**2 + f**2) - f) / delta
        t = t_eff - h_in
        t[t < 0] = 0
        profile = t * (kinoform.aperture_field > 0)
        
        return profile, np.insert(err, 0, 0.)    
    
    @staticmethod
    def zone_removal(kinoform: Kinoform | FZP, m: int | np.ndarray, proportion: float | np.ndarray, direction: str ="out", extend=False, remove_last=False) -> tuple[np.ndarray, np.ndarray | None]:
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
            print("Warning: Mutating original aperture profile...")
            R_new = kinoform.R - kinoform.zone_widths[-1]
            kinoform_new = Kinoform(kinoform.wavelength, kinoform.f, R_new, n=kinoform.n, simulation=kinoform.simulation.copy(), z=kinoform.z, **kinoform.kwargs)    
            kinoform.R = R_new
            # kinoform.aperture = kinoform_new.aperture
            kinoform.aperture_field = kinoform_new.aperture_field
            kinoform.zone_locations = kinoform.zone_locations[:-1]
            kinoform.zone_widths = kinoform.zone_locations[:-1]
        
            # ms = np.append(ms, m_total - 1)
            # proportions = np.append(proportions, 1.0)

        assert len(proportions) == len(ms), "proportions must match m in length (or be scalar)"

        # band radii, clipped at the physical aperture so the partial zone is handled correctly
        r_m_in = kinoform.zone_locations[:-1]#kinoform.zone_location(ms)
        r_m_out = kinoform.zone_locations[1:]#np.minimum(kinoform.zone_location(ms + 1), kinoform.R)

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
    
    @staticmethod
    def kinoform_sidewall_taper(kinoform: Kinoform, err: float | np.ndarray, proportion=1.) -> tuple[np.ndarray, np.ndarray | None]:
        
        errs = kinoform.add_error(LensErrors.kinoform_zone_placement, err=err, **{"mutable": True})
        assert errs is not None
        
        r_start = np.array(kinoform.zone_locations)
        r_end = r_start + proportion*errs
        
        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)
            
        t_2pi = kinoform.height #kinoform.wavelength/kinoform.delta
        profile = kinoform.profile
        
        for r1, r2 in zip(r_start[1:], r_end[1:]):
            mask = (r < r2) & (r >= r1)
            r_mask = r[mask]
            taper = -t_2pi/(r2-r1)*(r_mask-r2)
            profile[mask] = taper
                
        return profile, errs
    
    @staticmethod
    def FZP_sidewall_taper(FZP: FZP, err: float | np.ndarray, proportion=1.) -> tuple[np.ndarray, np.ndarray | None]:
        zone_locations = np.array(FZP.zone_locations)
        zone_locations = zone_locations[zone_locations > 0]
        
        if FZP.positive:
            r_left, r_right = zone_locations[:-1:2], zone_locations[1:-1:2]
        else:
            r_right, r_left = zone_locations[:-1:2], zone_locations[1:-1:2]

        r_start, r_end = r_left - err, r_right + err
        
        if FZP.dim == 1:
            r = np.abs(FZP.grid)
        else:
            X, Y = FZP.grid
            r = np.sqrt(X**2 + Y**2)
            
        t0 = FZP.height 
        profile = FZP.profile
        
        for r1, r2, r3, r4 in zip(r_start, r_left, r_right, r_end):
            mask_left = (r < r2) & (r >= r1)
            r_mask = r[mask_left]
            taper_left = t0/(r2-r1)*(r_mask-r1)
            profile[mask_left] = taper_left
            
            mask_right = (r < r4) & (r >= r3)
            r_mask = r[mask_right]
            taper_right = -t0/(r4-r3)*(r_mask-r4)
            profile[mask_right] = taper_right
        
        return profile, None
    
