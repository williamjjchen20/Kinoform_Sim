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
        
    def reshape(self, R):
        self.R = R
        
        F = ApertureFunctions()
        if self.simulation.dim == 2:
            assert R <= self.simulation.Lx/2 and R <= self.simulation.Ly/2 #type: ignore
            new_aperture_func = lambda X, Y, r=R, **kw: F.circular_mask(X, Y, r=r)
        else:
            assert R <= self.simulation.Lx/2
            new_aperture_func = lambda X, r=R, **kw: F.single_slit_1D(X, r=r)
        
        super().reshape(new_aperture_func)
        
    def plot_profile(self, ax=None, savedir="", labels=None, R_min=None, R_max=None):
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
    
        if R_min is None: R_min = -self.R
        if R_max is None: R_max = self.R
        ax.set_xlim(R_min * x_scale_factor, R_max* x_scale_factor)
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
        # print(self.f)
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
    
class PhaseWrappedLens(CircularLens):
    '''
    Shared base for phase-wrapped diffractive lenses (FZP, Kinoform).

    Subclasses are expected to set (typically in their own __init__ before
    calling super().__init__):
      - self.wavelength : design wavelength [m]
      - self.delta      : 1 - Re(n)
      - self.height     : full phase-step height [m]
      - self.zones      : (fractional) number of zones inside R
      - self.zone_locations : 1D array of zone-boundary radii (length zones+1)
      - self.zone_widths    : zone_locations[1:] - zone_locations[:-1]

    This class intentionally does not define __init__ or thickness so the
    existing FZP / Kinoform constructors and profile logic are untouched.
    '''
        
    def mth_zone(self, m):
        self.zone_locations: np.ndarray
        return self.zone_locations[m]

    @staticmethod
    def calc_zone_widths(zone_locations):
        assert len(zone_locations) > 1
        return zone_locations[1:] - zone_locations[:-1]

class FZP(PhaseWrappedLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, zone_height: float | None = None, p=2, positive=True, **kwargs):
        self.wavelength = wavelength
        self.delta = (1.-n).real
        self.height = zone_height if zone_height is not None else wavelength/(p*self.delta)
        self.p = p
        self.positive = positive
        
        self.zones = (np.sqrt(f**2+R**2)-f)/(self.wavelength/p)
        zone_locations = FZP.calc_zone_locations(self.wavelength/p, f, R, np.arange(int(np.ceil(self.zones))+1))
        self.zone_locations = zone_locations
        self.zone_left = zone_locations[:-1]
        self.zone_right = zone_locations[1:]
        self.zone_widths = FZP.calc_zone_widths(self.zone_left, self.zone_right)
        super().__init__(f, R, n, simulation, z,**kwargs)
        self.mutated = False
        
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
    
    def reshape(self, zone_left, zone_right):
        self.zone_locations = np.insert(zone_right, 0, 0.)
        self.zone_left = zone_left
        self.zone_right = zone_right
        self.zone_widths = self.calc_zone_widths(zone_left, zone_right)
        
        R_new = zone_right[-1]
        super().reshape(R=R_new)
        self.mutated = True

    @staticmethod
    def calc_zone_locations(wavelength, f, R, m):
        zone_locations = np.sqrt(2*m*f*wavelength + (m*wavelength)**2)
        zone_locations = np.clip(zone_locations, 0, R)
        return zone_locations
    
    @staticmethod
    def calc_zone_widths(zone_left, zone_right):
        assert (len(zone_left) == len(zone_right)) & (np.all(zone_left <= zone_right))
        zone_widths = zone_right - zone_left
        return zone_widths
        
class Kinoform(PhaseWrappedLens):
    def __init__(self, wavelength, f, R, n, simulation: SimulationObject, z, full=True, zone_height: float | None = None, **kwargs):
        self.wavelength = wavelength
        self.delta = (1.-n).real
        self.height = self.wavelength/self.delta if zone_height is None else zone_height
        self.effective_wavelength = wavelength if zone_height is None else self.delta*self.height
        
        self.zones = (np.sqrt(f**2+R**2)-f)/self.effective_wavelength
        if full: self.zones = int(np.ceil(self.zones))
        zone_locations = Kinoform.calc_zone_locations(self.effective_wavelength, f, np.arange(self.zones+1))
        if not full: zone_locations = np.clip(zone_locations, 0, R)
        
        self.zone_locations = zone_locations
        self.zone_left = zone_locations[:-1]
        self.zone_right = zone_locations[1:]
        self.zone_widths = Kinoform.calc_zone_widths(self.zone_left, self.zone_right)
        super().__init__(f, zone_locations[-1], n, simulation, z,**kwargs)
        
        self.R = zone_locations[-1]
        self.mutated = False
        
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
        
    def reshape(self, zone_left, zone_right, R=None):
        self.zone_locations = np.insert(zone_right, 0, 0.)
        self.zone_left = zone_left
        self.zone_right = zone_right
        self.zone_widths = self.calc_zone_widths(zone_left, zone_right)
        
        R_new = zone_right[-1]
        if R is None: R = R_new
        super().reshape(R=R)
        self.mutated = True
        
    @staticmethod
    def calc_zone_locations(wavelength, f, m,  R=np.inf):
        zone_locations = np.sqrt(2*m*f*wavelength + (m*wavelength)**2)
        return zone_locations
    
    @staticmethod
    def calc_zone_widths(zone_left, zone_right):
        assert (len(zone_left) == len(zone_right)) & (np.all(zone_left <= zone_right))
        zone_widths = zone_right - zone_left
        return zone_widths
    
class CompoundLens(ThinLens):
    pass
    
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
    def zone_shift(lens: Kinoform | FZP, err, verbose=False):
        m_total = int(np.ceil(lens.zones))
        # cumulative per-zone shift: eps[m] is applied to outer boundary r_m[m]
        if isinstance(err, (int, float)): 
            err = np.full(m_total, err)
            # err[0] = 0.
        else: 
            assert (len(err) <= m_total)
            err = np.asarray(err)
            
        eps = np.cumsum(err)
        
        if lens.dim == 1:
            r = np.abs(lens.grid)
        else:
            X, Y = lens.grid
            r = np.sqrt(X**2 + Y**2)
        
        profile = np.zeros_like(lens.profile)
        # zone_locations = lens.zone_locations
        r_left_all, r_right_all = lens.zone_left, lens.zone_right#zone_locations[:-1], zone_locations[1:]
        
        assert (len(eps) == len(r_left_all) == len(r_right_all))
        
        for r_l, r_r, e, in zip(r_left_all, r_right_all, eps):
            mask = (r >= r_l) & (r < r_r)
            if not np.any(mask):
                continue

            r_src = r[mask]
            h_src = lens.profile[mask]
            
            # sort radius/thickness pairs for linear interpolation
            order = np.argsort(r_src)
            r_src, h_src = r_src[order], h_src[order]

            r_l_new, r_r_new = r_l + e, r_r + e
            mask_new = (r >= r_l_new) & (r < r_r_new)
            if not np.any(mask_new):
                continue

            # shift back into the source zone's frame and interpolate
            r_query = r[mask_new] - e
            
            profile[mask_new] = np.interp(r_query, r_src, h_src)
            
        # if mutable:
        if verbose: print("Warning: Mutating original lens profile...")
        
        r_left_new= r_left_all + eps
        r_right_new = r_right_all + eps
        lens.reshape(r_left_new, r_right_new)
        profile *= lens.aperture_field > 0
            
        return profile, err

    @staticmethod
    def zone_removal(kinoform: Kinoform, m: int | np.ndarray = -1, err: float | np.ndarray = 0., direction: str ="out", extend=False, remove_last=False, mutable=False) -> tuple[np.ndarray, np.ndarray | None]:
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
          aperture clips). Equivalent to appending m_total-1 with proportion=1.
        - mutable: if True, shrink the lens aperture (R, aperture_field,
          zone_locations, zone_widths) to exclude fully-removed outer zones,
          mirroring the mutation semantics in `zone_placement`.
        '''
        zones = kinoform.zones
        zone_widths = kinoform.zone_widths
        # number of zone bands, including the trailing partial one inside R
        m_total = int(np.ceil(zones))
        if isinstance(m, (int, float)): ms = np.array([m])
        else: ms = np.asarray(m)
        if isinstance(err, (int, float)): errs = np.array([err], dtype=np.float64)
        else: errs = np.asarray(err, dtype=np.float64)

        # convert negative indices to positive equivalents wrt zone bands
        ms = np.where(ms < 0, m_total + ms, ms)
        assert np.all((ms >= 0) & (ms < m_total)), f"m must be in [0, {m_total})"

        if extend:
            m_min = int(np.min(ms))
            ms = np.arange(m_min, m_total)
            if errs.size != ms.size:
                errs = np.append(errs,
                                        np.full(ms.size - errs.size, errs[-1]))

        # ensure the partial last band is fully removed if requested
        if remove_last:
            last = m_total - 1
            if last not in ms:
                ms = np.append(ms, last)
                errs = np.append(errs, 1.1*zone_widths[-1]) # overshoot by 10% for floating point errors
            else:
                errs[ms == last] = 1.1*zone_widths[-1]

        # print(errs)
        assert len(errs) == len(ms), "errs must match m in length (or be scalar)"
        assert np.all(ms < len(zone_widths))

        # band radii, clipped at the physical aperture so the partial zone is handled correctly
        r_m_in = kinoform.zone_left#kinoform.zone_locations[:-1]
        r_m_out = kinoform.zone_right#kinoform.zone_locations[1:]

        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)

        profile = np.array(kinoform.profile)
        # track which bands are fully removed (for aperture mutation)
        removed_full = np.zeros(m_total, dtype=bool)
        for mi, p in zip(ms, errs):
            r_in, r_out = r_m_in[mi], r_m_out[mi]
            # print("Removing...", r_in, r_out)
            
            width = r_out - r_in
            if p >= width:
                removed_full[mi] = True
                p = width
                
            if width <= 0: continue   # band lies entirely outside the aperture
            if direction.lower() == "out":
                r_cut = r_in + p
                mask = (r <= r_cut) & (r >= r_in)
            elif direction.lower() == "in":
                r_cut = r_out - p
                mask = (r <= r_out) & (r >= r_cut)
            else:
                raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
            profile[mask] = 0.

        if mutable:
            # shrink aperture down to the outermost surviving band
            surviving = np.flatnonzero(~removed_full)
            if surviving.size == 0:
                print("Warning: all zones removed; leaving aperture unchanged.")
            else:
                m_last = int(surviving.max())
                # only shrink if the outer tail (m_last+1 .. m_total-1) is fully gone
                if np.all(removed_full[m_last+1:]):
                    R_new = float(r_m_out[m_last])
                    if R_new < kinoform.R:
                        print("Warning: Shrinking aperture...")
                        kinoform.zones = len(surviving)
                kinoform.reshape(r_m_in[:m_last+1], r_m_out[:m_last+1]-errs[:m_last+1])
        
        profile = profile * (kinoform.aperture_field > 0)

        return profile, errs
    
    @staticmethod
    def FZP_sidewall_taper(FZP: FZP, err: float | np.ndarray, proportion=1.) -> tuple[np.ndarray, np.ndarray | None]:
        zone_locations = np.array(FZP.zone_locations)
        zone_locations = zone_locations[zone_locations > 0]
        
        if FZP.positive:
            r_left, r_right = zone_locations[:-1:2], zone_locations[1:-1:2]
        else:
            r_right, r_left = zone_locations[:-1:2], zone_locations[1:-1:2]

        r_start, r_end = r_left - proportion*err, r_right + proportion*err
        
        if FZP.dim == 1:
            r = np.abs(FZP.grid)
        else:
            X, Y = FZP.grid
            r = np.sqrt(X**2 + Y**2)
            
        t0 = FZP.height 
        profile = FZP.profile
        
        for r1, r2, r3, r4 in zip(r_start, r_left, r_right, r_end):
            mask_left = (r <= r2) & (r >= r1)
            r_mask = r[mask_left]
            taper_left = t0/(r2-r1)*(r_mask-r1)
            profile[mask_left] = taper_left
            
            mask_right = (r <= r4) & (r >= r3)
            r_mask = r[mask_right]
            taper_right = -t0/(r4-r3)*(r_mask-r4)
            profile[mask_right] = taper_right
        
        return profile, None
    
    @staticmethod
    def zone_quantization(lens: Kinoform, points, m : int | np.ndarray=-1) -> tuple[np.ndarray, np.ndarray | None]:
        '''
        Quantize the profile within one or more zones using user-supplied
        (radius, height) points. Within each target zone, the profile is
        the piecewise-linear interpolation of the anchors that fall inside that
        zone, extended to the zone boundaries by anchoring
            (r_left, 0) and (r_right, h)   (h = lens.height, the 2pi step).
        Untargeted zones are left untouched.

        Parameters
        ----------
        lens : Kinoform | FZP
            Lens whose zone_locations define the bands.
        points : array-like, shape (K, 2)
            Anchor (radius, height) pairs. Anchors outside every targeted zone
            are ignored.
        m : int | array-like of int, default -1
            Zone indices to quantize. Negative indices count from the last band.
        '''
        
        r_left_all, r_right_all = np.asarray(lens.zone_left), np.asarray(lens.zone_right) #zone_locations[:-1], zone_locations[1:]
        m_total = int(np.ceil(lens.zones))

        ms = np.atleast_1d(np.asarray(m)).astype(int)
        ms = np.where(ms < 0, m_total + ms, ms)
        assert np.all((ms >= 0) & (ms < m_total)), f"m must be in [0, {m_total})"
        ms = np.unique(ms)

        r_left = r_left_all[ms]
        r_right = r_right_all[ms]

        if lens.dim == 1:
            r = np.abs(lens.grid)
        else:
            X, Y = lens.grid
            r = np.sqrt(X**2 + Y**2)
            
        assert (len(points) == len(ms) == len(r_left) == len(r_right))
        profile = np.array(lens.profile)
        
        for points, r_l, r_r in zip(points, r_left, r_right):
            rs, hs = points[:,0], points[:,1]
            assert np.all(rs <= r_r) and np.all(rs >= r_l)
            mask = (r >= r_l) & (r < r_r)
            r_mask = r[mask] 

            h_mask = np.interp(r_mask, rs, hs)
            profile[mask] = h_mask
        
        profile = profile * (lens.aperture_field > 0)
        return profile, None
    
    @staticmethod 
    def kinoform_zone_placement(kinoform: Kinoform, direction:str="in") -> tuple[np.ndarray, np.ndarray | None]:
        profile = np.asarray(kinoform.profile)
        
        return profile, None
    
    @staticmethod
    def kinoform_sidewall_taper(kinoform: Kinoform, err: float | np.ndarray, proportion=1., zone_shift=False) -> tuple[np.ndarray, np.ndarray | None]:
        
        if zone_shift:
            errs = kinoform.add_error(LensErrors.zone_shift, err=err)
        else:
            errs = kinoform.add_error(LensErrors.zone_removal, err=err, m=0, extend=True, direction="in", remove_last=False, mutable=True)

        assert errs is not None


        r_start = np.asarray(kinoform.zone_right) #np.array(kinoform.zone_locations[:-1])
        r_end = r_start + proportion*errs
        
        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)
            
        height = kinoform.height 
        profile = np.asarray(kinoform.profile)
    
        if np.count_nonzero(errs) != 0: 
            for r1, r2 in zip(r_start, r_end):
                
                # ids = np.argsort(np.abs(r-r1)) # left and right from the symmetric zone boundaries
                # heights = profile.ravel()[ids]
                # sample = heights[heights > 0]
                # height = sample[0]
                
                mask = (r <= r2) & (r >= r1)
                r_mask = r[mask]
                taper = -height/(r2-r1)*(r_mask-r2)
                profile[mask] = taper

            kinoform.reshape(kinoform.zone_left, r_end)

        return profile, errs

    @staticmethod
    def kinoform_zone_warping(kinoform: Kinoform, R_min=None, R_max=None, beam_width=1e-8) -> tuple[np.ndarray, np.ndarray | None]:
        '''
        Warp thin outer zones toward an FZP-like rectangle by resampling them
        with only `ns = floor(zone_width / beam_width)` interior anchors, so
        that the linear interpolation done by `zone_quantization` collapses
        to a step as the zone approaches the beam width. Wider zones (below
        `tol`) are left untouched.

        Endpoint anchors are sampled from the current `kinoform.profile` at
        each zone's boundary, so no artificial 0/h endpoints are imposed.
        '''
        h = kinoform.height
        R = kinoform.R
        
        zone_widths = np.asarray(kinoform.zone_widths)
        
        r_left_all, r_right_all = np.asarray(kinoform.zone_left), np.asarray(kinoform.zone_right)#zone_locations[:-1], zone_locations[1:]
        
        m_total = int(np.ceil(kinoform.zones))
        ms = np.arange(m_total)
        
        if kinoform.dim == 1:
            r = np.abs(kinoform.grid)
        else:
            X, Y = kinoform.grid
            r = np.sqrt(X**2 + Y**2)
            
        if R_min is None: R_min = 0
        if R_max is None: R_max = R
        # mask1 = (zone_widths < max_width) & (zone_widths >= min_width)
        mask = (r_left_all < R_max) & (r_left_all >= R_min)
        
        # per-zone anchor count: thinner zones get fewer interior samples,
        # so the piecewise-linear fill flattens toward a rectangle.
        beam_width = min(beam_width, zone_widths[-1])
        ns = np.floor(zone_widths / beam_width).astype(int)
        # print(ns)

        profile = np.asarray(kinoform.profile)
        points = []
        ms_target = ms[mask]
        r_left, r_right = r_left_all[mask], r_right_all[mask]
        # print(ms_target)
        ## zone1 points
        for mi, r_l, r_r in zip(ms_target, r_left, r_right):
            n = max(ns[mi], 0)

            # interior anchor radii, evenly spaced strictly inside (r_l, r_r)
            xs = np.arange(1, int(n)+1)
            if n > 0:
                # interp n points between r_l and r_r
                interior_r = r_l + xs/(n+1) * (r_r - r_l)
            else:
                interior_r = np.empty(0)

            # restrict nearest-grid lookup to samples inside this zone so a
            # slightly-out-of-band grid point in the neighbouring zone can't
            # be chosen (which would leak a spurious spike from that zone).
            in_zone = (r.ravel() >= r_l) & (r.ravel() < r_r)
            zone_idx = np.flatnonzero(in_zone)
            zone_r = r.ravel()[zone_idx]
            zone_profile = profile.ravel()[zone_idx]

            def sample(rq):
                if zone_idx.size == 0:
                    return 0.0  # zone has no grid samples; fall back to 0
                # clamp query into the in-zone radius range for safety
                rq_c = np.clip(rq, zone_r.min(), zone_r.max())
                local = np.argmin(np.abs(zone_r - rq_c))
                return zone_profile[local]

            h_l = 0 #sample(r_l)
            h_r = h #sample(r_r)
            
            ## logistic sampling function for height
            # p in (0, 1]: fraction of the zone that a single beam spans.
            # p -> 0 (wide zones)  => mag -> 0  => uniform (linear) sampling.
            # p -> 1 (beam-limited) => mag -> inf => sigmoid becomes a step,
            # collapsing all interior anchors to a single radius for an FZP-like rectangle.
            width = zone_widths[mi]
            p = min(beam_width / width, 1.0)
            mag = p / (1.0 - p + 1e-12)
            # print("Mag:", mag)

            a = (r_l-r_r)/(0.5-1/(1+np.exp(-(int(n)+1)*mag)))
            b = r_l-a/2
 
            sampling_r = a/(1+np.exp(-mag*xs))+b
            interior_h = np.array([sample(rq) for rq in sampling_r])

            zone_pts = np.column_stack([
                np.concatenate([[r_l], interior_r, [r_r]]),
                np.concatenate([[h_l], interior_h, [h_r]]),
            ])
            points.append(zone_pts)

        if len(ms_target) == 0:
            return kinoform.profile, None
        # print(points)
        profile, _ = LensErrors.zone_quantization(kinoform, points, m=ms_target)
        return profile, None
        