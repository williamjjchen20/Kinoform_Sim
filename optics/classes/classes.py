import numpy as np
import xraylib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import scipy.constants as const
import functools, os
from abc import abstractmethod


__all__ = ["SimulationObject", "Propagator", "Waveform", "Aperture", "ThinLens"]

JOULE_TO_EV = 1/const.e
        
class SimulationObject:
    
    def __init__(self, Lx, Nx, Lz, Ly=None, Ny=None, n=1):
        if Ly is None or Ny is None:
            self.dim = 1
        elif Ly is not None and Ny is not None:
            self.dim = 2
        else:
            raise Exception("Ly and Ny must be defined!")

        self.Lx = Lx
        self.Nx = Nx
        self.dx = Lx/Nx
        self.Ly = Ly
        self.Ny = Ny
        self.dy = Ly/Ny if Ly is not None and Ny is not None else None
        self.Lz = Lz   
           
        self.center = np.zeros(self.dim+1)
        self.center[-1] = Lz/2
        self.n = n
        
        ## Add in objects (sources, apertures, lenses)
        self.propagator = None
        self.objects = dict()
        self.zs = []
        
    def __repr__(self):
        if self.dim == 1: # 2D box
            return(f"Simulation box centered at {self.center} with dimensions {self.Lx} m x {self.Lz} m")
        else: # 3D box
            return(f"Simulation box centered at {self.center} with dimensions {self.Lx} m x {self.Ly} m x {self.Lz} m")
        
    ## Setting up and using the simulation         
    def check_collisions(self):
        objects = self.objects
        source = objects.get("source", None)
        if source is None: raise Exception("No source defined.")
        
        lens_locations = np.array([lens.z for lens in objects.get("lens", [])])
        aperture_locations = np.array([aperture.z for aperture in objects.get("aperture", [])])
        
        concat = np.concatenate([lens_locations, aperture_locations])
        unique, counts = np.unique(np.sort(concat), return_counts=True)
        
        # check for aperture position collisions
        if len(concat) > len(unique):
            repeated = unique[counts > 1]
            raise Exception(f"Collision Detected at z={repeated}")
        # check for invalid propagation distances
        if np.all(unique[1:]-unique[:-1] < 10*source.wavelength):
            raise Exception("Objects are located too close for precision.")
        # Source position
        if np.any(source.z >= unique):
            raise Exception("Source should be leftmost object.")
        else:
            print("No collisions detected. Proceeding...")
            return
        
    def add_object(self, object):
        # Create check for object collisions
        match object:
            case Waveform():
                # if len(self.objects["sources"]) == 1: raise Exception("Only one source")
                self.objects["source"] = object
            case ThinLens():
                lenses = self.objects.get("lens", [])
                lenses.append(object)
                self.objects["lens"] = lenses
            case Aperture():
                apertures = self.objects.get("aperture", [])
                apertures.append(object)
                self.objects["aperture"] = apertures
            case _:
                raise Exception("Unknown Object")
        self.zs.append(object.z)
        
    def add_propagator(self, propagator):
        self.propagator = propagator
        
    def copy(self, dim=None):
        if dim is None: dim = self.dim
        
        # 2D -> 3D
        if dim == 2 and self.dim == 1:
            sim = SimulationObject(Lx=self.Lx, Nx=np.sqrt(self.Nx), Lz=self.Lz, Ly=self.Lx, Ny=np.sqrt(self.Nx), n=self.n)
        # 3D -> 2D
        elif dim == 1 and self.dim == 2:
            sim = SimulationObject(Lx=self.Lx, Nx=(self.Nx)**2//2, Lz=self.Lz, Ly=None, Ny=None, n=self.n) 
        # xD -> xD   
        else:
            sim = SimulationObject(Lx=self.Lx, Nx=self.Nx, Lz=self.Lz, Ly=self.Ly, Ny=self.Ny, n=self.n)
            
        return sim
    
    def view(self):
        pass
    
class Propagator():
    def __init__(self, propagation_func, simulation: SimulationObject, **kwargs):
        self.dim=simulation.dim
        self.propagator = functools.partial(propagation_func, **kwargs)
        self.validated=False
        simulation.add_propagator(self)

    @abstractmethod
    def _conditions(self):
        pass
            
class Object():
    def __init__(self, simulation: SimulationObject, z: float, func=None, **kwargs):
        # Associate object with provided simulation suite
        self.kwargs = kwargs
        self.simulation = simulation
        self.dim = simulation.dim
        # intialize physical properties
        self.z = z
        self.center = np.zeros(simulation.dim+1)
        self.center[-1] = z
        
        if func is not None: self.func = func
            
        self._build_grid()
        # self.field = None
        self._build_field(**kwargs)
        
        simulation.add_object(self)
        
    def _build_grid(self):
        sim = self.simulation
        if sim.dim == 1:
            self.grid = (np.arange(sim.Nx)-sim.Nx/2)*sim.dx#np.linspace(-sim.Lx/2, sim.Lx/2, sim.Nx)
        else: # dim == 2
            x = (np.arange(sim.Nx)-sim.Nx/2)*sim.dx
            y = (np.arange(sim.Ny)-sim.Ny/2)*sim.dy #type: ignore
            self.grid = np.meshgrid(x, y)
        self.Rx, self.Ry = 1., 1.

    def _build_field(self, **kwargs):
        z = self.center[-1]
        if self.simulation.dim == 1:
            self.field = self.func(self.grid, z=z,**kwargs)
        else: # dim == 2
            self.field = self.func(*self.grid, z=z,**kwargs)

    def func(self, *args, **kwargs):
        raise NotImplementedError("Distribution function has not been provided!")
    
    def add_error(self, error_func):
        self.error = error_func 
    
    def view(self, ax=None, savedir="", labels: dict | None = None, show_cbar=False, _3d=False, **kwargs):
        
        labels = dict(labels or {})
        cmap = labels.get("cmap", "Greys_r")
        color = labels.get("color", "Black")
        _phase_cmap = labels.get("phase_cmap", "hsv")

        def _view_data(self):
            '''
            Return (data, norm, scale, label, c_label, sm, rgb) for `view`.
            - data: scalar field for plotting (intensity / phase / field).
            - norm: matplotlib Normalize for `data`.
            - label: axis label for the data quantity.
            - c_label: colorbar label.
            - sm: optional ScalarMappable used for the colorbar (overrides imshow).
            - rgb: optional precomputed HxWx3 array for domain coloring (2D Waveform).
            '''
            sm = None
            rgb = None
            if isinstance(self, Waveform):
                data = self.intensity()
                label = "Intensity"
                c_label = label
                phase = self.phase()
                
                if self.dim != 1:
                    # domain coloring: |field| -> lightness (log), phase -> cyclic cmap
                    vmin = 1e-4*data.max() if data.max() > 0 else 0.0
                    vmax = data.max() if data.max() > 0 else 1.0
                    lnorm = colors.Normalize(vmin=vmin, vmax=vmax)
                    lv = lnorm(data)
                    v = np.clip(lv.filled(0) if hasattr(lv, "filled") else lv, 0, 1)
                    hue = (phase + np.pi)/ (2*np.pi)
                    rgb_phase = plt.get_cmap(_phase_cmap)(hue)[..., :3]
                    rgb = rgb_phase * v[..., None]
                    norm = lnorm
                    sm = cm.ScalarMappable(norm=lnorm, cmap="Greys_r")
                else:
                    norm = sm = rgb = None
                    
            elif isinstance(self, ThinLens):
                data = self.phase()
                norm = colors.Normalize(vmin=data.min(), vmax=data.max())
                label = c_label = "Phase"
            else:
                data = self.field
                norm = colors.Normalize(vmin=0., vmax=data.max())
                label = c_label = ""
            return data, norm, label, c_label, sm, rgb
        
        def _add_phase_wheel(host_ax, size=0.22, pad=0.1,
                             cmap=_phase_cmap, text_color="white"):
            '''Add a polar phase color wheel anchored at the top-right of `host_ax`.'''
            # bbox in host_ax coords: top-right corner, size x size of the axes
            bbox = (1.0 - size - pad, 1.0 - size - pad, size, size)
            wax = host_ax.inset_axes(bbox, projection="polar")
            theta = np.linspace(0, 2*np.pi, 360)
            r = np.linspace(0.5, 1.0, 2)
            T, _ = np.meshgrid(theta, r)
            wax.pcolormesh(theta, r, (T + np.pi) % (2*np.pi), cmap=cmap, vmin=0, vmax=2*np.pi, shading="auto")
            wax.set_yticks([])
            wax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            wax.set_xticklabels(["0", "π/2", "±π", "-π/2"], color=text_color)
            wax.tick_params(colors=text_color, pad=-2)
            for spine in wax.spines.values():
                spine.set_edgecolor(text_color)
            wax.patch.set_alpha(0.0)
            return wax
            
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if self.field is None: raise NotImplementedError

        ## decide what scalar data, norm, and colorbar mappable to use per type
        data, norm, label, c_label, sm, rgb = _view_data(self)

        xscale = labels.get("xscale", "linear")
        yscale = labels.get("yscale", "linear")
        x_scale_factor = labels.get("x_scale_factor", 1.0)
        y_scale_factor = labels.get("y_scale_factor", 1.0)
        xlabel = labels.get("xlabel", "x [m]")
        ylabel = labels.get("ylabel", "y [m]" if self.simulation.dim == 2 else label)
        title  = labels.get("title", f"{type(self).__name__} View")
        xlim = labels.get("xlim", None)
        ylim = labels.get("ylim", None)

        if xlim is not None: xlim = np.array(xlim) * x_scale_factor
        if ylim is not None: ylim = np.array(ylim) * y_scale_factor
    
        ## plotting treatments based on dimension specified
        if self.simulation.dim == 2:
            # `extent` in `labels` overrides the default (full simulation box).
            # Provided in unscaled units; x_scale_factor / y_scale_factor are applied here.
            Lx, Ly = self.simulation.Lx, self.simulation.Ly
            default_extent = [-Lx/2, Lx/2, -Ly/2, Ly/2] # type: ignore
            raw_extent = labels.get("extent", default_extent)
            extent = [raw_extent[0]*x_scale_factor, raw_extent[1]*x_scale_factor,
                      raw_extent[2]*y_scale_factor, raw_extent[3]*y_scale_factor]

            if not _3d:
                # if we have a precomputed RGB (domain coloring), display it directly
                im = ax.imshow(rgb if rgb is not None else data,
                               norm=None if rgb is not None else norm,
                               cmap=None if rgb is not None else cmap,
                               extent=extent, origin="lower") #type: ignore

                if show_cbar:
                    mappable = sm if sm is not None else im
                    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical",
                                        fraction=0.046, pad=0.08)
                    cbar.set_label(c_label)
                    if rgb is not None:
                        _add_phase_wheel(ax, cmap=_phase_cmap)

                ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
                used_ax = ax
            else:
                ss = ax.get_subplotspec()
                ax.remove()
                ax3d = fig.add_subplot(ss, projection="3d")
                ax3d.tick_params(axis="both", pad=2)

                X, Y = self.grid
                X, Y = X*x_scale_factor, Y*y_scale_factor
                mask_x = np.ones(X.shape[1], dtype=bool) if xlim is None \
                    else (X[0, :] >= xlim[0]) & (X[0, :] <= xlim[1])
                mask_y = np.ones(Y.shape[0], dtype=bool) if ylim is None \
                    else (Y[:, 0] >= ylim[0]) & (Y[:, 0] <= ylim[1])
                # downsample dense grids: per-facet coloring is slow at large N
                stride = max(1, min(mask_x.sum(), mask_y.sum()) // 256)
                idx = np.ix_(np.flatnonzero(mask_y)[::stride],
                             np.flatnonzero(mask_x)[::stride])
                Xc, Yc, Dc = X[idx], Y[idx], data[idx]

                if rgb is not None:
                    # height = intensity; color = phase domain-coloring
                    surf = ax3d.plot_surface(
                        Xc, Yc, Dc, facecolors=rgb[idx],
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False,
                    )
                else:
                    surf = ax3d.plot_surface(Xc, Yc, Dc, norm=norm, cmap=cmap)

                if show_cbar:
                    mappable = sm if sm is not None else surf
                    cbar = fig.colorbar(mappable, ax=ax3d, orientation="vertical",
                                        shrink=0.7, pad=0.12)
                    cbar.set_label(c_label)
                    if rgb is not None:
                        _add_phase_wheel(ax3d, cmap=_phase_cmap, text_color="Black")

                if xlim is not None: ax3d.set_xlim3d(*xlim)
                if ylim is not None: ax3d.set_ylim3d(*ylim)
                ax3d.set_box_aspect((1, 1, 0.6))
                ax3d.set(xlabel=xlabel, ylabel=ylabel, title=title, zlabel=label)
                used_ax = ax3d
        else:
            ax.plot(self.grid*x_scale_factor*self.Rx, data*y_scale_factor, color=color)
            ax.set(xlabel=xlabel, ylabel=label, xscale=xscale, yscale=yscale, title=title,
                   xlim=xlim, ylim=ylim)
            used_ax = ax

        if savedir:
            fig.savefig(os.path.join(savedir, f"{type(self).__name__}_z={self.center[-1]}.png"))
        return used_ax

class Waveform(Object):
    def __init__(self, energy: float, simulation: SimulationObject, z: float, func=None, **kwargs):
        '''
        energy: energy of waveform in eV
        '''
        self.energy = energy # [eV]
        self.wavelength =  (const.h * const.c) * JOULE_TO_EV / energy # [m]
        self.frequency = const.c / self.wavelength # [Hz]
        
        super().__init__(simulation, z, func=func, **kwargs) #functools.partial(distribution_func, wavelength=self.wavelength, n=simulation.n)) # type: ignore
        
    def __repr__(self):
        return f"{self.simulation.dim}-D Waveform with energy {self.energy:.3e} eV at {self.center}"
    
    def copy(self, dim=None):
        return Waveform(self.energy, self.simulation.copy(dim=None), z=self.z)
        
    def propagate(self, dz, propagator: Propagator):
        if dz+self.z > self.simulation.Lz: 
            print(f"Propagation must stay within box length {self.simulation.Lz}")
            diff = dz+self.z - self.simulation.Lz
            dz = np.round(diff,10)
            print(f"Propagating for {dz}.")
        propagation_func = propagator.propagator

        self.field = propagation_func(self, dz)
        self.z += dz
        
    def intensity(self):
        I = np.abs(self.field)**2
        return I
    
    def phase(self):
        field = np.where(self.field != 0, self.field, 0.)
        return np.angle(field)
    
    def filter(self, aperture): #type: ignore
        match aperture:
            case ThinLens(): 
                self.field *= aperture.aperture_field
            case Aperture():
                self.field *= aperture.field
            case _:
                raise Exception("Invalid aperture!")
        
class Aperture(Object):
    def __init__(self, simulation: SimulationObject, z, func=None, **kwargs):
        super().__init__(simulation, z, func=func, **kwargs)
        
    def __repr__(self):
        return f"Thin aperture located at {self.center}"
    
    def copy(self, dim=None):
        return Aperture(self.simulation.copy(dim=dim), z=self.z, func=self.func, **self.kwargs)
        
    def transform(self, wave: Waveform):
        wave.field *= self.field
        
class ThinLens(Aperture):
    
    def __init__(self, f, aperture_func, simulation: SimulationObject, z:float, thickness_func=None, n=1.0, **kwargs):
        '''
        args
        - f: focal length [m]
        - aperture_func: lens aperture function over grid
        '''
        self.f = f
        self.n = n
        self.orig_aperture = aperture_func
        if thickness_func is not None: self.thickness = thickness_func
        
        # self._init_aperture(aperture_func, **kwargs)
          
        super().__init__(simulation, z, func=aperture_func, **kwargs)
        self.aperture_field = np.array(self.field)
    
        self.build_profile(**kwargs)        
        self._transmittance_initialized = False
             
    def __repr__(self):
        return f"Lens located at {self.center} with focal length {self.f}"
    
    ## lens profile
    def thickness(self, *args, **kwargs):
        raise NotImplementedError("Lens must have a thickness profile!")
    
    def build_profile(self, **kwargs):
        try:
            if self.simulation.dim == 1:
                t = self.thickness(self.grid, **kwargs)
            else:
                t = self.thickness(*self.grid, **kwargs)
        except:
            raise Exception("Warning: Invalid/no thickness profile provided.")
            
        self.profile = self.aperture_field * t
        # self.orig_profile=np.array(self.profile)
        
    def reshape(self, aperture_func):
        print("Warning! Resetting changes...")
        super().__init__(self.simulation, self.z, func=aperture_func, **self.kwargs)
        self.aperture_field = np.array(self.field)
    
        self.build_profile(**self.kwargs)        
        self._transmittance_initialized = False
        self.mutated = True
        
    ## transmittance features
    def transmittance(self, wavelength):
        k = 2*const.pi/wavelength
        n = self.n
        t = np.exp(1j*k*(n-1.)*self.profile)
        return t
    
    ## implements lens phase screen          
    def init_transmittance(self, wave: Waveform):
        assert(not self._transmittance_initialized)
        t_l = self.transmittance
        base = self.aperture_field.astype(np.complex128)
        self.field = base * t_l(wave.wavelength)
        self._transmittance_initialized = True
        
    def transform(self, wave: Waveform):
        if not self._transmittance_initialized:
            print("Warning: Transmittance was not initialized...")
            self.init_transmittance(wave)
            self._transmittance_initialized = True

        wave.field = wave.field.astype(np.complex128)*self.field
    
    ## error functions
    def add_error(self, error_func, **kwargs):
        profile, err = error_func(self, **kwargs)
        self.profile = profile
        return err
        
    def reset(self):
        super().__init__(self.simulation, self.z, func=self.orig_aperture, **self.kwargs)
        self.build_profile(**self.kwargs)        
        self._transmittance_initialized = False
        
    ## plotting
    def phase(self):
        field = np.where(self.field != 0, self.field, 0.)
        return np.angle(field)
        
    @abstractmethod
    def plot_profile(self):
        raise NotImplementedError
    
    