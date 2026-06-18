import numpy as np
import xraylib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import functools, os

# plt.style.use('_mpl-gallery')

JOULE_TO_EV = 1/const.e
__all__ = ["SimulationObject", "Waveform", "Aperture", "ThinLens"]

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
        self.objects = dict()
        
    def __repr__(self):
        if self.dim == 1: # 2D box
            return(f"Simulation box centered at {self.center} with dimensions {self.Lx} m x {self.Lz} m")
        else: # 3D box
            return(f"Simulation box centered at {self.center} with dimensions {self.Lx} m x {self.Ly} m x {self.Lz} m")
        
    ## Setting up and using the simulation         
    def __check_collisions(self):
        pass
            
    def add_object(self, object):
        # Create check for object collisions
        match object:
            case Waveform():
                # if len(self.objects["sources"]) == 1: raise Exception("Only one source")
                self.objects["source"] = object
            case Aperture():
                self.objects["aperture"] = object
            case ThinLens():
                self.objects["lens"] = object
            case _:
                raise Exception("Unknown Object")
            
    def view(self):
        pass
            
class Object():
    def __init__(self, simulation: SimulationObject, z: float, func=None, **kwargs):
        # Associate object with provided simulation suite
        self.kwargs = kwargs
        self.simulation = simulation
        self.dim = simulation.dim
        simulation.add_object(self)
        # intialize physical properties
        self.center = np.zeros(simulation.dim+1)
        self.center[-1] = z
        
        if func is not None: self.func = func
            
        self._build_grid()
        # self.field = None
        self._build_field(**kwargs)
        
    def _build_grid(self):
        sim = self.simulation
        if sim.dim == 1:
            self.grid = np.linspace(-sim.Lx/2, sim.Lx/2, sim.Nx)
        else: # dim == 2
            x = np.linspace(-sim.Lx/2, sim.Lx/2, int(sim.Nx))
            y = np.linspace(-sim.Ly/2, sim.Ly/2, int(sim.Ny)) #type: ignore
            self.grid = np.meshgrid(x, y)
            
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
    
    def view(self, ax=None, xlim=None, ylim=None, savedir="", cmap="Greys_r", color="Black", show_cbar=False, extend=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if self.field is None: raise NotImplementedError
        
        ## initialize specs based on which object is being plotted
        if isinstance(self, Waveform):
            data = self.intensity()
            label="Intensity"
            scale="log"
            if extend:
                phase = self.phase()
                norm = colors.Normalize(vmin=phase.min(), vmax=phase.max())
                c_label="Phase"
            else:
                norm = colors.LogNorm(vmin=1e-4, vmax=data.max())
                c_label = label
        elif isinstance(self, ThinLens):
            data = self.phase()
            norm = colors.Normalize(vmin=data.min(), vmax=data.max())
            label = c_label = "Phase"
            scale="linear"
        else:
            data = self.field
            norm = colors.Normalize(vmin=0., vmax=data.max())
            label = c_label = ""
            scale="linear"
        
        ## plotting treatments based on dimension specified
        if self.simulation.dim == 2:
            if not extend:
                Lx, Ly = self.simulation.Lx, self.simulation.Ly
                im = ax.imshow(
                    data,
                    norm = norm,
                    cmap=cmap,
                    extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], #type: ignore
                )
                if show_cbar:
                    cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                                fraction=0.046, pad=0.08)

                    cbar.set_label(c_label)
                ax.set(xlabel="x [m]", ylabel="y [m]", xlim=xlim, ylim=ylim)
                used_ax = ax
            else:
                ss = ax.get_subplotspec()
                ax.remove()
                ax3d = fig.add_subplot(ss, projection="3d")
                ax3d.tick_params(axis='both', pad=2)
                X, Y = self.grid
                surf = ax3d.plot_surface(X, Y, data, norm=norm, cmap=cmap)
                ax3d.set(xlabel="x [m]", ylabel="y [m]", zlabel=label)
                if show_cbar:
                    cbar = fig.colorbar(surf, ax=ax3d, orientation='vertical',
                                shrink=0.7, pad=0.12)
                    cbar.set_label(c_label)
                used_ax = ax3d
        else:
            ax.plot(self.grid, data, color=color)
            ax.set(xlabel="x [m]", ylabel=label, yscale=scale, xlim=xlim, ylim=ylim)
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
        
    def propagate(self, z, propagation_func):
        if z+self.center[-1] > self.simulation.Lz: raise Exception(f"Propagation must stay within box length {self.simulation.Lz}")
        U = propagation_func(self.field, z, self.simulation, self.wavelength)
        self.field = U
        self.center[-1] += z
        
    def intensity(self):
        I = np.abs(self.field)**2
        return I
    
    def phase(self):
        field = np.where(self.field != 0, self.field, 0.)
        return np.angle(field)

        
class Aperture(Object):
    def __init__(self, simulation: SimulationObject, z, func=None, **kwargs):
        super().__init__(simulation, z, func=func, **kwargs)
        
    def __repr__(self):
        return f"Thin aperture located at {self.center}"
        
    def transform(self, wave: Waveform):
        wave.field *= self.field
        
class ThinLens(Aperture):
    
    def __init__(self, f, aperture_func, simulation: SimulationObject, z:float, thickness_func = None, n =1.0, **kwargs):
        '''
        args
        - f: focal length [m]
        - aperture_func: lens aperture function over grid
        '''
        self.f = f
        self.n = n
        self.aperture = aperture_func
        if thickness_func is not None: self.thickness = thickness_func # type: ignore
          
        super().__init__(simulation, z, func=aperture_func, **kwargs)
        self.aperture_field = self.field
        
        if self.simulation.dim == 1:
            self.build_profile(self.grid, **kwargs)
        else:
            self.build_profile(*self.grid, **kwargs)
        
        self._transmittance_initialized = False
             
    def __repr__(self):
        return f"Lens located at {self.center} with focal length {self.f}"
    
    ## lens profile
    def thickness(self, *args, **kwargs):
        raise NotImplementedError("Lens must have a thickness profile!")
    
    def build_profile(self, *args, **kwargs):
        self.profile = self.aperture_field * self.thickness(*args, **kwargs)
        self.orig_profile=self.profile
        
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
        base = self.field.astype(np.complex128)
        self.field = base * t_l(wave.wavelength)
        self._transmittance_initialized = True
        
    def transform(self, wave: Waveform):
        if not self._transmittance_initialized:
            self.init_transmittance(wave)
            self._transmittance_initialized = True
        
        wave.field = wave.field.astype(np.complex128)*self.field
        
    ## errors and approximations
    def quantization(self, N: int):
        '''
        args 
        - N: quantization steps
        '''
        
        return
    
    def add_error(self, error_func, **kwargs):
        self.profile = error_func(self, **kwargs)
    
    ## plotting
    def phase(self):
        field = np.where(self.field != 0, self.field, 0.)
        return np.angle(field)
        
    def plot_profile(self):
        raise NotImplementedError
    
    