import numpy as np
import xraylib
import matplotlib.pyplot as plt
import scipy.constants as const
import functools, os

JOULE_TO_EV = 1/const.e
__all__ = ["SimulationObject", "Waveform", "Aperture", "Lens"]

class SimulationObject:
    
    def __init__(self, Lx, Nx, Lz, Ly=None, Ny=None, n=1):
        if Ly is None and Ny is None:
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
            case Lens():
                self.objects["lens"] = object
            case _:
                raise Exception("Unknown Object")
            
    def view(self):
        pass
            
class Object():
    def __init__(self, simulation: SimulationObject, z: float, **kwargs):
        # Associate object with provided simulation suite
        self.simulation = simulation
        simulation.add_object(self)
        # intialize physical properties
        self.center = np.zeros(simulation.dim+1)
        self.center[-1] = z
        
        func = kwargs.get("func", None)
        if func is not None:
            self.func = func
            
        # print("Dimensions:", simulation.dim)
        if simulation.dim == 1:
            Lx = simulation.Lx
            Nx = simulation.Nx
            X = np.linspace(-Lx/2, Lx/2, Nx)
            self.grid = X
            self.field = self.func(X, z, **kwargs)
        elif simulation.dim == 2:
            Lx, Ly = simulation.Lx, simulation.Ly
            Nx, Ny = simulation.Nx, simulation.Ny
            
            assert Ly is not None and Ny is not None
            x, y = np.linspace(-Lx/2, Lx/2, int(Nx)), np.linspace(-Ly/2, Ly/2, int(Ny))
            X, Y = np.meshgrid(x, y)
            self.grid = (X, Y)
            self.field = self.func(X, Y, z, **kwargs)   
        else:
            raise Exception
        
    def func(self, *args, **kwargs):
        raise NotImplementedError
    
    def view(self, ax=None, xlim=None, ylim=None, savedir="", cmap="Greys_r", show_label=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if isinstance(self, Waveform):
            data = self.intensity()
        elif isinstance(self, Lens):
            data = self.angle()
        else:
            data = self.field

            
        if self.simulation.dim == 2:
            Lx, Ly = self.simulation.Lx, self.simulation.Ly
            im = ax.imshow(
                data,
                cmap=cmap,
                extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], #type: ignore
                vmin=0.0,
                vmax=1.0
            )
            if show_label:
                fig.colorbar(im, ax=ax, orientation='vertical',
                            fraction=0.02, pad=0.04, extend='max')
            ax.set(xlabel="x [m]", ylabel="y [m]", xlim=xlim, ylim=ylim)
        else:
            ax.plot(self.grid, data)
            ax.set(xlabel="x [m]", xlim=xlim)

        if savedir:
            plt.savefig(os.path.join(savedir, f"{type(self).__name__}_z={self.center[-1]}.png"))

        
class Waveform(Object):
    def __init__(self, energy: float, simulation: SimulationObject, z: float, **kwargs):
        '''
        energy: energy of waveform in eV
        '''
        self.energy = energy # [eV]
        self.wavelength =  (const.h * const.c) * JOULE_TO_EV / energy # [m]
        self.frequency = const.c / self.wavelength # [Hz]
        
        super().__init__(simulation, z, **kwargs) #functools.partial(distribution_func, wavelength=self.wavelength, n=simulation.n)) # type: ignore
        
    def __repr__(self):
        return f"{self.simulation.dim}-D Waveform with energy {self.energy:.3e} eV at {self.center}"
        
    def propagate(self, z, propagation_func):
        if z+self.center[-1] > self.simulation.Lz: raise Exception(f"Propagation must stay within box length {self.simulation.Lz}")
        U = propagation_func(self.field, z, self.simulation, self.wavelength)
        self.field = U
        self.center[-1] += z
        
    def intensity(self):
        return np.abs(self.field)**2
        
class Aperture(Object):
    def __init__(self, simulation: SimulationObject, z, **kwargs):
        super().__init__(simulation, z, **kwargs)
        
    def __repr__(self):
        return f"Thin aperture located at {self.center}"
        
    def transform(self, wave: Waveform):
        wave.field *= self.field
        
    
class Lens(Aperture):
    
    def __init__(self, f, aperture_func, simulation: SimulationObject, z:float, **kwargs):
        '''
        args
        - f: focal length [m]
        
        '''
        self.f = f
        
        # func = lambda *args, **kwargs: aperture.func(*args, **kwargs)*transmittance_func(*args, **kwargs)
        super().__init__(simulation, z, **kwargs)
        self.field *= aperture_func(*self.grid)
        # print(self.field)
        
        # self.d = d
        # self.R1 = R1
        # self.R2 = R2
        # self.f = 1/((n-1)*(1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))) # Lens maker equation
        
    def __repr__(self):
        return f"Lens located at {self.center} with focal length {self.f}"
    
    def angle(self):
        return np.angle(self.field)
        
    def side_view(self):
        pass