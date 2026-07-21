from ..optics import *
import time
## Building functionalities 
def build_simulation(N, Lx, Ly = None, Lz=1000):
    return SimulationObject(Lx=Lx, Nx=N, Lz=Lz, Ly=Ly, Ny=N)

# simulation checks
def rayleigh(wavelength, f, D):
    return wavelength * f/D

def run(simulation: SimulationObject, z):
    print("Running simulation...")
    # verify simulation setup
    simulation.check_collisions()
    z = min(simulation.Lz, z)
    
    dim = simulation.dim
    propagator : Propagator = simulation.propagator # type:ignore

    objects = simulation.objects
    lens = np.array(objects.get("lens", []))
    aperture = np.array(objects.get("aperture", []))
    prop_objects = np.concatenate([lens, aperture])
    
    locations = np.array([obj.z for obj in prop_objects])
    order = np.argsort(locations)
    prop_objects = prop_objects[order]

    locations = np.concatenate([[objects["source"].z], locations[order]])
    
    mask = locations <= z()
    locations = locations[mask]
    dzs = np.round(locations[1:] - locations[:-1], 10)
    assert len(dzs) <= len(prop_objects)

    t_s = time.time()
    for obj, dz in zip(prop_objects, dzs):
        print(f"Propagating by {dz} m")
        source.propagate(dz, propagator)
        obj.transform(source)
        print(f"    {type(obj).__name__} Reached.")
    
    dz = np.round(z-locations[-1], 10)
    print(f"Propagating to {z} m")
    source.propagate(dz, propagator)
        
    t_e = time.time()
    print(f"Finished running in {np.abs(t_e-t_s):3f} s")
    

if __name__ == "__main__":
    
    E, f, R = 8e3, 0.1, 1e-4
    # E, f, R = 8e3, 1.0, 5e-5
    Lx, N = 3e-4, 100000
    Lz = 2*f
    n = xrl.Refractive_Index("Si", E / 1000, 2.329)

    sim = SimulationObject(Lx=Lx, Lz=Lz, Nx=N, Ny=N)
    propagator = AngularSpectrum(simulation=sim)
    source = GaussianBeam(energy=E, simulation=sim, z=0, w0=R)

    lens = Kinoform(wavelength=source.wavelength, f=f, R=R, n=n,
                    simulation=sim, z=f/2)
    lens = SingleSlit(sim, z=0.9*f, width=R)
    
    run(sim, Lz)