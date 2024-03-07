import meep as mp
from material import make_gold_material 

def eV_to_meep_frequency(eV):
    return eV / (241.798 * 1e-6)  

def init_simulation(geometry, cell_size, source_frequency):
    resolution = 20  
    padding = 0.1

    source_position = mp.Vector3(0, 0, 0.1)
    source_size = mp.Vector3(0.5, 0.5, 0)
    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.1 * source_frequency),
                         component=mp.Ez,  
                         center=source_position,
                         size=source_size)]

    symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]
    pml_layers = [mp.PML(0.1)]

    simulation = mp.Simulation(cell_size=cell_size,
                               boundary_layers=pml_layers,
                               geometry=geometry,  
                               sources=sources,
                               resolution=resolution,
                               symmetries=symmetries) 
    return simulation

def run_simulation(geometry, cell_size):
    source_frequency = eV_to_meep_frequency(2.34)  
    simulation = init_simulation(geometry, cell_size, source_frequency)
    simulation.run(until_after_sources=mp.stop_when_fields_decayed(
                   50, mp.Ez, mp.Vector3(), 1e-6)) 
    field = simulation.get_array(component=mp.Ez)
    return field  
