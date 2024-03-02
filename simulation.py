import meep as mp
from material import Material  # Assuming you have a materials.py
import numpy as np 
from meep import Simulation 

# Meep unit scaling 
um_scale = 1e-6

def eV_to_meep_frequency(eV):  
    return eV / (241.798 * um_scale)

def init_simulation(geom, source_frequency):
    resolution = 10
    #materials, material_regions = geometry_params.get_material_regions()
    geometry = geom  
    padding = 0.1
    cell_size = mp.Vector3(0.5 + padding, 0.5 + padding, 1) 
    source_position = mp.Vector3(0, 0, 0.1)
    source_size = mp.Vector3(0.5, 0.5, 0)  # Adjusted for the size of the simulation cell
    symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]

    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.1 * source_frequency),
                        component=mp.Ez,
                        center=source_position,
                        size=source_size)]
    pml_layers = [mp.PML(0.1)]

    k_point = mp.Vector3(1,1,1)  
    simulation = mp.Simulation(cell_size=cell_size,
                               boundary_layers=pml_layers,
                               geometry=geometry,
                               sources=sources,
                               resolution=resolution,
                               symmetries=symmetries, 
                               k_point=k_point)
    return simulation

def calculate_field_enhancement(sim):
    max_field_z = np.max(np.abs(sim.get_array(component=mp.Ez)))
    max_field_y = np.max(np.abs(sim.get_array(component=mp.Ez)))
    max_field_x = np.max(np.abs(sim.get_array(component=mp.Ez)))

    return np.sqrt(max_field_z*10 + max_field_x*10 + max_field_y*10)
