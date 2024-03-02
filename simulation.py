import meep as mp
from material import Material  # Assuming you have a materials.py

# Meep unit scaling 
um_scale = 1e-6

def eV_to_meep_frequency(eV):  
    return eV / (241.798 * um_scale)

def init_simulation(geometry_params, source_frequency):
    resolution = 20
    materials, material_regions = geometry_params.get_material_regions()
    geometry = materials  
    source_position = mp.Vector3(0, 0, 0.1)
    source_size = mp.Vector3(0.5, 0.5, 0)  # Adjusted for the size of the simulation cell

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
    return sim

def calculate_field_enhancement(sim):
    field = sim.get_array(component=mp.Ez) 
    return np.max(np.abs(field)) 
def save_fields(sim):
    sim.output_efield_z('ez')