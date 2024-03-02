import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def eV_to_meep_frequency(eV, a=1e-6):
    return eV / (a * 241.798)

def make_gold_material():
    um_scale = 1.0
    eV_um_scale = um_scale / 1.23984193  # Scale factor for converting eV to um^-1
    metal_range = mp.FreqRange(min=um_scale / 6.1992, max=um_scale / 0.24797)

    # Plasma frequency and Drude model parameters for gold
    Au_plasma_frq = 9.03 * eV_um_scale
    Au_susc = [mp.DrudeSusceptibility(frequency=1e-10, gamma=0.053 * eV_um_scale, sigma=Au_plasma_frq**2 * 0.760)]
    Au_susc += [mp.LorentzianSusceptibility(frequency=f * eV_um_scale, gamma=g * eV_um_scale, sigma=Au_plasma_frq**2 * f_strength / (f * eV_um_scale)**2)
                for f, g, f_strength in ((0.415, 0.241, 0.024), (0.830, 0.345, 0.010), (2.969, 0.870, 0.071), (4.304, 2.494, 0.601), (13.32, 2.214, 4.384))]

    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, valid_freq_range=metal_range)

def init_simulation(source_frequency):
    cell_size = mp.Vector3(7, 7, 0)
    pml_layers = [mp.PML(1.0)]
    resolution = 200
    gold = make_gold_material()

    triangle_base = 1.0
    triangle_height = 2.0
    gap = 0.1
    vertices1 = [mp.Vector3(-triangle_base/2 - gap/2, -triangle_height/2),
                 mp.Vector3(-triangle_base/2 - gap/2, triangle_height/2),
                 mp.Vector3(-gap/2, 0)]
    vertices2 = [mp.Vector3(triangle_base/2 + gap/2, -triangle_height/2),
                 mp.Vector3(triangle_base/2 + gap/2, triangle_height/2),
                 mp.Vector3(gap/2, 0)]
    geometry = [mp.Prism(vertices1, height=mp.inf, material=gold),
                mp.Prism(vertices2, height=mp.inf, material=gold)]

    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.2),
                         component=mp.Ez,
                         center=mp.Vector3(0, 3),
                         size=mp.Vector3(cell_size.x, 0, 0))]

    # Initialize simulation
    return mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=geometry, sources=sources, resolution=resolution)

def measure_baseline_field_intensity(simulation):
    """Runs a short simulation to measure the baseline field intensity at the source location."""
    simulation.run(until=20)
    ez_data = simulation.get_array(component=mp.Ez, center=mp.Vector3(0, 3), size=mp.Vector3(0, 0, 0))
    return np.abs(ez_data)

def plot_field_magnitude_heatmap(simulation):
    # Define simulation domain dimensions
    sim_size = simulation.cell_size
    resolution = simulation.resolution
    
    # Extract the field components
    Ex = simulation.get_array(component=mp.Ex, size=sim_size, center=mp.Vector3())
    Ey = simulation.get_array(component=mp.Ey, size=sim_size, center=mp.Vector3())
    Ez = simulation.get_array(component=mp.Ez, size=sim_size, center=mp.Vector3())
    
    # Compute the magnitude of the electric field
    E_magnitude = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(E_magnitude.transpose(), interpolation='spline36', cmap='inferno', extent=[-sim_size.x/2, sim_size.x/2, -sim_size.y/2, sim_size.y/2], origin='lower')
    plt.colorbar(label='|E| Field Magnitude')
    plt.title('Electric Field Magnitude Heatmap')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.show()

source_frequency = eV_to_meep_frequency(2.34)
simulation = init_simulation(source_frequency)
e0_intensity = measure_baseline_field_intensity(simulation)
simulation.reset_meep()  
plot_field_heatmap(simulation)
