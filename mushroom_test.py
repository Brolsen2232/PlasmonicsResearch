import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def eV_to_meep_frequency(eV, a=1e-6):
    return eV / (a * 241.798)


def make_gold_material():
    um_scale = 1.0
    eV_um_scale = um_scale / 1.23984193
    metal_range = mp.FreqRange(min=um_scale / 6.1992, max=um_scale / 0.24797)

    Au_plasma_frq = 9.03 * eV_um_scale
    Au_gamma = 0.053 * eV_um_scale
    Au_sigma = Au_plasma_frq**2 * 0.760

    Au_susc = [mp.DrudeSusceptibility(frequency=1e-10, gamma=Au_gamma, sigma=Au_sigma)]

    interband_transitions = [
        (0.415, 0.241, 0.024),
        (0.830, 0.345, 0.010),
        (2.969, 0.870, 0.071),
        (4.304, 2.494, 0.601),
        (13.32, 2.214, 4.384)
    ]
    for f, g, s in interband_transitions:
        sigma = Au_plasma_frq**2 * s / (f * eV_um_scale)**2
        Au_susc.append(mp.LorentzianSusceptibility(frequency=f * eV_um_scale, gamma=g * eV_um_scale, sigma=sigma))

    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, valid_freq_range=metal_range)


def init_simulation(source_frequency):
    cell_size = mp.Vector3(4, 4, 0)
    resolution = 150
    gold = make_gold_material()

    stalk_height = 0.5
    stalk_radius = 0.2
    cap_radius = 0.1
    mushroom1_center = mp.Vector3(1.001, 0)
    mushroom2_center = mp.Vector3(-1.001, 0)

    geometry = [
        mp.Cylinder(height=stalk_height, radius=stalk_radius, center=mushroom1_center, material=gold),
        mp.Sphere(radius=cap_radius, center=mushroom1_center + mp.Vector3(z=stalk_height/2 + cap_radius/2), material=gold),
        mp.Cylinder(height=stalk_height, radius=stalk_radius, center=mushroom2_center, material=gold),
        mp.Sphere(radius=cap_radius, center=mushroom2_center + mp.Vector3(z=stalk_height/2 + cap_radius/2), material=gold)
    ]

    source_position = mp.Vector3(-1.5 + stalk_radius + 0.1, 0, stalk_height / 2)
    source_size = mp.Vector3(0.1, 0.1, 0)  # A smaller, more localized source

    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.1 * source_frequency),
                        component=mp.Ez,
                        center=source_position,
                        size=source_size)]

    pml_layers = [mp.PML(1.0)]

    # Use an adaptive grid refinement strategy
    refinement_region = mp.Box(size=mp.Vector3(2, 2, 2), center=source_position)
    refinement_resolution = 300

    simulation = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=geometry, sources=sources, resolution=resolution,
                               refinement_region=refinement_region, refinement_resolution=refinement_resolution)

    return simulation

def draw_structure(ax, center, stalk_height, stalk_radius, cap_radius):
    # Draw stalk
    #stalk_bottom = center[1] - stalk_height / 2
    #stalk_top = center[1] + stalk_height / 2
    #ax.plot([center[0] - stalk_radius, center[0] + stalk_radius], [stalk_bottom, stalk_bottom], 'k-', linewidth=2)
    #ax.plot([center[0] - stalk_radius, center[0] + stalk_radius], [stalk_top, stalk_top], 'k-', linewidth=2)
    #ax.plot([center[0] - stalk_radius, center[0] - stalk_radius], [stalk_bottom, stalk_top], 'k-', linewidth=2)
    #ax.plot([center[0] + stalk_radius, center[0] + stalk_radius], [stalk_bottom, stalk_top], 'k-', linewidth=2)

    # Draw cap
    x = center[0]
    y = center[1]
    ax.plot(x, y, 'r', linewidth=2)


def plot_field_magnitude_heatmap(simulation):
    sim_size = simulation.cell_size
    resolution = simulation.resolution
    
    Ex = simulation.get_array(component=mp.Ex, size=sim_size, center=mp.Vector3())
    Ey = simulation.get_array(component=mp.Ey, size=sim_size, center=mp.Vector3())
    Ez = simulation.get_array(component=mp.Ez, size=sim_size, center=mp.Vector3())
    
    E_magnitude = np.sqrt(np.abs(Ex)**2)# + np.abs(Ey)**2) #+ np.abs(Ez)**2)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(E_magnitude.transpose(), interpolation='none', cmap='inferno', extent=[-sim_size.x, sim_size.x, -sim_size.y, sim_size.y], origin='lower')
    plt.colorbar(label='|E| Field Magnitude')
    plt.title('Electric Field Magnitude Heatmap with Structures Overlay Ex')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    ax = plt.gca()
    draw_structure(ax, [-1.001, 0], 0.5, 0.2, 1)  
    draw_structure(ax, [1.001, 0], 0.5, 0.2, 1)   

    E_magnitude = np.sqrt(np.abs(Ey)**2)# + np.abs(Ey)**2) #+ np.abs(Ez)**2)
    plt.figure(figsize=(8, 6))
    plt.imshow(E_magnitude.transpose(), interpolation='none', cmap='viridis', extent=[-sim_size.x, sim_size.x, -sim_size.y, sim_size.y], origin='lower')
    plt.colorbar(label='|E| Field Magnitude')
    plt.title('Electric Field Magnitude Heatmap with Structures Overlay Ey')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    ax = plt.gca()
    draw_structure(ax, [-1.001, 0], 0.5, 0.2, 1)  
    draw_structure(ax, [1.001, 0], 0.5, 0.2, 1)   
    
    E_magnitude = np.sqrt(np.abs(Ez)**2)# + np.abs(Ey)**2) #+ np.abs(Ez)**2)
    plt.figure(figsize=(8, 6))
    plt.imshow(E_magnitude.transpose(), interpolation='none', cmap='plasma', extent=[-sim_size.x, sim_size.x, -sim_size.y, sim_size.y], origin='lower')
    plt.colorbar(label='|E| Field Magnitude')
    plt.title('Electric Field Magnitude Heatmap with Structures Overlay Ez')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    ax = plt.gca()
    draw_structure(ax, [-1.001, 0], 0.5, 0.2, 1)  
    draw_structure(ax, [1.001, 0], 0.5, 0.2, 1)   


    E_magnitude = np.sqrt(np.abs(Ez)**2 + np.abs(Ex)**2 + np.abs(Ey)**2)# + np.abs(Ey)**2) #+ np.abs(Ez)**2)
    plt.figure(figsize=(8, 6))
    plt.imshow(E_magnitude.transpose(), interpolation='none', cmap='inferno', extent=[-sim_size.x, sim_size.x, -sim_size.y, sim_size.y], origin='lower')
    plt.colorbar(label='|E| Field Magnitude')
    plt.title('Electric Field Magnitude Heatmap with Structures Overlay E_norm')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    ax = plt.gca()
    draw_structure(ax, [-1.001, 0], 0.5, 0.2, 1)  
    draw_structure(ax, [1.001, 0], 0.5, 0.2, 1)   

    plt.show()



source_frequency = eV_to_meep_frequency(2.34)
simulation = init_simulation(source_frequency)
simulation.run(until=200)  
plot_field_magnitude_heatmap(simulation)
