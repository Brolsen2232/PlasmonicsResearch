import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

def eV_to_meep_frequency(eV, a=1e-6):
    return eV / (a * 241.798)
um_scale = 1.0
eV_um_scale = um_scale / 1.23984193

def make_gold_material():
    metal_range = mp.FreqRange(min=um_scale / 6.1992, max=um_scale / 0.24797)
    Au_plasma_frq = 9.03 * eV_um_scale
    Au_f0 = 0.760
    Au_frq0 = 1e-10
    Au_gam0 = 0.053 * eV_um_scale
    Au_sig0 = Au_f0 * Au_plasma_frq**2 / Au_frq0**2
    Au_f1 = 0.024
    Au_frq1 = 0.415 * eV_um_scale  # 2.988 μm
    Au_gam1 = 0.241 * eV_um_scale
    Au_sig1 = Au_f1 * Au_plasma_frq**2 / Au_frq1**2
    Au_f2 = 0.010
    Au_frq2 = 0.830 * eV_um_scale  # 1.494 μm
    Au_gam2 = 0.345 * eV_um_scale
    Au_sig2 = Au_f2 * Au_plasma_frq**2 / Au_frq2**2
    Au_f3 = 0.071
    Au_frq3 = 2.969 * eV_um_scale  # 0.418 μm
    Au_gam3 = 0.870 * eV_um_scale
    Au_sig3 = Au_f3 * Au_plasma_frq**2 / Au_frq3**2
    Au_f4 = 0.601
    Au_frq4 = 4.304 * eV_um_scale  # 0.288 μm
    Au_gam4 = 2.494 * eV_um_scale
    Au_sig4 = Au_f4 * Au_plasma_frq**2 / Au_frq4**2
    Au_f5 = 4.384
    Au_frq5 = 13.32 * eV_um_scale  # 0.093 μm
    Au_gam5 = 2.214 * eV_um_scale
    Au_sig5 = Au_f5 * Au_plasma_frq**2 / Au_frq5**2

    Au_susc = [
        mp.DrudeSusceptibility(frequency=Au_frq0, gamma=Au_gam0, sigma=Au_sig0),
        mp.LorentzianSusceptibility(frequency=Au_frq1, gamma=Au_gam1, sigma=Au_sig1),
        mp.LorentzianSusceptibility(frequency=Au_frq2, gamma=Au_gam2, sigma=Au_sig2),
        mp.LorentzianSusceptibility(frequency=Au_frq3, gamma=Au_gam3, sigma=Au_sig3),
        mp.LorentzianSusceptibility(frequency=Au_frq4, gamma=Au_gam4, sigma=Au_sig4),
        mp.LorentzianSusceptibility(frequency=Au_frq5, gamma=Au_gam5, sigma=Au_sig5),
    ]

    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, valid_freq_range=metal_range)

def make_mushroom_structures(material):
    cap_radius =0.5  # Radius of the semicircle cap
    stem_height = 0.5  # Height of the stem
    stem_width = 0.2  # Width of the stem
    gap = 0.1  # Gap between the two mushrooms
    
    center_left = -gap/2 - cap_radius/2
    center_right = gap/2 + cap_radius/2

    mushroom_left_cap = mp.Cylinder(radius=cap_radius, height=mp.inf, center=mp.Vector3(center_left, 0), axis=mp.Vector3(0,0,1), material=material)
    mushroom_left_stem = mp.Block(size=mp.Vector3(stem_width, stem_height, mp.inf), center=mp.Vector3(center_left, -cap_radius - stem_height/2), material=material)
    
    mushroom_right_cap = mp.Cylinder(radius=cap_radius, height=mp.inf, center=mp.Vector3(center_right, 0), axis=mp.Vector3(0,0,1), material=material)
    mushroom_right_stem = mp.Block(size=mp.Vector3(stem_width, stem_height, mp.inf), center=mp.Vector3(center_right, -cap_radius - stem_height/2), material=material)

    return [mushroom_left_cap, mushroom_left_stem, mushroom_right_cap, mushroom_right_stem]

def init_simulation():
    source_frequency = eV_to_meep_frequency(2.34)
    cell_size = mp.Vector3(7, 7, 0)
    pml_layers = [mp.PML(1.0)]
    resolution = 150
    gold = make_gold_material()
    
    triangle_base = 1.0
    triangle_height = 2.0
    gap = 0.1

    #chunk_layout = mp.BinaryPartition(data=[(mp.X, 0),  # Split along X at the center
                                            #[0,  # Left half
                                            # [(mp.Y, 0), 1, 2]],  # Split left half along Y
                                            #[0,  # Right half
                                             #[(mp.Y, 0), 3, 4]]]) 
 
    
    vertices1 = [mp.Vector3(-triangle_base/2 - gap/2, -triangle_height/2),
                 mp.Vector3(-triangle_base/2 - gap/2, triangle_height/2),
                 mp.Vector3(-gap/2, 0)]
    vertices2 = [mp.Vector3(triangle_base/2 + gap/2, -triangle_height/2),
                 mp.Vector3(triangle_base/2 + gap/2, triangle_height/2),
                 mp.Vector3(gap/2, 0)]

    
    geometry = [mp.Prism(vertices1, height=mp.inf, material=gold),
            mp.Prism(vertices2, height=mp.inf, material=gold)]
    #geometry = make_mushroom_structures(gold)


    source_y_position = 3
    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.2),  
                         component=mp.Ez,
                         center=mp.Vector3(0, source_y_position),
                         size=mp.Vector3(cell_size.x, 0, 0))]
    
    simulation = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=geometry, sources=sources, resolution=resolution)#, chunk_layout=chunk_layout)
    return simulation

def measure_baseline_field_intensity():
    cell_size = mp.Vector3(7, 7, 0)
    pml_layers = [mp.PML(1.0)]
    resolution = 200
    source_frequency = eV_to_meep_frequency(2.34)
    sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.2),  
                         component=mp.Ez,
                         center=mp.Vector3(0, 3),
                         size=mp.Vector3(cell_size.x, 0, 0))]

    simulation = mp.Simulation(cell_size=cell_size,
                               boundary_layers=pml_layers,
                               sources=sources,
                               resolution=resolution)

    # Run the simulation for a short time to stabilize
    simulation.run(until=20)
    
    # Measure the field intensity at the source location or another representative point
    ez_data = simulation.get_array(component=mp.Ez, center=mp.Vector3(0, 3), size=mp.Vector3(0, 0, 0))
    e0_intensity = np.abs(ez_data)
    return e0_intensity

'''
def plot_field_enhancement(simulation, field_component=mp.Ez):
    y_positions = np.linspace(-5, 5, 200)  
    field_values = []

    for y in y_positions:
        field_value = simulation.get_array(component=field_component, center=mp.Vector3(0, y), size=mp.Vector3(0, 0, 0))
        field_values.append(np.abs(field_value))  

    plt.figure(figsize=(6, 4))
    plt.plot(y_positions, field_values, label='Field Enhancement')
    plt.xlabel('Position along Y (um)')
    plt.ylabel('|Ez| (a.u.)')
    plt.title('Field Enhancement in the Bowtie Gap')
    plt.legend()
    plt.grid(True)
    plt.show()
'''
from mpl_toolkits.mplot3d import Axes3D  

def plot_steady_state_field_3D(simulation, field_component=mp.Ez, steady_state_time=200):
    simulation.run(until=steady_state_time)
    field_data = simulation.get_array(component=field_component, size=simulation.cell_size, center=mp.Vector3())
    
    x = np.linspace(-simulation.cell_size.x/5, simulation.cell_size.x/5, field_data.shape[0])
    y = np.linspace(-simulation.cell_size.y/5, simulation.cell_size.y/5, field_data.shape[1])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.abs(field_data).transpose(), cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='|Ez| (a.u.)')
    ax.set_title('Steady State Field Enhancement (3D View)')
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.set_zlabel('|Ez| (a.u.)')
    plt.savefig('steady_state_plot')

def plot_steady_state_field_heatmap(simulation, field_component=mp.Ez, steady_state_time=200, filename='steady_state_heatmap.png'):
    simulation.run(until=steady_state_time)
    field_data = simulation.get_array(component=field_component, size=simulation.cell_size, center=mp.Vector3())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(field_data).transpose(), interpolation='spline36', cmap='viridis', extent=[-simulation.cell_size.x/2, simulation.cell_size.x/2, -simulation.cell_size.y/2, simulation.cell_size.y/2])
    plt.colorbar(label='|Ez| (a.u.)')
    plt.title('Steady State Field Enhancement (Heatmap)')
    plt.xlabel('X position (um)')
    plt.ylabel('Y position (um)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_field_enhancement(simulation, field_component=mp.Ez, e0_intensity=1):
    y_positions = np.linspace(-5, 5, 200)  
    field_values = []

    for y in y_positions:
        field_value = simulation.get_array(component=field_component, center=mp.Vector3(0, y), size=mp.Vector3(0, 0, 0))
        normalized_field_value = np.abs(field_value) / e0_intensity  # Normalize by E0
        field_values.append(normalized_field_value)  

    plt.figure(figsize=(6, 4))
    plt.plot(y_positions, field_values, label='Field Enhancement (E/E0)')
    plt.xlabel('Position along Y (um)')
    plt.ylabel('|Ez/E0|')
    plt.title('Field Enhancement in the Bowtie Gap (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_field_enhancement_heatmap(simulation, field_component=mp.Ez):
    # Assuming simulation has been run for a while; if not, run it for a steady state
    field_data = simulation.get_array(component=field_component, size=simulation.cell_size, center=mp.Vector3())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(field_data).transpose(), interpolation='spline36', cmap='hot')
    plt.colorbar(label='|Ez| (a.u.)')
    plt.title('Field Enhancement Heatmap')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.tight_layout()
    plt.show()


def run_and_visualize(simulation, duration=200, fps=30, plot_steady_state=False, steady_state_time=200):
    if not plot_steady_state:
        ez_data = []
        def get_efield(sim):
            ez_data.append(np.real(sim.get_array(component=mp.Ez)))
        simulation.run(mp.at_every(1 / fps, get_efield), until=duration)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_axis_off()
        img = ax.imshow(ez_data[0], cmap='RdBu', animated=True)
        def update(frame):
            img.set_data(ez_data[frame])
            return (img,)
        ani = animation.FuncAnimation(fig, update, frames=len(ez_data), interval=1000/fps)
        ani.save('mushroom.mp4', writer='ffmpeg')
    else:
        # Run the simulation for a given time and then plot the steady-state field as a heatmap
        simulation.run(until=steady_state_time)
        field_data = simulation.get_array(component=mp.Ez, size=simulation.cell_size, center=mp.Vector3())
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(field_data).transpose(), interpolation='spline36', cmap='viridis', extent=[-simulation.cell_size.x/6, simulation.cell_size.x/6, -simulation.cell_size.y/6, simulation.cell_size.y/6])
        plt.colorbar(label='|Ez| (a.u.)')
        plt.title('Steady State Field Enhancement Heatmap')
        plt.xlabel('X position (um)')
        plt.ylabel('Y position (um)')
        plt.tight_layout()
        plt.show()

    
#simulation = init_simulation()

#run_and_visualize(simulation, duration=100, fps=30, plot_steady_state=False)
#run_and_visualize(simulation, duration=200, fps=30, plot_steady_state=True, steady_state_time=150)
# Measure the baseline field intensity
e0_intensity = measure_baseline_field_intensity()

# Initialize and run the simulation with geometry as before
simulation = init_simulation()
run_and_visualize(simulation, duration=200, fps=30, plot_steady_state=True, steady_state_time=150)

# Use the modified plot functions with E0 normalization
plot_field_enhancement(simulation, mp.Ez, e0_intensity)
