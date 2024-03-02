import h5py
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
def save_fields(sim):
    sim.output_efield_z('ez')


def init_simulation(source_frequency):
    padding = 0.1  # um, padding around the mushroom structure
    cell_size = mp.Vector3(0.5 + padding, 0.5 + padding, 1)  # 3D simulation space with padding
    
    resolution = 200
    
    gold = make_gold_material()

    stalk_height = 0.1  # 100 nm
    stalk_radius = 0.025  # 25 nm for the stalk radius
    cap_radius = 0.05  # 50 nm for the cap radius
    mushroom_center = mp.Vector3(0, 0, stalk_height / 2)
    symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]
    geometry = [
        mp.Cylinder(height=stalk_height, radius=stalk_radius, center=mushroom_center, axis=mp.Vector3(0,0,1), material=gold),
        mp.Sphere(radius=cap_radius, center=mushroom_center + mp.Vector3(0, 0, stalk_height / 2 + cap_radius / 2), material=gold)
        #mp.Cylinder(height=stalk_height, radius=stalk_radius, center=mushroom_center + mp.Vector3(0, 0.2, 0) , axis=mp.Vector3(0,0,1), material=gold),
        #mp.Sphere(radius=cap_radius, center=mushroom_center + mp.Vector3(0, 0.2, stalk_height / 2 + cap_radius / 2), material=gold), 
    ]

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

    return simulation



source_frequency = eV_to_meep_frequency(2.34)
simulation = init_simulation(source_frequency)
simulation.run(mp.at_end(mp.output_efield_z), until=200)
def plot_ez_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        ez_data = file['ez'][...]  
        
        plt.figure(figsize=(10, 8))
        plt.imshow(ez_data.T, extent=[-1, 1, -1, 1], origin='lower')  
        plt.colorbar(label='Ez')
        plt.title('Ez Field Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


