class SimulationSetup:
    def __init__(self, geometry, resolution=70):
        self.geometry = geometry
        self.resolution = resolution
    
    def init_simulation():
        source_frequency = eV_to_meep_frequency(2.34)
        cell_size = mp.Vector3(10, 10, 0)
        pml_layers = [mp.PML(1.0)]
        resolution = 70
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

        source_y_position = 3
        sources = [mp.Source(src=mp.GaussianSource(frequency=source_frequency, fwidth=0.01),  
                            component=mp.Ez,
                            center=mp.Vector3(0, source_y_position),
                            size=mp.Vector3(cell_size.x, 0, 0))]
        
        simulation = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=geometry, sources=sources, resolution=resolution)
        return simulation
