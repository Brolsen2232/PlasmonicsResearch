class Geometry:
    def __init__(self, shape='triangle', material=None):
        self.shape = shape
        self.material = material
    
    def create_geometry(self):
        if self.shape == 'triangle':
            return self.create_triangle_geometry(self.material)
        else:
            raise ValueError("Unsupported geometry shape")
    
    def create_triangle_geometry(self, material):
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
