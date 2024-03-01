import Simulation
import Visualization
import Geometry 
import Material 

gold = Material(material_type='gold')
geom = Geometry(shape='triangle', material=gold)
sim = SimulationSetup(geom, resolution=100)
