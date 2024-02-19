import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class Visualization:

    def __init__(self, simulation):
        self.simulation = simulation

    @staticmethod
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
    
    @staticmethod
    def plot_steady_state_field_3D(simulation, field_component=mp.Ez, steady_state_time=200):
        simulation.run(until=steady_state_time)
        field_data = simulation.get_array(component=field_component, size=simulation.cell_size, center=mp.Vector3())
        
        x = np.linspace(-simulation.cell_size.x/2, simulation.cell_size.x/2, field_data.shape[0])
        y = np.linspace(-simulation.cell_size.y/2, simulation.cell_size.y/2, field_data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, np.abs(field_data).transpose(), cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='|Ez| (a.u.)')
        ax.set_title('Steady State Field Enhancement (3D View)')
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_zlabel('|Ez| (a.u.)')
        plt.show()
    @staticmethod
    def run_and_visualize(simulation, duration=50, fps=30, plot_steady_state=False, steady_state_time=200):
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
        ani.save('bowtie_simulation.mp4', writer='ffmpeg')
        plot_field_enhancement(simulation)  
        plt.close(fig)
    else:
        plot_steady_state_field_3D(simulation, field_component=mp.Ez, steady_state_time=200)

        
