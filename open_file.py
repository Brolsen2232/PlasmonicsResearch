import matplotlib.pyplot as plt
import numpy as np
import h5py

def plot_ez_multislice_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        ez_i = np.array(file['ez.i'])  
        ez_r = np.array(file['ez.r'])  

        ez_data = ez_i + 1j * ez_r  

        xz_slice = ez_data.shape[0] // 2
        xy_slice = ez_data.shape[2] // 2
        yz_slice = ez_data.shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # xz plane (Magnitude)
        im = axes[0].imshow(np.abs(ez_data[:, :, xy_slice]).T, extent=[-0.5, 0.5, -0.5, 0.5], 
                            origin='lower', cmap='viridis') # Use a suitable colormap
        axes[0].set_title('Magnitude of Ez in xz plane')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('z')
        fig.colorbar(im, ax=axes[0])  # Add a colorbar

        # xy plane (Magnitude)
        im = axes[1].imshow(np.abs(ez_data[xz_slice, :, :]).T, extent=[-0.5, 0.5, -0.5, 0.5], 
                            origin='lower', cmap='viridis')
        axes[1].set_title('Magnitude of Ez in xy plane')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        fig.colorbar(im, ax=axes[1])  # Add a colorbar

        # yz plane (Magnitude)
        im = axes[2].imshow(np.abs(ez_data[:, yz_slice, :]).T, extent=[-0.5, 0.5, -0.5, 0.5], 
                            origin='lower', cmap='viridis')
        axes[2].set_title('Magnitude of Ez in yz plane')
        axes[2].set_xlabel('y')
        axes[2].set_ylabel('z')
        fig.colorbar(im, ax=axes[2])  # Add a colorbar

        plt.tight_layout()
        plt.show()

plot_ez_multislice_from_h5('mushroom_test_3d-ez-000080000.h5')
