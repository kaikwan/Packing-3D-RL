import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


class Display:
    
    def __init__(self, space_size=(10, 10, 10)):
        """Visualize the state of the container

        Args:
            space_size (tuple, optional): Size of the container (z, x, y). Defaults to (10, 10, 10).
        """        

        plt.ion() #Enable interactive mode - key function for success
        # Set up 3D view display
        self.fig = plt.figure() 
             
        self.space_size = space_size
        
        # Use different colors for each object
        self.colors = ['lightcoral', 'lightsalmon', 'gold', 'olive',
            'mediumaquamarine', 'deepskyblue', 'blueviolet', 'pink',
            'brown', 'darkorange', 'yellow', 'lawngreen', 'turquoise',
            'dodgerblue', 'darkorchid', 'hotpink', 'deeppink', 'peru',
            'orange', 'darkolivegreen', 'cyan', 'purple', 'crimson']

        # Basic small cube
        self.origin_voxel = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
                             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                             [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
                             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
                             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]

    def set_size(self, space_size):
        self.space_size = space_size

    def set_ax3d(self):

        self.ax = self.fig.add_subplot(projection='3d')  
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, self.space_size[1]])
        self.ax.set_ylim([0, self.space_size[2]])
        self.ax.set_zlim([0, self.space_size[0]])

        # Remove the custom projection that stretches z axis
        # self.ax.get_proj = lambda: np.dot(Axes3D.get_proj(self.ax), np.diag([1, 1, 1.3, 1]))
        
        # Set equal aspect ratio for all axes
        self.ax.set_box_aspect([self.space_size[1], self.space_size[2], self.space_size[0]])

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.ax.view_init(50, 45)
    
    def set_ax2d(self):

        self.ax = self.fig.add_subplot()  

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def show3d(self, geom):
        """
        Fast voxel rendering with Matplotlib’s ax.voxels (correct orientation).
        """
        plt.clf()
        
        self.set_ax3d()                        # (re)creates self.ax

        cube   = geom.cube                     # original shape: (z, x, y)
        active = cube > 0
        if not np.any(active):                 # nothing to draw
            plt.draw()
            return

        # ---------- colour table (pre‑computed RGBA) ----------
        color_rgba = np.asarray([to_rgba(c) for c in self.colors])   # shape (C, 4)

        idx_flat   = (cube[active].astype(int) % len(color_rgba))    # (N,)
        rgba_flat  = color_rgba[idx_flat]                            # (N, 4)

        rgba_full = np.zeros((*cube.shape, 4))
        rgba_full[active] = rgba_flat

        # ---------- correct axis order: (z, x, y) --> (x, y, z) ----------
        active_xyz = np.transpose(active,  (1, 2, 0))          # bool  (x, y, z)
        colors_xyz = np.transpose(rgba_full, (1, 2, 0, 3))     # RGBA (x, y, z, 4)

        # ---------- draw ----------
        self.ax.voxels(
            active_xyz,
            facecolors=colors_xyz,
            edgecolor=None          # set 'k' or similar if you want grid lines
        )
        plt.draw()
        plt.savefig('container.png')


    def show2d(self, mat):

        plt.clf()
        self.set_ax2d()
        bar = self.ax.imshow(mat)
        plt.colorbar(bar, ax=self.ax)
        plt.draw()
