import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


    def one_voxel(self, offset):
        new_voxel = self.origin_voxel + np.array(offset)
        return new_voxel
    
    def show3d(self, geom):

        # Clear all content on the canvas
        plt.clf() 
        # Reset the image information on the canvas
        self.set_ax3d()

        for x in range(geom.x_size):
            for y in range(geom.y_size):
                for z in range(geom.z_size):
                    if geom.cube[z][x][y] > 0:

                        color_idx = math.floor(geom.cube[z][x][y])
                        
                        pos = (x, y, z)
                        voxel = self.one_voxel(pos)

                        self.ax.add_collection3d(Poly3DCollection(verts=voxel, 
                                                    facecolors=self.colors[color_idx % len(self.colors)],))
        
        plt.draw()
    
    def show2d(self, mat):

        plt.clf()
        self.set_ax2d()
        bar = self.ax.imshow(mat)
        plt.colorbar(bar, ax=self.ax)
        plt.draw()
