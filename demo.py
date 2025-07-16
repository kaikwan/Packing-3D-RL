import random
import numpy as np
from packing3d.pack import *
from packing3d.show import *
import time
import queue

def getSurfaceItem(xSize, ySize, zSize):

    cube = np.ones((xSize, ySize, zSize))
    # Set all interior to 0, only keep the surface
    cube[1: xSize-1, 1: ySize-1, 1: zSize-1] = 0

    return Item(cube)

def Task():
    box_size = (15, 25, 25) # (25, 25, 15) (25, 15, 25) z, x, y
  
    # Hollow, only keep surface, faster calculation
    items = [
        getSurfaceItem(8, 9, 9),
             getSurfaceItem(7, 6, 10),
             getSurfaceItem(8, 10, 9), 
             getSurfaceItem(9, 8, 5),
             getSurfaceItem(8, 5, 4),
            #  getSurfaceItem(10, 7, 8),
             getSurfaceItem(7, 4, 4),
             getSurfaceItem(6, 6, 3),
             getSurfaceItem(6, 10, 10), # 6, 10, 5
             getSurfaceItem(9, 7, 6),
             getSurfaceItem(7, 6, 4),
             getSurfaceItem(7, 6, 4),
             getSurfaceItem(9, 6, 4), # 8, 5, 4
             getSurfaceItem(7, 7, 4),
             getSurfaceItem(10, 7, 4),
             getSurfaceItem(5, 5, 4),
            #  getSurfaceItem(3, 3, 3),
            ]

    # Solid cuboids
    # items = [Item(np.ones((5, 13, 15))),
    #         Item(np.ones((18, 6, 12))),
    #         Item(np.ones((10, 10, 9))), 
    #         Item(np.ones((16, 11, 13))),
    #         Item(np.ones((12, 8, 5))),
    #         Item(np.ones((8, 5, 4)))
    # ]

    problem = PackingProblem(box_size, items)

    # problem.pack_all_items()
    display = Display(box_size)
    
    for idx in range(len(items)):
        problem.autopack_oneitem(idx)   
        display.show3d(problem.container.geometry)
        # time.sleep(0.5)
        # Save the figure after drawing, not before/after plt.clf()
        display.fig.savefig('container.png')
    
    input("Demo complete, press Enter key to exit")

def main():
    # Assuming the original code is in a function or can be wrapped in one
    # Call your existing test code here
    Task()

if __name__ == "__main__":
    main()