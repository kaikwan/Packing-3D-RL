from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import time

from .show import Display
from .utils import *
from .object import *
from .container import *


class PackingProblem(object):

    def __init__(self, box_size, items):
        """Top-level class for solving bin packing problems

        Args:
            box_size (tuple): Size of the box in (z, x, y) order
            items (list): List of objects to be packed
        """        
        # Container
        self.container = Container(box_size)
        # List of objects to be packed
        self.items = items
        # Packing sequence
        self.sequence = list(range(len(items)))
        # Rotation matrices for each object
        self.transforms = list()
        # Current number of packed objects
        self.count = 0
    

    def load_new_items(self, items):
        """Load a new set of objects

        Args:
            items (list): New set of objects
        """        
        self.items = items
        self.sequence = list(range(len(items)))
        self.count = 0


    def pack(self, x, y):
        """Place current object at specified coordinates

        Args:
            x (int): x coordinate
            y (int): y coordinate

        Returns:
            bool : True indicates successfully placed, False indicates cannot be placed
        """

        # Get the current object to be placed in the box
        item: Item = self.items[self.sequence[self.count]]

        # Translate to current position
        item.position = Position(x, y, 0)

        # Calculate object's coordinates when placing from top to bottom
        # Actually only changes the z-coordinate value, xy coordinates remain unchanged
        # Determine if it can be placed (may fail due to container volume constraints)
        result = self.container.add_item_topdown(item, x, y)

        if result is True:
            # Actually place the object in the container
            self.container.add_item(item)
            self.count += 1
            return True
        
        return False

    
    def autopack_oneitem(self, item_idx):

        # print("item index: ", item_idx)
        # t1 = time.time()
        curr_item: Item = self.items[item_idx]
        transforms = self.container.search_possible_position(curr_item)
        # t2 = time.time()
        # print("search possible positions: ", t2 - t1)

        # If no placement position can be found
        assert len(transforms) > 0, "Could not find a position and orientation for placement"
        
        # Temporarily ignore stability of object stack after placement
        # Directly place the object according to the first-ranked transformation matrix
        curr_item.transform(transforms[0])

        # t3 = time.time()
        # print("rotate to a position: ", t3 - t2)
        self.container.add_item(curr_item)

        # t4 = time.time()
        # print("add item to container: ", t4 - t3, '\n')

    
    def autopack_allitems(self):
        for idx in self.sequence:
            self.autopack_oneitem(idx)


if __name__ == "__main__": 
    pass