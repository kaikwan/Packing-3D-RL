from copy import deepcopy
from queue import PriorityQueue
import math
import time

from .utils import *
from .object import *
from . import stability_checker

class Container(object):

    def __init__(self, box_size):
        """Container class providing operations for placing objects

        Args:
            box_size (tuple): Size of the box (z, x, y)
        """        
        self.box_size = box_size
        # Create empty geometry
        self.geometry = Geometry(np.zeros(tuple(box_size)))
        # Calculate heightmap
        self.heightmap = self.geometry.heightmap_topdown()
        # Current number of objects placed (for distinguishing with different colors)
        self.number = 0
        # Track all placed items for stability checking
        self.placed_items = []


    # Clear the container
    def clear(self):
        self.geometry = Geometry(np.zeros(tuple(self.box_size)))
        self.heightmap = self.geometry.heightmap_topdown()
        self.number = 0
        self.placed_items = []

    
    # Calculate score based on given heuristic function
    def heuristic_score(self, item: Item, centroid, method):
        # Small constant used in heuristic function calculations
        c = 0.5
        # item.position is just the coordinate closest to the origin of the object frame
        # Need to calculate the position of object's centroid relative to the object's origin
        # Then add them together to get the actual centroid of the object in the container
        true_centroid = item.position + centroid
        x = true_centroid.x
        y = true_centroid.y
        z = true_centroid.z

        if method == "DBLF":
            return z + c * (x + y)

        elif method == "HM":
            # Copy the old heightmap
            new_heightmap = deepcopy(self.heightmap)

            # Convert to integers for indexing
            x_index = item.position.x
            y_index = item.position.y

            # Calculate the heightmap of the container after adding the object
            for i in range(item.curr_geometry.x_size):
                for j in range(item.curr_geometry.y_size):
                    if item.heightmap_topdown[i][j] > 0:
                        new_heightmap[x_index + i][y_index + j] = z + item.heightmap_topdown[i][j]
            # Calculate score defined by heuristic function based on heightmap
            score = c * (x + y)
            for i in range(self.geometry.x_size):
                for j in range(self.geometry.y_size):
                    score += new_heightmap[i][j]
            return score

        else:
            return -1


    def add_item(self, item: Item):
        self.number += 1
        # Calculate new point cloud model
        self.geometry.add(item.curr_geometry, item.position, self.number)
        # Recalculate heightmap
        self.heightmap = self.geometry.heightmap_topdown()
        # Add to placed items list
        self.placed_items.append(deepcopy(item))


    def add_item_topdown(self, item: Item, x, y):
        """Place object from top to bottom

        Args:
            item (Item): Current object to be placed
            x (int): x-coordinate for placement
            y (int): y-coordinate for placement

        Returns:
            bool: True indicates can be placed in the container, False indicates cannot
        """        

        assert type(x) == int and type(y) == int \
            and x >= 0 and y >= 0, "x, y must be positive integers"

        # If the object cannot fit into the container in the plane dimension
        if x + item.curr_geometry.x_size > self.geometry.x_size \
            or y + item.curr_geometry.y_size > self.geometry.y_size:
            return False

        # Calculate the z-coordinate of the upper surface of the object at this position
        item_upper_z = 0
        for i in range(item.curr_geometry.x_size):
            for j in range(item.curr_geometry.y_size):
                item_upper_z = max(item_upper_z, 
                                self.heightmap[x + i][y + j] + item.heightmap_bottomup[i][j])

        # If the upper surface exceeds the upper bound of the container
        if item_upper_z > self.geometry.z_size:
            return False

        # Object's z-coordinate (specifically the z-coordinate of the origin of the 3D body containing the object)
        z = round(item_upper_z - item.curr_geometry.z_size)

        # Set the object's coordinates (pass by reference, directly affects the parameter passed in)
        item.position = Position(x, y, z)

        return True


    def search_possible_position(self, item: Item, grid_num=10, step_width=45):
        
        # Store all possible transformation matrices
        # stable_transforms_score = PriorityQueue(TransformScore(score=10000))
        stable_transforms_score = PriorityQueue()

        # Divide the container into grid_num * grid_num grids
        # Try to place the object for each grid
        grid_coords = []
        for i in range(grid_num):
            for j in range(grid_num):
                x = math.floor(self.geometry.x_size * i / grid_num)
                y = math.floor(self.geometry.y_size * j / grid_num)
                grid_coords.append([x, y])

        t1 = time.time()

        # step_width is the step size for traversing roll, pitch, yaw
        # Preprocessing: Find some relatively stable roll, pitch in advance
        # yaw does not affect the stability of the object on the plane
        stable_attitudes = item.planar_stable_attitude(step_width)

        t2 = time.time()
        print("find state attitudes: ", t2 - t1)

        # for att in stable_attitudes:
        #     print(att)

        attCnt = 0
        # Traverse relatively stable attitudes (excluding yaw)
        for part_attitude in stable_attitudes:

            t3 = time.time()

            # For each roll, pitch combination, traverse yaw
            for yaw in range(0, 360, step_width):
                # Generate complete attitude (including yaw)
                curr_attitude = Attitude(part_attitude.roll, part_attitude.pitch, yaw)
                # Generate rotated object
                item.rotate(curr_attitude)
                # Get the top-down and bottom-up heightmaps of the object
                item.calc_heightmap()
                # Calculate the current object's centroid
                centroid = item.curr_geometry.centroid()

                # --------- DEBUG BEGIN -----------
                # print(centroid)
                # print(curr_attitude)
                # display.show3d(item.curr_geometry)
                # input()
                # ---------- DEBUG END ------------

                # Traverse grid intersection points
                for [x, y] in grid_coords:

                    # Try to place the object at position (x, y)
                    if self.add_item_topdown(item, x, y):
                        # If current position can fit in the container
                        # Calculate score for current position
                        score = self.heuristic_score(item, centroid, "DBLF")

                        curr_position = item.position
                        curr_transform = Transform(curr_position, curr_attitude)
                        # Combine for sorting preparation
                        tf_score = TransformScore(curr_transform, score)
                        stable_transforms_score.put(tf_score)

                    # Otherwise skip
                    else:
                        continue

            t4 = time.time()
            print("try number {} yaw in this attitude: {}".format(attCnt, t4 - t3))
            attCnt += 1

        # Remove score, only keep transform
        # Take top 10
        stable_transforms = []
        cnt = 0
        while not stable_transforms_score.empty() and cnt < 10:
            cnt += 1
            transform_score = stable_transforms_score.get()

            stable_transforms.append(transform_score.transform)
        
        return stable_transforms

    def get_placed_items(self):
        return self.placed_items

    def check_stability_with_candidate(self, candidate_item, mass=1.0, mu=0.5):
        """Check stability of the current placed items plus a candidate item using the stability checker."""
        # Use the new stability_checker API directly
        items = self.placed_items + [candidate_item]
        return stability_checker.check_stability(items, mass=mass, mu=mu, plot=True, container_size=self.box_size)


