from copy import deepcopy
import math
from queue import PriorityQueue
from typing import Sequence

from .utils import *
import numpy as np

class Geometry(object):

    def __init__(self, cube: np.ndarray):
        self.z_size = len(cube)
        self.x_size = len(cube[0])
        self.y_size = len(cube[0][0])      
        self.cube = cube
        self.pointsMat = None
        
        self.get_points()
    
    def get_points(self):
        tmpPoints = []
        # How many points
        pointCnt = 0
        for z in range(self.z_size):
            for x in range(self.x_size):
                for y in range(self.y_size):
                    if self.cube[z][x][y] == 1:
                        tmpPoints.append(np.asmatrix([x, y, z]).T)
                        pointCnt += 1
        
        # Create a numpy matrix with 3 rows and columns equal to the total number of points, fill with points to accelerate rotation matrix calculations
        self.pointsMat = np.zeros((3, pointCnt))
        for idx in range(pointCnt):
            self.pointsMat[:, idx: idx + 1] = tmpPoints[idx]


    
    def centroid(self):
        # Calculate object's centroid
        centroid = Position()
        counter = 0
        for z in range(self.z_size):
            for x in range(self.x_size):
                for y in range(self.y_size):
                    if self.cube[z][x][y] == 1:
                        counter += 1
                        centroid += Position(x, y, z)

        centroid /= counter
        return centroid
    

    def stability(self):
        # Initial version: only consider cuboid objects

        # Calculate contact area with bottom surface
        contact_area = 0
        # Consider contact if distance from bottom is less than margin
        margin = 1
        for x in range(self.x_size):
            for y in range(self.y_size):
                for z in range(margin):
                    if self.cube[z][x][y] > 0:
                        contact_area += 1
                        break    

        return contact_area

    def heightmap_topdown(self):
        heightmap = np.zeros((self.x_size, self.y_size))
        for x in range(self.x_size):
            for y in range(self.y_size):
                # Find position of first non-zero element from top down
                max_z = self.z_size - 1
                while max_z >= 0 and self.cube[max_z][x][y] == 0:
                    max_z -= 1
                heightmap[x][y] = max_z + 1
        return heightmap
    
    def heightmap_bottomup(self):
        heightmap = np.zeros((self.x_size, self.y_size))
        for x in range(self.x_size):
            for y in range(self.y_size):
                # Find position of first non-zero element from bottom up
                min_z = 0
                while min_z < self.z_size and self.cube[min_z][x][y] == 0:
                    min_z += 1
                heightmap[x][y] = self.z_size - min_z
        return heightmap

    def get_rotate_matrix(self, attitude: Attitude):
        """Get rotation matrix for given attitude

        Args:
            attitude (Attitude): Attitude

        Returns:
            np.mat: Rotation matrix (3×3)
        """
        # roll is rotation around x-axis, pitch is rotation around y-axis, yaw is rotation around z-axis
        # Convert angles to radians
        alpha = attitude.roll * math.pi / 180
        beta = attitude.pitch * math.pi / 180
        theta = attitude.yaw * math.pi / 180

        # When rotating around origin, all points are in a sphere with the following radius
        # radius = math.sqrt(pow(self.x_size, 2) 
        #                 + pow(self.y_size, 2) 
        #                 + pow(self.z_size, 2))
        
        T_roll = np.asmatrix([[1,                 0,                0], 
                         [0,   math.cos(alpha), -math.sin(alpha)],
                         [0,   math.sin(alpha),  math.cos(alpha)]])

        T_pitch = np.asmatrix([[ math.cos(beta),  0,   math.sin(beta)],
                          [              0,  1,                0],
                          [-math.sin(beta),  0,   math.cos(beta)]])

        T_yaw = np.asmatrix([[math.cos(theta), -math.sin(theta),   0],
                        [math.sin(theta),  math.cos(theta),   0],
                        [              0,                0,   1]])

        # Given rotation execution order is roll, pitch, yaw
        T_rotate = T_yaw * T_pitch * T_roll
        return T_rotate


    
    def rotate(self, attitude: Attitude):
        """Rotate geometry (old version)

        Args:
            attitude (Attitude): Target attitude for rotation
        """        

        # t_rotateStart = time.time()

        # When rotating around origin, all points are in a sphere with the following radius
        radius = math.sqrt(pow(self.x_size, 2) 
                        + pow(self.y_size, 2) 
                        + pow(self.z_size, 2))


        # Add offset to ensure all coordinates are positive
        offset = np.asmatrix([radius, radius, radius]).T

        # Given rotation execution order is roll, pitch, yaw
        # T_rotate = T_yaw * T_pitch * T_roll
        T_rotate = self.get_rotate_matrix(attitude)

        # Store transformed points
        new_points = []

        # Directly transformed objects have holes because discrete point mapping may map to the same integer point
        # Therefore, optimize mapping by mapping one point to multiple target points

        # Use pre-processed self.points to accelerate matrix calculations
        # newPointMat has both positive and negative values at this point
        newPointMat = T_rotate * self.pointsMat

        distThsld = 0.8661

        for idx in range(newPointMat.shape[1]):
            
            # Add offset to get positive coordinates
            newPoint = newPointMat[:, idx: idx + 1] + offset
            # Get coordinates of a point (decimals)
            [nx, ny, nz] = [newPoint[i, 0] for i in range(3)]
            pxList = [math.floor(nx), math.ceil(nx)]
            pyList = [math.floor(ny), math.ceil(ny)]
            pzList = [math.floor(nz), math.ceil(nz)]
            
            for px in pxList:
                for py in pyList:
                    for pz in pzList:
                        # Calculate distance from transformed point to surrounding integer points
                        ptDist = dist(nx, ny, nz, px, py, pz)
                        # Add integer points with distance less than threshold to new_points
                        if ptDist < distThsld:
                            new_points.append(np.asmatrix([px, py, pz]).T)
        if not new_points:
            raise ValueError("No points generated after rotation. Check distThsld, input geometry, or rotation matrix.")

        min_x = min_y = min_z = math.ceil(radius)
        max_x = max_y = max_z = 0

        # Find maximum and minimum values in each axis direction for all points
        for point in new_points:
            min_x = min(min_x, point[0, 0])
            min_y = min(min_y, point[1, 0])
            min_z = min(min_z, point[2, 0])

            max_x = max(max_x, point[0, 0])
            max_y = max(max_y, point[1, 0])
            max_z = max(max_z, point[2, 0])
        
        # Make object's frame tight against the coordinate system's "corner"
        for point in new_points:
            point -= np.asmatrix([min_x, min_y, min_z]).T
        
        # Size of the object's frame after rotation transformation
        self.x_size = round(max_x - min_x + 1)
        self.y_size = round(max_y - min_y + 1)
        self.z_size = round(max_z - min_z + 1)

        assert self.x_size > 0 and self.y_size > 0 and self.z_size > 0, \
            print("{} {} {}".format(self.x_size, self.y_size, self.z_size))
        # "Object frame size incorrect"

        # Create new empty cube
        self.cube = np.zeros((self.z_size, self.x_size, self.y_size))
        # Fill in the valued parts of the cube
        for point in new_points:
            [x, y, z] = [int(point[i, 0]) for i in range(3)]
            self.cube[z][x][y] = 1
        
        # t_afterCreate = time.time()
        # print("Create New Geometry Time: ", t_afterCreate - t_beforeCreate, "\n")

        # t_rotateEnd = time.time()
        # print("Rotat Run Time: ", t_rotateEnd - t_rotateStart, "\n\n")


    def add(self, geom, position: Position, coef=1):
        # x, y, z are coordinates of each point in the small object being added
        for z in range(geom.z_size):
            for x in range(geom.x_size):
                for y in range(geom.y_size):
                    # nx, ny, nz are corresponding coordinates in the large object
                    nz = z + position.z
                    nx = x + position.x
                    ny = y + position.y
                    # If out of bounds
                    if nz >= self.z_size \
                        or nx >= self.x_size \
                        or ny >= self.y_size:
                        continue
                    # Add valued parts of small geometry to large geometry, preserve original values elsewhere
                    if geom.cube[z][x][y] > 0:
                        self.cube[nz][nx][ny] = geom.cube[z][x][y] * coef



class Item(object):

    def __init__(self, cube, position: Position = Position(), 
                    attitude: Attitude = Attitude()):
        self.init_geometry = Geometry(cube)
        self.curr_geometry = Geometry(cube)

        self.position = position
        self.attitude = attitude
        self.heightmap_topdown = None
        self.heightmap_bottomup = None
    
    # Calculate the two heightmaps
    def calc_heightmap(self):
        self.heightmap_topdown = self.curr_geometry.heightmap_topdown()
        self.heightmap_bottomup = self.curr_geometry.heightmap_bottomup()

    # Rotate init_geometry by a certain angle to get curr_geometry
    def rotate(self, attitude: Attitude):
        self.curr_geometry = Geometry(self.init_geometry.cube)
        self.curr_geometry.rotate(attitude)
        self.attitude = attitude

    # Transformation including rotation and translation
    def transform(self, transform: Transform):
        self.rotate(transform.attitude)
        self.position = transform.position

    # Get object attitudes with planar stability
    def planar_stable_attitude(self, step_width):

        # Take the top 6 attitudes with highest stability
        stable_attitudes_score = PriorityQueue()

        # Traverse all roll and pitch angles
        for roll in range(0, 360, step_width):
            for pitch in range(0, 360, step_width):
                # Current attitude parameters
                curr_attitude = Attitude(roll, pitch, 0)
                self.curr_geometry = Geometry(self.init_geometry.cube)
                self.curr_geometry.rotate(curr_attitude)
                # Calculate stability corresponding to current attitude
                stability = self.curr_geometry.stability()

                # ------DEBUG BEGIN------
                # print("roll: ", roll, "    pitch: ", pitch)
                # print("stability: ", stability)
                # display = Display([15, 15, 15])
                # display.show(self.curr_geometry)
                # input()
                # -------DEBUG END-------

                # Add to priority queue for sorting
                stable_attitudes_score.put(AttitudeStability(curr_attitude, stability))

        # Remove stability values, only keep attitudes
        # Take top 6 attitudes with highest stability
        stable_attitudes = []
        cnt = 0
        while not stable_attitudes_score.empty() and cnt < 6:
            cnt += 1
            attitude_score = stable_attitudes_score.get()
            # print(attitude_score)
            stable_attitudes.append(attitude_score.attitude)
        
        return stable_attitudes