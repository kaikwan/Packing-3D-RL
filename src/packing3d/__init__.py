"""
Packing3D - 3D Bin Packing Solver

This package provides methods for solving 3D bin packing problems using
heuristic search and reinforcement learning approaches.
"""

from .container import Container
from .object import Item, Geometry
from .pack import PackingProblem

from .utils import Position, Attitude, Transform
from .show import Display

__version__ = "0.1.0"