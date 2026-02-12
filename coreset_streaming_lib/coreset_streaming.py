"""
Class implementing the streaming coresets algorithm from
"Coresets for k-Means and k-Median Clustering and their Applications"
by Har-Peled and Mazumdar, 2003.
Vendored from: https://github.com/kvombatkere/CoreSets-Algorithms
"""

import numpy as np
import math
import random
import warnings


class Coreset_Streaming:
    def __init__(self, max_size, epsilon=0.01, d=2):
        """Initializes the Coreset_Streaming class.

        Args:
            max_size (int): Maximum size we allow the coreset to get before
                doubling the resolution.
            epsilon (float): Grid resolution parameter. Default 0.01.
            d (int): Dimensionality of data. Default 2.
        """
        self.d = d
        self.resolution = 1
        self.side_length = (epsilon * 2 / (10 * self.d))
        self.max_size = max_size
        self.coreset = []
        self.grid_points = dict()

    def add_point(self, point):
        """Adds point to the coreset, with a default weight of 1."""
        if len(point) != self.d:
            raise ValueError("Point is expected to have {} dimensions.".format(self.d))
        weight = 1
        self.coreset.append((list(point), weight))

    def grid_location(self, point):
        """Returns the location in the grid of the argument point."""
        loc = []
        for dim in point:
            loc.append(dim - dim % self.side_length)
        return loc

    def snap_points_to_grid(self):
        """Snaps each point in the coreset to a box in the grid."""
        self.grid_points = dict()
        for (point, weight) in self.coreset:
            loc = repr(self.grid_location(point))
            if loc in self.grid_points.keys():
                self.grid_points[loc].append((point, weight))
            else:
                self.grid_points[loc] = [(point, weight)]

    def build_coreset_from_grid(self):
        """Builds the coreset from grid representatives with cumulative weights."""
        self.coreset = []
        for (loc, points) in self.grid_points.items():
            (representative, _) = random.choice(self.grid_points[loc])
            repr_weight = 0
            for (point, weight) in self.grid_points[loc]:
                repr_weight += weight
            self.coreset.append((representative, repr_weight))

    def build_coreset(self):
        """Builds a coreset. Idempotent."""
        self.snap_points_to_grid()
        self.build_coreset_from_grid()
        if len(self.coreset) > self.max_size:
            self.double_resolution()

    def double_resolution(self, verbose=False):
        """Doubles the resolution of the grid."""
        if verbose:
            print("Doubling resolution rank from {} to {}".format(
                self.resolution, self.resolution + 1))
        self.resolution += 1
        self.side_length *= 2
        self.build_coreset()

    def can_union(self, cs):
        """Returns True if the other Coreset can be unioned with self."""
        if cs.resolution == self.resolution:
            return True
        warnings.warn(
            "Current Coreset resolution is {}, other's resolution is {}".format(
                self.resolution, cs.resolution))
        return False

    def union(self, cs):
        """Adds the other coreset cs to current coreset."""
        if self.resolution != cs.resolution:
            raise ValueError(
                "Current Coreset resolution is {}, other's resolution is {}".format(
                    self.resolution, cs.resolution))
        for (point, weight) in cs.coreset:
            self.coreset.append((point, weight))
        self.build_coreset()
