import random
from utils import astar
from map import UNKNOWN
import numpy as np

random.seed(5)

class Robot:
    def __init__(self, map, pos, id, sense_radius, comm_range=None):
        # Config
        self.map = map
        self.pos = pos
        self.id = id
        self.sense_radius = sense_radius
        self.comm_range = comm_range

        self.robots = []
        self.neighbors = []

        self.path = []
        self.target = None

        # Initial sensing
        x, y = self.pos
        self.map.sense(x, y, self.sense_radius)

    # -------------------- Neighbor / Map Sharing --------------------
    def find_neighbors(self):
        # If no communication constraint
        if not self.comm_range:
            return [robot for robot in self.robots if robot != self]

        neighbors = []
        for robot in self.robots:
            if robot == self:
                continue
            dx = self.pos[0] - robot.pos[0]
            dy = self.pos[1] - robot.pos[1]
            distance = (dx**2 + dy**2)**0.5
            if distance <= self.comm_range:
                neighbors.append(robot)
        return neighbors

    def exchange_map(self, neighbor):
        neighbor_map = neighbor.map.explored_map
        my_map = self.map.explored_map
        update_mask = (neighbor_map != UNKNOWN)
        my_map[update_mask] = neighbor_map[update_mask]
        update_mask_neighbor = (my_map != UNKNOWN)
        neighbor_map[update_mask_neighbor] = my_map[update_mask_neighbor]

    # -------------------- Frontier Detection --------------------
    def find_frontiers(self):
        frontiers = []
        height, width = self.map.explored_map.shape
        for y in range(height):
            for x in range(width):
                if self.map.explored_map[y, x] == 0:
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if self.map.explored_map[ny, nx] == -1:
                                frontiers.append((x, y))
                                break
        return frontiers

    # -------------------- Exploration --------------------
    def explore(self):
        if not self.path:
            frontiers = self.find_frontiers()
            if not frontiers:
                self.target = None
                return False

            # Sort frontiers by distance to robot
            frontiers.sort(key=lambda p: abs(self.pos[0]-p[0]) + abs(self.pos[1]-p[1]))
            self.target_candidates = frontiers

            while self.target_candidates:
                target = self.target_candidates.pop(0)  # pick closest frontier
                path = astar(self.pos, target, self.map.explored_map)
                if path:  # reachable
                    self.path = path[1:]  # skip current position
                    self.target = target
                    return True
                # else: try the next closest frontier

            # None of the frontiers are reachable
            self.target = None
            return False

        # Move one step along the path
        if self.path:
            self.pos = self.path.pop(0)
            self.map.sense(*self.pos, self.sense_radius)

        return True

    # -------------------- Step --------------------
    def step(self):
        # If complete
        if not (UNKNOWN in self.map.explored_map):
            return False

        # Share maps with nearby robots
        self.neighbors = self.find_neighbors()
        for neighbor in self.neighbors:
            self.exchange_map(neighbor)

        # Explore normally
        return self.explore()