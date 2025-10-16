import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

class Map:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.explored_map = np.full((height, width), UNKNOWN, dtype=int)
        self.ground_truth = np.full((height, width), FREE, dtype=int)

        for ox, oy in obstacles:
            self.ground_truth[oy, ox] = OCCUPIED

    def sense(self, x, y, radius):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.explored_map[ny, nx] = self.ground_truth[ny, nx]

    def info_gain(self, point, radius):
        """
        Calculate how many UNKNOWN cells would be observed if the robot
        were at position (x, y) with a given sensing radius.
        """
        x, y = point

        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.explored_map[ny, nx] == UNKNOWN:
                        count += 1
        return count