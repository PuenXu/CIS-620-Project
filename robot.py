import random
from map import UNKNOWN
from utils import solve_tsp
from collections import defaultdict
from task_allocation import TaskAllocation

random.seed(5)

class Robot:
    def __init__(self, map, pos, id, sense_radius, comm_range=None, strategy="closest_frontier"):
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
        self.target_candidates = []
        
        # Auction
        self.auctions_started = 0
        self.auctions_kept = 0
        self.auctions_lost_to = defaultdict(int)

        # Task Allocation
        self.task_allocator = TaskAllocation(self, strategy=strategy)

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
    
    # -------------------- Game Theoretic Utilities --------------------
    def find_nash_eq(self, neighbors):
        # To do
        # Estimate the neighbors' payoff to the tasks, approximate NE
        pass

    def sample_tasks(self):
        # To do
        # Should be similar to sample_candidates, but also need to check they are reachable to me by A*
        pass

    # -------------------- Auction Utilities --------------------
    def sample_candidates(self, frontiers, len_candidates):
        candidates = [p for p in self.target_candidates if p in frontiers]
        num_to_sample = len_candidates - len(candidates)
        if num_to_sample > 0:
            possible_new = [p for p in frontiers if p not in candidates]
            if possible_new:
                new_candidates = random.sample(
                    possible_new,
                    min(num_to_sample, len(possible_new))
                )
            else:
                new_candidates = []
            candidates.extend(new_candidates)
        return candidates

    def find_target(self, tsp=False):
        if tsp:
            sorted_tour = solve_tsp(self.pos, self.target_candidates)
            target = sorted_tour[0]
        else:
            target = max(self.target_candidates, key=lambda p: self.utility(p), default=None)
        return target

    def cost(self, point):
        return abs(self.pos[0] - point[0]) + abs(self.pos[1] - point[1])

    def utility(self, point):
        return self.map.info_gain(point, self.sense_radius) - self.cost(point)
    
    # -------------------- Exploration --------------------
    def explore(self):

        if not self.path:
            frontiers = self.find_frontiers()
            self.path, self.target = self.task_allocator.assign_target(frontiers)
            if not self.path:
                return False
            return True

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