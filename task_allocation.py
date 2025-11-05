from utils import astar
from nash_eq import run_ga

class TaskAllocation:
    """
    Handles task allocation (frontier selection) for a robot.
    Supports multiple strategies:
      - 'closest_frontier': picks the nearest reachable frontier
      - 'auction': distributed frontier assignment using neighbor bidding
    """
    def __init__(self, robot, strategy="closest_frontier"):
        self.robot = robot
        self.strategy = strategy

    def assign_target(self, frontiers):
        if self.strategy == "closest_frontier":
            return self.closest_frontier(frontiers)
        elif self.strategy == "auction":
            return self.auction(frontiers)
        elif self.strategy == "competitive":
            return self.competitive(frontiers)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # Strategy 1 — Closest Frontier
    def closest_frontier(self, frontiers):
        if not frontiers:
            return None, None

        # Sort by Manhattan distance
        frontiers.sort(key=lambda p: abs(self.robot.pos[0]-p[0]) + abs(self.robot.pos[1]-p[1]))
        target_candidates = list(frontiers)

        while target_candidates:
            target = target_candidates.pop(0)
            path = astar(self.robot.pos, target, self.robot.map.explored_map)
            if path:
                return path[1:], target
        return None, None

    # Strategy 2 — Auction
    def auction(self, frontiers):
        if not frontiers:
            return None, None

        len_candidates = 8
        robot = self.robot
        robot.target_candidates = robot.sample_candidates(frontiers, len_candidates)

        # --- Auction Phase ---
        if robot.neighbors:
            candidates_to_auction = robot.target_candidates[:]
            for point in candidates_to_auction:
                robot.auctions_started += 1
                best_robot = robot
                best_utility = robot.utility(point)

                for neighbor in robot.neighbors:
                    utility = neighbor.utility(point)
                    if utility > best_utility:
                        best_utility = utility
                        best_robot = neighbor

                if best_robot != robot:
                    robot.auctions_lost_to[best_robot.id] += 1
                    if point in robot.target_candidates:
                        robot.target_candidates.remove(point)
                    if point not in best_robot.target_candidates:
                        best_robot.target_candidates.append(point)
                else:
                    robot.auctions_kept += 1

        # --- Target Selection ---
        if not robot.target_candidates:
            return None, None

        while robot.target_candidates:
            target = robot.find_target()
            path = astar(robot.pos, target, robot.map.explored_map)
            if path:
                return path[1:], target
            robot.target_candidates.remove(target)

        return None, None
    
    # Strategy 3 — Competitive
    def competitive(self, frontiers):
        if not frontiers:
            return None, None

        robot = self.robot
        neighbors = robot.neighbors
        all_robots = [robot] + neighbors
        num_tasks = len(all_robots)

        # Sample frontiers for allocation
        tasks = robot.sample_tasks(frontiers, num_tasks)
        if not tasks:
            return None, None

        # Prepare positions
        robot_positions = [r.pos for r in all_robots]
        task_positions = tasks

        # GA-based task allocation using info_gain - α * distance
        best_allocation, best_utility = run_ga(
            robots=robot_positions,
            tasks=task_positions,
            map_obj=robot.map,          # your map instance providing info_gain
            pop_size=50,
            generations=50,
            sense_radius=2,             # radius for info_gain
            alpha=1.0,                  # cost weighting (tune as needed)
            verbose=False
        )

        # Pick task assigned to this robot (first in list)
        task = task_positions[best_allocation[0]]

        # Plan path
        path = astar(robot.pos, task, robot.map.explored_map)
        if path:
            return path[1:], task
        return None, None