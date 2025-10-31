import random
import math
from utils import astar
from nash_eq import run_ga
from collections import defaultdict

# We need a way to classify leaders and citizens first; paper said 10 robots per leader but we can tune this
def classify_leaders_and_citizens(robots, p=0.1):
    N = len(robots)
    target_num_leaders = max(1, int(p * N))

    # pick the leaders
    shuffled = robots[:] 
    random.shuffle(shuffled)
    
    leaders = []

    for r in shuffled:
        if any(n in leaders for n in r.neighbors):
            continue
        leaders.append(r)
        if len(leaders) >= target_num_leaders:
            break

    # This guarantees at least one leader, but we can error check just in case
    assert len(leaders) > 0, "must have at least one leader"
    
    citizens = defaultdict(list)

    for r in robots:
        if r in leaders:
            continue 

        ln = [L for L in leaders if L in r.neighbors]

        #if there is a leader neighbor, pick from neighbors, else pick absolute closest
        if ln:
            # choose closest neighbor
            leader = min(ln, key=lambda L: math.dist(L.pos, r.pos))
        else:
            # choose closest non-neighbor
            leader = min(leaders, key=lambda L: math.dist(L.pos, r.pos))

        citizens[leader].append(r)

    return leaders, citizens

def aggregate_votes(votes, V, all_tasks):
    vote_counts = defaultdict(lambda: defaultdict(int))
    for vote in votes:
        for r, t in vote.items():
            vote_counts[r][t] += 1

    assignment = {}
    for r, task_dict in vote_counts.items():
        best_task = max(task_dict, key=task_dict.get)  # plurality, which was used in the paper, but we could maybe enhance this
        assignment[r] = best_task

    unassigned = [r for r in V if r not in assignment]

    assigned_tasks = set(assignment.values())
    remaining_tasks = [t for t in all_tasks if t not in assigned_tasks]

    if unassigned:
        robot_pos = [r.pos for r in unassigned]
        chrom, _ = run_ga(robot_pos, remaining_tasks)
        for i, r in enumerate(unassigned):
            assignment[r] = remaining_tasks[chrom[i]]

    return assignment

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
        elif self.strategy == "cooperative":
            raise ValueError(f"Assigning cooperative target at independent location") #realistically this should not be called
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

        # GA-based task allocation
        robot_positions = [r.pos for r in all_robots]
        task_positions = tasks
        
        best_allocation, _ = run_ga(robot_positions, task_positions, generations=50, pop_size=50)

        # Pick task assigned to this robot (first in list)
        task = task_positions[best_allocation[0]]

        # Plan path
        path = astar(robot.pos, task, robot.map.explored_map)
        if path:
            return path[1:], task
        return None, None

    # Strategy 4 - Cooperative
    def cooperative(self):
        leaders, citizens = classify_leaders_and_citizens(self.robot.robots)   
        for leader in leaders:
            V = [leader] + citizens[leader]
            num_agents = len(V)

            frontiers = leader.find_frontiers()
            tasks = leader.sample_tasks(frontiers, num_agents)
            if not tasks:
                continue

            votes = []
            # Idea: we find the nash equilibrium based on a vote
            # The vote itself is just what the robot thinks is the best tasks for each cluster
            for r in V:
                vote_scores = r.vote_on_tasks(tasks)  # dict {r_i: best task for r_i according to r}
                votes.append(vote_scores)

            allocation = aggregate_votes(votes, V, tasks)

            for i, r in enumerate(V):
                assigned_task = allocation[r]
                path = astar(r.pos, assigned_task, r.map.explored_map)
                if path:
                    r.path = path[1:]
                    r.target = assigned_task
    
   

        

