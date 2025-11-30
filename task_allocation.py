import random
import math
from utils import astar
from nash_eq import run_ga
from collections import defaultdict

PREFERENCE_FLOOR = -1e6
TRUST_ETA = 0.2
TRUST_EPS = 1e-3

# We need a way to classify leaders and citizens first; paper said 10 robots per leader but we can tune this
def classify_leaders_and_citizens(robots, p=0.1):
    N = len(robots)
    target_num_leaders = max(1, int(p * N))

    # pick the leaders
    shuffled = [r for r in robots if not getattr(r, "is_malicious", False)]
    if len(shuffled) < target_num_leaders:
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

def plurality_scores(weighted_ballots, tasks):
    counts = defaultdict(float)
    for weight, ballot in weighted_ballots:
        choice = ballot.get("top")
        if choice in tasks:
            counts[choice] += weight
    return counts
def compute_preference_scores(weighted_ballots, tasks):
    if not tasks:
        return {}
    scores = plurality_scores(weighted_ballots, tasks)
    if scores:
        return scores
    return {task: 0.0 for task in tasks}

def maximize_preferences(robots, tasks, preference_scores, pop_size=60, generations=60, mutation_prob=0.1):
    num_robots = len(robots)
    num_tasks = len(tasks)
    if num_tasks == 0:
        return {robot: None for robot in robots}

    # If there are fewer tasks than robots, allow reuse: give each robot its best task
    if num_tasks < num_robots:
        assignment = {}
        for robot in robots:
            scores = preference_scores.get(robot, {})
            if scores:
                task = max(tasks, key=lambda t: scores.get(t, PREFERENCE_FLOOR))
            else:
                task = tasks[0] if tasks else None
            assignment[robot] = task
        return assignment

    def random_chromosome():
        return random.sample(range(num_tasks), num_robots)

    def crossover(parent1, parent2):
        if num_robots < 2:
            return parent1[:]
        start, end = sorted(random.sample(range(num_robots), 2))
        child = [None] * num_robots
        child[start:end+1] = parent1[start:end+1]
        p2_genes = [gene for gene in parent2 if gene not in child]
        j = 0
        for i in range(num_robots):
            if child[i] is None:
                child[i] = p2_genes[j]
                j += 1
        return child

    def mutate(chrom):
        if num_robots < 2:
            return
        if num_tasks > num_robots and random.random() < 0.5:
            idx = random.randrange(num_robots)
            available = [gene for gene in range(num_tasks) if gene not in chrom]
            if available:
                chrom[idx] = random.choice(available)
        else:
            i, j = random.sample(range(num_robots), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]

    def score_chromosome(chrom):
        total = 0.0
        for i, gene in enumerate(chrom):
            robot = robots[i]
            task = tasks[gene]
            total += preference_scores.get(robot, {}).get(task, PREFERENCE_FLOOR)
        return total

    population = [random_chromosome() for _ in range(pop_size)]
    population_scores = [score_chromosome(chrom) for chrom in population]

    for _ in range(generations):
        paired = sorted(zip(population, population_scores), key=lambda item: item[1], reverse=True)
        elites = [chrom for chrom, _ in paired[:max(1, pop_size // 10)]]
        new_population = elites[:]
        while len(new_population) < pop_size:
            parents = random.sample(paired[:max(2, pop_size // 2)], 2)
            child = crossover(parents[0][0], parents[1][0])
            if random.random() < mutation_prob:
                mutate(child)
            new_population.append(child)
        population = new_population
        population_scores = [score_chromosome(chrom) for chrom in population]

    best_idx = max(range(len(population)), key=lambda i: population_scores[i])
    best_chrom = population[best_idx]

    assignment = {}
    for i, robot in enumerate(robots):
        assignment[robot] = tasks[best_chrom[i]]
    return assignment

def update_trust_weights(leader, votes, allocation):
    """Follow-the-regularized-leader (exp-grad) update for trust weights."""
    if leader is None:
        return
    if not hasattr(leader, "trust_loss"):
        leader.trust_loss = defaultdict(float)
    if not hasattr(leader, "trust_weights"):
        leader.trust_weights = defaultdict(lambda: 1.0)

    # Accumulate loss: 1 - agreement rate
    voters = set()
    for voter, vote in votes:
        voters.add(voter)
        agreements = 0
        total = 0
        for robot, assigned_task in allocation.items():
            ballot = vote.get(robot, {})
            if ballot.get("top") == assigned_task:
                agreements += 1
            total += 1
        if total == 0:
            continue
        loss = 1.0 - (agreements / total)
        leader.trust_loss[voter] += loss

    # Compute exp(-eta * cumulative loss) and normalize
    raw_weights = {}
    Z = 0.0
    for voter in voters:
        w = math.exp(-TRUST_ETA * leader.trust_loss[voter])
        w = max(w, TRUST_EPS)
        raw_weights[voter] = w
        Z += w
    if Z == 0:
        Z = TRUST_EPS * len(raw_weights)
    for voter, w in raw_weights.items():
        leader.trust_weights[voter] = max(w / Z, TRUST_EPS)

def aggregate_votes(votes, V, all_tasks, leader=None):
    if not all_tasks:
        return {r: None for r in V}

    ballots = defaultdict(list)
    for voter, vote in votes:
        trust_map = getattr(leader, "trust_weights", {}) if leader else {}
        weight = trust_map.get(voter, 1.0)
        for robot, ballot in vote.items():
            ballots[robot].append((weight, ballot))

    preference_scores = {}
    for robot in V:
        robot_ballots = ballots.get(robot, [])
        preference_scores[robot] = compute_preference_scores(robot_ballots, all_tasks)

    return maximize_preferences(V, all_tasks, preference_scores)

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

        # Prepare positions
        robot_positions = [r.pos for r in all_robots]
        task_positions = tasks

        # GA-based task allocation using info_gain - α * distance
        best_allocation, best_utility = run_ga(
            robots=robot_positions,
            tasks=task_positions,
            map_obj=robot.map,          
            pop_size=50,
            generations=50,
            sense_radius=2,           
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

    # Strategy 4 - Cooperative
    def cooperative(self):
        if self.robot.leaders is None or self.robot.citizens_map is None:
            leaders, citizens = classify_leaders_and_citizens(self.robot.robots)   
            for r in self.robot.robots:
                r.leaders = leaders
                r.citizens_map = citizens
        leaders = self.robot.leaders
        citizens = self.robot.citizens_map
        for leader in leaders:
            V = [leader] + citizens[leader]
            num_agents = len(V)

            # Sync maps with leader so frontier detection uses freshest data
            for r in V:
                leader.exchange_map(r)

            frontiers = leader.find_frontiers()
            tasks = leader.sample_tasks(frontiers, num_agents)
            if not tasks:
                continue

            votes = []
            # Idea: we find the nash equilibrium based on a vote
            # The vote itself is just what the robot thinks is the best tasks for each cluster
            for r in V:
                vote_scores = r.vote_on_tasks(tasks, cluster=V)  # dict {r_i: best task for r_i according to r}
                votes.append((r, vote_scores))

            allocation = aggregate_votes(votes, V, tasks, leader=leader)
            update_trust_weights(leader, votes, allocation)

            for i, r in enumerate(V):
                assigned_task = allocation[r]
                if assigned_task:
                    path = astar(r.pos, assigned_task, r.map.explored_map)
                    if path:
                        r.path = path[1:]
                        r.target = assigned_task
    
   

        

