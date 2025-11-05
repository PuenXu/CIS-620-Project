import random
import math

# -------------------------------
# Helper Functions
# -------------------------------
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# -------------------------------
# Individual Class
# -------------------------------
class Individual:
    def __init__(self, chromosome, robots, tasks, map_obj=None, sense_radius=2, alpha=1.0):
        """
        chromosome: list mapping robot i -> task index
        robots: list of (x, y)
        tasks: list of (x, y)
        map_obj: object providing info_gain(point, radius)
        sense_radius: sensing radius used for info_gain
        alpha: weighting factor for distance cost
        """
        self.chromosome = chromosome
        self.robots = robots
        self.tasks = tasks
        self.map_obj = map_obj
        self.sense_radius = sense_radius
        self.alpha = alpha
        self.fitness = self.cal_fitness()
    
    @classmethod
    def create_chromosome(cls, robots, tasks, map_obj=None, sense_radius=2, alpha=1.0):
        chrom = list(range(len(tasks)))
        random.shuffle(chrom)
        return cls(chrom, robots, tasks, map_obj, sense_radius, alpha)
    
    def cal_fitness(self):
        """
        Define fitness as negative total utility
        (since GA minimizes fitness but we want to maximize utility).
        Utility = info_gain - alpha * distance
        """
        total_utility = 0.0
        for i, task_idx in enumerate(self.chromosome):
            robot_pos = self.robots[i]
            task_pos = self.tasks[task_idx]
            
            dist = euclidean(robot_pos, task_pos)
            
            info_gain = 0.0
            if self.map_obj is not None:
                info_gain = self.map_obj.info_gain(task_pos, self.sense_radius)
            
            total_utility += info_gain - self.alpha * dist

        # Negate because GA minimizes fitness
        return -total_utility
    
    def mate(self, par2):
        R = len(self.chromosome)
        if R < 2:
            return Individual(self.chromosome[:], self.robots, self.tasks, self.map_obj, self.sense_radius, self.alpha)
    
        start, end = sorted(random.sample(range(R), 2))
        child_chrom = [None]*R
        child_chrom[start:end+1] = self.chromosome[start:end+1]

        p2_genes = [gene for gene in par2.chromosome if gene not in child_chrom]
        j = 0
        for i in range(R):
            if child_chrom[i] is None:
                child_chrom[i] = p2_genes[j]
                j += 1
        
        return Individual(child_chrom, self.robots, self.tasks, self.map_obj, self.sense_radius, self.alpha)
    
    def mutate(self):
        R = len(self.chromosome)
        if R < 2:
            return
        i, j = random.sample(range(R), 2)
        self.chromosome[i], self.chromosome[j] = self.chromosome[j], self.chromosome[i]
        self.fitness = self.cal_fitness()

# -------------------------------
# GA Algorithm Function
# -------------------------------
def run_ga(robots, tasks, map_obj=None, pop_size=100, generations=100, mutation_prob=0.1,
           sense_radius=2, alpha=1.0, verbose=False):
    """
    Run the GA for multi-robot task allocation.
    
    Args:
        robots: list of (x, y) robot positions
        tasks: list of (x, y) task positions
        map_obj: provides info_gain(point, radius)
        pop_size: number of individuals in population
        generations: number of generations
        mutation_prob: probability of mutation
        sense_radius: sensing radius for info_gain
        alpha: weight for distance cost
        verbose: print progress
    Returns:
        best_allocation: list mapping robot i -> task index
        best_utility: total utility (info_gain - alpha * distance)
    """
    
    # Initialize population
    population = [
        Individual.create_chromosome(robots, tasks, map_obj, sense_radius, alpha)
        for _ in range(pop_size)
    ]

    for gen in range(generations):
        population.sort(key=lambda x: x.fitness)
        best = population[0]
        if verbose:
            best_utility = -best.fitness  # reverse sign
            print(f"Gen {gen}: Best utility = {best_utility:.2f}")
        
        # Elitism: keep top 10%
        new_pop = population[:pop_size//10]

        # Generate offspring
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:pop_size//2], 2)
            child = p1.mate(p2)
            if random.random() < mutation_prob:
                child.mutate()
            new_pop.append(child)

        population = new_pop

    # Return final best solution
    best = min(population, key=lambda x: x.fitness)
    best_utility = -best.fitness
    return best.chromosome, best_utility