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
    def __init__(self, chromosome, robots, tasks):
        self.chromosome = chromosome
        self.robots = robots
        self.tasks = tasks
        self.fitness = self.cal_fitness()
    
    @classmethod
    def create_chromosome(cls, robots, tasks):
        R = len(robots)
        T = len(tasks)

        # cooperative will have less robots than tasks
        if R < T:
            chrom = random.sample(range(T), R)
        else:
            chrom = list(range(T))
            random.shuffle(chrom)

        return cls(chrom, robots, tasks)
    
    def cal_fitness(self):
        total = 0
        for i, task_idx in enumerate(self.chromosome):
            total += euclidean(self.robots[i], self.tasks[task_idx])
        return total
    
    def mate(self, par2):
        R = len(self.chromosome)

        if R < 2:
            return Individual(self.chromosome[:], self.robots, self.tasks)
    
        start, end = sorted(random.sample(range(R), 2))
        child_chrom = [None]*R
        child_chrom[start:end+1] = self.chromosome[start:end+1]

        p2_genes = [gene for gene in par2.chromosome if gene not in child_chrom]
        j = 0
        for i in range(R):
            if child_chrom[i] is None:
                child_chrom[i] = p2_genes[j]
                j += 1
        return Individual(child_chrom, self.robots, self.tasks)
    
    def mutate(self):
        R = len(self.chromosome)
        if R < 2:
            # Nothing to swap, skip mutation
            return
        i, j = random.sample(range(len(self.chromosome)), 2)
        self.chromosome[i], self.chromosome[j] = self.chromosome[j], self.chromosome[i]
        self.fitness = self.cal_fitness()

# -------------------------------
# GA Algorithm Function
# -------------------------------
def run_ga(robots, tasks, pop_size=100, generations=100, mutation_prob=0.1, verbose=False):
    """
    Run the GA for task allocation.

    Returns:
        best_allocation: list of task indices assigned to robots
        best_distance: total distance
    """
    
    # Initialize population
    population = [Individual.create_chromosome(robots, tasks) for _ in range(pop_size)]

    for gen in range(generations):
        population.sort(key=lambda x: x.fitness)
        best = population[0]
        if verbose:
            print(f"Gen {gen}: Best distance = {best.fitness:.2f}")
        
        # Elitism: top 10%
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
    return best.chromosome, best.fitness