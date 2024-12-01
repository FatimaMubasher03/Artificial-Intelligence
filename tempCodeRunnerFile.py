import random

# Define the Chromosome class
class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
        self.calculate_fitness()

    def calculate_fitness(self):
        # Define the weights and values of each item
        weights = [10, 20, 30, 15, 25]
        values = [60, 100, 120, 75, 90]
        total_weight = sum(w * g for w, g in zip(weights, self.genes))
        total_value = sum(v * g for v, g in zip(values, self.genes))
        
        # Calculate fitness as the total value if weight is within limit
        if total_weight <= 50:
            self.fitness = total_value
        else:
            self.fitness = 0

# Define the GeneticAlgorithm class
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        # Create an initial random population of chromosomes
        return [Chromosome([random.choice([0, 1]) for _ in range(5)]) for _ in range(self.population_size)]

    def selection(self):
        # Roulette Wheel Selection
        total_fitness = sum(chromosome.fitness for chromosome in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for chromosome in self.population:
            current += chromosome.fitness
            if current >= pick:
                return chromosome

    def crossover(self, parent1, parent2):
        # Perform crossover with a set crossover rate
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1.genes) - 1)
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(child1_genes), Chromosome(child2_genes)
        else:
            return parent1, parent2

    def mutate(self, chromosome):
        # Apply mutation by flipping random bits
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]
        chromosome.calculate_fitness()

    def evolve(self):
        # Run the GA for the defined number of generations
        for generation in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = self.selection()
                parent2 = self.selection()
                
                # Perform crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                
                # Add children to the new population
                new_population.extend([child1, child2])
            
            # Replace the old population with the new one
            self.population = new_population[:self.population_size]
            
            # Track the best solution in the current generation
            best_solution = self.get_best_solution()
            print(f"Generation {generation + 1}: Best Fitness = {best_solution.fitness}")

    def get_best_solution(self):
        # Identify and return the best chromosome in the population
        return max(self.population, key=lambda chromo: chromo.fitness)

# Parameters for the Genetic Algorithm
population_size = 10
mutation_rate = 0.01
crossover_rate = 0.7
generations = 20

# Run the Genetic Algorithm
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, generations)
ga.evolve()

# Display the best solution found
best_solution = ga.get_best_solution()
print("\nBest solution found:")
print("Genes:", best_solution.genes)
print("Fitness:", best_solution.fitness)
