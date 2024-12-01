import random

def initialize_population(pop_size, string_length):
    return [''.join(random.choice('01') for _ in range(string_length)) for _ in range(pop_size)]

def calculate_fitness(individual):
    return individual.count('1')

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population, probabilities, k=1)[0]
    parent2 = random.choices(population, probabilities, k=1)[0]
    return parent1, parent2

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def mutate(individual, mutation_rate):
    return ''.join(
        bit if random.random() > mutation_rate else ('0' if bit == '1' else '1')
        for bit in individual
    )


def genetic_algorithm(string_length, pop_size, num_generations, mutation_rate):
    population = initialize_population(pop_size, string_length)
    
    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        next_generation = []

        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            next_generation.extend([offspring1, offspring2])
        
        population = next_generation

        best_individual = max(population, key=calculate_fitness)
        print(f"Generation {generation + 1}: Best fitness = {calculate_fitness(best_individual)}, Best string = {best_individual}")

    return max(population, key=calculate_fitness)

best_solution = genetic_algorithm(string_length=10, pop_size=20, num_generations=50, mutation_rate=0.01)
print("Optimal Solution:", best_solution)