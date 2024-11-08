import random


def create_distance_matrix(num_cities):
    matrix = [[random.randint(10, 100) if i != j else 0 for j in range(num_cities)] for i in range(num_cities)]
    return matrix

def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def calculate_distance(tour, distance_matrix):
    distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    distance += distance_matrix[tour[-1]][tour[0]]  
    return distance

def evaluate_fitness(population, distance_matrix):
    fitness = []
    for tour in population:
        total_distance = calculate_distance(tour, distance_matrix)
        fitness.append(1 / total_distance) 
    return fitness


def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]


def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations):
    num_cities = len(distance_matrix)
    population = initialize_population(pop_size, num_cities)
    
    for generation in range(num_generations):
        fitness = evaluate_fitness(population, distance_matrix)
        
    
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population
 
    final_fitness = evaluate_fitness(population, distance_matrix)
    best_tour_index = final_fitness.index(max(final_fitness))
    best_tour = population[best_tour_index]
    best_distance = calculate_distance(best_tour, distance_matrix)
    
    return best_tour, best_distance

def main():
    num_cities = 5
    pop_size = 10
    num_generations = 100
    

    distance_matrix = create_distance_matrix(num_cities)
    

    best_tour, best_distance = genetic_algorithm_tsp(distance_matrix, pop_size, num_generations)
    

    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)

if __name__ == "__main__":
    main()
