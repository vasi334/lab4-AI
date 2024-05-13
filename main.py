from utils.PSO import PSO
from utils.evolution_npl import evolutionary_algorithm


def main():

    # best_solution, best_fitness = evolutionary_algorithm(n_dimensions=2, population_size=500,
    #                                                      num_generations=1000, num_parents=100,
    #                                                      crossover_rate=1, mutation_rate=1)
    # print("Best solution found:", best_solution)
    # print("Fitness:", best_fitness)

    best_solution, best_fitness = PSO(num_dimensions=2, num_particles=20, inertia_weight=0.5,
                                      cognitive_weight=0.5, social_weight=0.5, max_iterations=100,
                                      domain_min=-32.768, domain_max=32.768)
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    main()
