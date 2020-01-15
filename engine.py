"""
Evolution algorithm engine
"""
from random import random

from evolution import get_random_neural_network_population


def run(pop_size: int, gen_nbr: int, mutation_probability: float, evaluate):

    population = get_random_neural_network_population(pop_size, 2, 1)
    population_history = [population]

    for _ in range(gen_nbr):
        try:
            # reproduce and mutate
            new_population = [individual.copy() for individual in population]
            [individual.mutate() for individual in new_population if random() <= mutation_probability]
            population += new_population
            # select fittest
            population = sorted(population, key=lambda individual: evaluate(individual))[:pop_size]
            population_history.append(population)
        except Exception as e:
            print(e)
            # data corruption may happen
            if input("Continue ? (y/n)") == "n":
                break
    return population_history
