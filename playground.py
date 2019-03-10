from random import randint
from statistics import mean

from evaluation import evaluate_sum, evaluate_sqrt
from evolution import get_random_neural_network_population

gen_nbr = 2000

# generate
pop_size = 2000
population = get_random_neural_network_population(pop_size, 1, 1, lambda x: x)


evaluate = evaluate_sqrt

for _ in range(gen_nbr):
    try:
        # select
        population = population[: pop_size // 2]
        population += [individual.copy() for individual in population]
        # mutate
        [individual.mutate() for individual in population if randint(0, 100) == 0]
        # sort
        population = sorted(population, key=lambda individual: evaluate(individual))

        scores = [evaluate(individual) for individual in population]
        print(round(mean(scores), 2), round(min(scores), 2))
    except Exception as e:
        print(e)
        # data corruption may happen
        if input("Continue ? (y/n)") == "n":
            break
