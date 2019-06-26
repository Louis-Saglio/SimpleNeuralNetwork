import dill as pickle
import time
from random import randint
from statistics import mean

from evaluation import evaluate_sum, evaluate_sqrt, evaluate_and, evaluate_is_divisible_by
from evolution import get_random_neural_network_population
from matplotlib import pyplot as plt

from utils import batch

gen_nbr = 1000

# generate
pop_size = 2000
population = get_random_neural_network_population(pop_size, 2, 1)


evaluate = evaluate_sum

avg_list = []
best_list = []

population_history = []

for _ in range(gen_nbr):
    try:
        population_history.append(population)
        # select
        population = population[: pop_size // 2]
        population += [individual.copy() for individual in population]
        # mutate
        [individual.mutate() for individual in population if randint(0, 100) == 0]
        # sort
        population = sorted(population, key=lambda individual: evaluate(individual))

        scores = [evaluate(individual) for individual in population]
        avg = round(mean(scores), 2)
        best = round(scores[0], 2)
        # print(avg, best)
        avg_list.append(avg)
        best_list.append(best)
    except Exception as e:
        print(e)
        # data corruption may happen
        if input("Continue ? (y/n)") == "n":
            break

plt.plot([mean(data) for data in batch(avg_list[10:], 20)])
plt.title("Moyenne")
plt.show()

plt.plot([mean(data) for data in batch(best_list, 10)])
plt.title("Best")
plt.show()

# with open(f"saved_{int(time.time())}", "wb") as f:
#     pickle.dump(population_history, f)
