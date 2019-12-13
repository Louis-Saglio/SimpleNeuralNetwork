from statistics import mean

from matplotlib import pyplot as plt

from engine import run
from evaluation import evaluate_sum, evaluate_multiplication
from utils import batch

evaluate = evaluate_multiplication

pop_size = 1000
population_history = run(pop_size=pop_size, gen_nbr=2000, mutation_probability=0.01, evaluate=evaluate)

# with open(f"saved_{int(time.time())}", "wb") as f:
#     pickle.dump(population_history, f)


avg_list = []
best_list = []
species_count_history = []
species_repr = {}
for population in population_history:
    species_count = {}
    # print(population[0])
    scores = [evaluate(individual) for individual in population]

    avg_list.append(round(mean(scores), 2))
    best_list.append(round(min(scores), 2))

    for individual in population:
        if individual.species_id in species_count:
            species_count[individual.species_id] += 1
        else:
            species_repr[individual.species_id] = str(individual)
            species_count[individual.species_id] = 1

    species_count_history.append(species_count)

print(repr(population[0]))

print(population[0].run((-100, 100)))


plt.plot([mean(data) for data in batch(avg_list[10:], int(pop_size / 10))])
plt.title("Moyenne")
plt.show()

plt.plot([mean(data) for data in batch(best_list[10:], int(pop_size / 10))])
plt.title("Best")
plt.show()

species_count_summary = {}
for species_count in species_count_history:
    for species_id, species_size in species_count.items():
        if species_id in species_count_summary:
            species_count_summary[species_id].append((species_size / pop_size) * 50)
        else:
            species_count_summary[species_id] = [(species_size / pop_size) * 50]


for species_id, species_size_history in species_count_summary.items():
    if mean(species_size_history) > pop_size / 100:
        plt.plot(species_size_history)
        plt.title(species_repr[species_id])
        plt.show()

# while True:
#     numbers = tuple(int(i) for i in input(">>> ").split(","))
#     print(population_history[-1][0].run(numbers))
#     if numbers == (66, 66):
#         break
