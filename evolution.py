"""
Genetic algorithm layer built on the top of the neural_network module
in order to adapt NeuralNetwork instances into individuals of a population of a genetic algorithm
Currently it supports only NN with 2 inputs and 1 output
"""


from __future__ import annotations

from random import randint, random, choice
from typing import List, Tuple

from neural_network import NeuralNetwork, Perceptron


def default_activation_function(x):
    return x


def get_random_neural_network(nbr: int = 30) -> List[Individual]:
    """
    Returns a list of <nbr> randomly set up Individual instances
    supporting an input of size 2 and giving an output of size 1
    """
    networks = []
    for i in range(nbr):
        layer_number = randint(1, 9)
        nt_structure = []
        layer_size = 2
        for layer_num in range(layer_number):
            last = layer_num == layer_number - 1
            layer = []
            layer_size, old_layer_size = 1 if last else randint(2, 9), layer_size
            for neuron_num in range(layer_size):
                layer.append(
                    Perceptron(random(), [random() for _ in range(old_layer_size)], default_activation_function)
                )
            nt_structure.append(layer)
        networks.append(Individual(nt_structure))
    return networks


class Individual(NeuralNetwork):
    def __call__(self, nbr1, nbr2):
        return self.run((nbr1, nbr2))[0]

    def mutate(self):
        layer = choice(self.layers)
        neuron = choice(layer)
        index = choice(range(len(neuron.weights)))
        neuron.weights[index] = random() * 9

    @staticmethod
    def get_fusion_points(layers1: List[List[Perceptron]], layers2: List[List[Perceptron]]) -> List[Tuple[int, int]]:
        fusion_points = []
        for i, layer1 in enumerate(layers1[:-2]):
            for n, layer2 in enumerate(layers2[:-2]):
                if len(layer1) == len(layer2):
                    fusion_points.append((i + 1, n + 1))
        return fusion_points

    @staticmethod
    def mate(net1: Individual, net2: Individual) -> Individual:
        fusion_points = Individual.get_fusion_points(net1.layers, net2.layers)
        if not fusion_points:
            return Individual(choice((net1.layers, net2.layers)))
        fusion_point = choice(fusion_points)
        if randint(0, 1) == 0:
            net1, net2 = net2, net1
            fusion_point = fusion_point[::-1]
        layer = net1.layers[: fusion_point[0]] + net2.layers[fusion_point[1] :]
        return Individual(layer)


def get_loss(nbr1, nbr2, sum_func):
    return abs((sum_func(nbr1, nbr2) - (nbr1 + nbr2))) / (abs(nbr1 + nbr2) or 0.0000001)


def test():
    assert get_loss(2, 3, lambda x, y: x + y + 5) == 1
    len_pop = 100
    pop = get_random_neural_network(len_pop)
    for i in range(10):
        for nt in pop:
            if randint(0, 5) == 0:
                nt.mutate()
            print(get_loss(5, -9, nt))
        pop = sorted(pop, key=lambda x: get_loss(randint(0, 9), randint(0, 9), x))[: len_pop // 2]
        for _ in range(len_pop // 2):
            new_net = Individual.mate(choice(pop), choice(pop))
            pop.append(new_net or get_random_neural_network(30)[0])
    for net in pop:
        print(net)
    print(repr(net))
    print(net.run((2, 2)))


if __name__ == "__main__":
    test()
