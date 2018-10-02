from __future__ import annotations

from random import randint, random, choice
from typing import List

from neural_network import Network as Network_, Neuron


def activation_function(x):
    return x


def get_random_nt(nbr=30) -> List[Network]:
    networks = []
    for i in range(nbr):
        layer_number = randint(1, 9)
        nt_structure = []
        layer_size = 2
        for layer_num in range(layer_number):
            last = layer_num == layer_number - 1
            layer = []
            layer_size, old_layer_size = 1 if last else randint(1, 9), layer_size
            for neuron_num in range(layer_size):
                layer.append(Neuron(random() * 9, [random() * 9 for _ in range(old_layer_size)], activation_function))
            nt_structure.append(layer)
        networks.append(Network(nt_structure))
    return networks


class Network(Network_):
    def __call__(self, nbr1, nbr2):
        return self.run((nbr1, nbr2))[0]

    def remove_layer(self):
        for i in range(len(self.layers) - 1):
            print(len(self.layers[i]), len(self.layers[i + 1]))
            if len(self.layers[i]) == len(self.layers[i+1]):
                print(self.layers[i], self.layers[i + 1])
                break
        else:
            return
        self.layers.remove(self.layers[i])

    def mutate(self):
        layer = choice(self.layers)
        neuron = choice(layer)
        index = choice(range(len(neuron.weights)))
        neuron.weights[index] = random() * 9


    def clone(self):
        return Network([layer.copy() for layer in self.layers])


def get_loss(nbr1, nbr2, sum_func):
    return abs((sum_func(nbr1, nbr2) - (nbr1 + nbr2))) / (abs(nbr1 + nbr2) or 0.0000001)


def test():
    assert get_loss(2, 3, lambda x, y: x + y + 5) == 1
    pop = get_random_nt()
    for i in range(1000):
        for nt in pop:
            if randint(0, 9) == 0:
                nt.mutate()
            print(get_loss(5, -9, nt))
        pop = sorted(pop, key=lambda x: get_loss(randint(0, 9), randint(0, 9), x))[15:]
        pop += get_random_nt(15)


if __name__ == '__main__':
    test()
