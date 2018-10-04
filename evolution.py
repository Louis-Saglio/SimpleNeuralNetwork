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

    def __repr__(self):
        string = []
        maxi = max([len(layer) for layer in self.layers])
        for i, layer in enumerate(self.layers):
            string.append(str(i) + '  ' + ' ' * (maxi - len(layer)) + ' '.join(['o' for _ in range(len(layer))]))
        return ('-' * maxi * 2) + '---\n' + '\n'.join(string)

    def mutate(self):
        layer = choice(self.layers)
        neuron = choice(layer)
        index = choice(range(len(neuron.weights)))
        neuron.weights[index] = random() * 9

    @staticmethod
    def get_fusion_points(layers1: List[List[Neuron]], layers2: List[List[Neuron]]) -> List[Tuple[int, int]]:
        fusion_points = []
        for i, layer1 in enumerate(layers1[:-2]):
            for n, layer2 in enumerate(layers2[:-2]):
                if len(layer1) == len(layer2):
                    fusion_points.append((i + 1, n + 1))
        return fusion_points

    @staticmethod
    def mate(net1: Network, net2: Network) -> Network:
        fusion_points = Network.get_fusion_points(net1.layers, net2.layers)
        if not fusion_points:
            return Network(choice((net1.layers, net2.layers)))
        fusion_point = choice(fusion_points)
        if randint(0, 1) == 0:
            net1, net2 = net2, net1
            fusion_point = fusion_point[::-1]
        layer = net1.layers[:fusion_point[0]] + net2.layers[fusion_point[1]:]
        return Network(layer)


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
