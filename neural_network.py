"""
Data structures to emulate simple feed forward & layer based neural networks
"""


from copy import deepcopy
from math import exp
from typing import Union, Callable, Collection, List

from utils import obj_to_json

Number = Union[int, float]


class Perceptron:
    def __init__(self, bias: Number, weights: Collection[Number]):
        self.bias = bias
        self.weights = weights
        self.id = hash((self.bias, tuple(self.weights)))

    def run(self, inputs: Collection[Number]) -> Number:
        assert len(inputs) == len(self.weights), (inputs, self.weights)
        return sum([input_ * weight for input_, weight in zip(inputs, self.weights)]) + self.bias

    def __repr__(self):
        return f"Perceptron{{weights : {self.weights}, bias : {self.bias}}}"


class NeuralNetwork:
    def __init__(self, layers: Collection[Collection[Perceptron]]):
        self.layers = layers

    def run(self, input_: Collection[Number]) -> List[Number]:
        for layer in self.layers:
            new_input = []
            for neuron in layer:
                new_input.append(neuron.run(input_))
            input_ = new_input
        return input_

    def __repr__(self):
        return obj_to_json(self.layers)

    def __str__(self):
        string = []
        maxi = max([len(layer) for layer in self.layers])
        for i, layer in enumerate(self.layers):
            string.append(
                str(i)
                + f" {len(layer)}"
                + "  "
                + " " * (maxi - len(layer))
                + " ".join(["o" for _ in range(len(layer))])
            )
        return ("-" * maxi * 2) + "-----\n" + "\n".join(string)

    def copy(self):
        return self.__class__(deepcopy(self.layers))


def test():
    n = Perceptron(5, (2, 3, 6))
    nbr = n.run((1, 2, 3))
    assert nbr == 31, nbr

    nt = NeuralNetwork(
        ((Perceptron(0, (1, 2)), Perceptron(0, (2, 1)), Perceptron(0, (2, 1))), (Perceptron(1, (1, 2, 3)),))
    )
    print(nt.run((3, 4)))
    print(nt)
    print(repr(nt))


def test2():
    perceptrons = {"p1": Perceptron(0, [1, -2]), "p2": Perceptron(0, [-3, 4]), "p3": Perceptron(0, [5, -6])}

    nt = NeuralNetwork(((perceptrons["p1"], perceptrons["p2"], perceptrons["p3"]),))

    while True:
        in_put = [float(i) for i in input("Choose input").split(",")]
        out = nt.run(in_put)

        for i, nbr in enumerate(out):
            out[i] = 1 / (1 + exp(-nbr))

        perceptron = perceptrons[input(f"Choose perceptron to change : {' '.join(perceptrons.keys())} >>> ")]
        print(f"current weights : {perceptron.weights}")
        # noinspection PyTypeHints
        perceptron.weights: List
        p_index = int(input("Choose weight to change >>> "))
        new_weight = float(input("Choose new weight"))
        perceptron.weights[p_index] = new_weight

        print(out)


if __name__ == "__main__":
    test2()
