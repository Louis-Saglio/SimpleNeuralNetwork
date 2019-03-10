"""
Data structures to emulate simple feed forward neural networks
"""


from copy import deepcopy
from typing import Union, Callable, Collection, List

from utils import obj_to_json

Number = Union[int, float]


class Perceptron:
    def __init__(self, bias: Number, weights: Collection[Number], activation_function: Callable[[Number], Number]):
        self.activation_function = activation_function
        self.bias = bias
        self.weights = weights

    def run(self, inputs: Collection[Number]) -> Number:
        assert len(inputs) == len(self.weights), (inputs, self.weights)
        return self.activation_function(
            sum([input_ * weight for input_, weight in zip(inputs, self.weights)]) + self.bias
        )

    def __repr__(self):
        return f"Perceptron{{weights : {self.weights}, bias : {self.bias}, activation : {self.activation_function}}}"


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
    n = Perceptron(5, (2, 3, 6), lambda x: x * 2)
    nbr = n.run((1, 2, 3))
    assert nbr == 62, nbr

    nt = NeuralNetwork(
        (
            (
                Perceptron(0, (1, 2), lambda x: x),
                Perceptron(0, (2, 1), lambda x: x - 1),
                Perceptron(0, (2, 1), lambda x: x - 1),
            ),
            (Perceptron(1, (1, 2, 3), lambda x: 2 * x),),
        )
    )
    print(nt.run((3, 4)))
    print(nt)
    print(repr(nt))


if __name__ == "__main__":
    test()
