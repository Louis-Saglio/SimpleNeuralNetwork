from __future__ import annotations

import math
from random import choices, random
from typing import Union, Callable, List

Number = Union[int, float]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Perceptron:
    def __init__(self, activation_function: Callable[[Number], Number], inputs: List[Perceptron]):
        self.inputs = inputs
        self.activation_function = activation_function
        self.current_value: Number = 0
        self.old_value: Number = 0
        self.weights: List[Number] = [random() for _ in self.inputs]

    def run(self):
        self.current_value = self.activation_function(
            sum([i.old_value for i, weight in zip(self.inputs, self.weights)])
        )

    def update(self):
        self.old_value = self.current_value

    @property
    def id(self):
        return str(id(self))[-5:]

    def add_as_input(self, perceptron: Perceptron):
        self.weights.append(random())
        self.inputs.append(perceptron)

    def __repr__(self):
        return (
            f"P(id={self.id}, current_value={self.current_value}, old_value={self.old_value},"
            f" weights={[round(w, 2) for w in self.weights]}, {[i.id for i in self.inputs]})"
        )


class Network:
    @staticmethod
    def init_perceptrons(nbr: int) -> List[Perceptron]:
        perceptrons = []
        for _ in range(nbr):
            perceptrons.append(Perceptron(sigmoid, []))
        for perceptron in perceptrons:
            for new_input in [p for p in choices(perceptrons, k=5) if p is not perceptron]:
                perceptron.add_as_input(new_input)
        return perceptrons

    def __init__(self):
        self.perceptrons = self.init_perceptrons(10)

    def feedforward(self):
        for perceptron in self.perceptrons:
            perceptron.run()
        for perceptron in self.perceptrons:
            perceptron.update()


if __name__ == "__main__":
    network = Network()
    while True:
        network.feedforward()
