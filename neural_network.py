from typing import Union, Callable, Collection

Number = Union[int, float]


class Neuron:
    def __init__(self, bias: Number, weights: Collection[Number, ...], activation_function: Callable[[Number], Number]):
        self.activation_function = activation_function
        self.bias = bias
        self.weights = weights

    def run(self, inputs: Collection[Number, ...]) -> Number:
        assert len(inputs) == len(self.weights)
        return self.activation_function(
            sum([input_ * weight for input_, weight in zip(inputs, self.weights)]) + self.bias
        )


class Network:
    def __init__(self, layers: Collection[Collection[Neuron, ...], ...]):
        self.layers = layers

    def run(self, input_: Collection[Number, ...]) -> Collection[Number]:
        for layer in self.layers:
            new_input = []
            for neuron in layer:
                new_input.append(neuron.run(input_))
            input_ = new_input
        return input_


def test():
    n = Neuron(5, (2, 3, 6), lambda x: x * 2)
    nbr = n.run((1, 2, 3))
    assert nbr == 62, nbr

    nt = Network(
        (
            (
                Neuron(0, (1, 2), lambda x: x),
                Neuron(0, (2, 1), lambda x: x-1)
            ),
            (
                Neuron(1, (1, 3), lambda x: 2*x),
            )
        )
    )
    print(nt.run((3, 4)))


if __name__ == "__main__":
    test()
