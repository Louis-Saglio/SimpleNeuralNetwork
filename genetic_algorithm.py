import math
from neural_network import Network, Neuron


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Being:
    def __init__(self):
        self.genome = (["4589", "7415", "212", "089", "277", "0254"], sigmoid)

    def decode(self):
        network = []
        layer = []
        size = len(self.genome[0])
        for gene in self.genome:
            if len(gene) != size:
                network.append(layer)
                layer = []
            layer.append(Neuron(int(gene[0]), [int(weight) for weight in gene[1:]], self.genome[1]))