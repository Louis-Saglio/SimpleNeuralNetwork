"""
Attempt to build a more biologic inspired genetic algorithm data-structure module
"""

from neural_network import Perceptron, NeuralNetwork
from utils import rpprint


class Being:
    def __init__(self, genome):
        self.genome = genome
        self.network = NeuralNetwork(self.decode())

    def decode(self):
        network = []
        previous_gene_size = 0
        layer = []
        for gene in self.genome[0]:
            if previous_gene_size != len(gene):
                if layer:
                    network.append(layer)
                layer = []
            layer.append(Perceptron(int(gene[0]), [int(weight) for weight in gene[1:]], self.genome[1]))
            previous_gene_size = len(gene)
        network.append(layer)
        return network


def test():
    b = Being((["4589", "7415", "212", "089", "277", "0254"], lambda x: x))
    rpprint(b.network)
    assert b.network.run((-1, 0, 1)) == [1188]


if __name__ == "__main__":
    test()
