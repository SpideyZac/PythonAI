"""
PythonAI - dep
v 0.0.1
By: Spidey Zac

PythonAI GeneticAlgorithm file
"""

__version__ = "0.0.1"

import numpy as np
from random import uniform as random

floor = np.math.floor
randmatrix = np.random.uniform

class GeneticAlgorithm:
    def __init__(self, network: list[int]) -> None:
        if len(network) < 2:
            raise Exception("Network Must Have At Least Two Layers")

        for i in range(len(network)):
            if network[i] <= 0:
                raise Exception("Network Layer {} Cannot have a size of 0 or less".format(i))

        self.network = network

        self.weights = np.zeros(len(network) - 1).tolist()
        self.biases = np.zeros(len(network) - 1).tolist()

        for i in range(len(network) - 1):
            self.weights[i] = randmatrix(-1, 1, (network[i + 1], network[i]))

        for i in range(1, len(network)):
            self.biases[i - 1] = randmatrix(-1, 1, (network[i], 1))

    def feedForward(self, inputs: list[float]) -> float:
        ins = np.zeros((len(inputs), 1))
        for i in range(len(inputs)):
            ins[i][0] = inputs[i]

        inputs = ins

        for i in range(len(self.weights)):
            inputs = np.matmul(self.weights[i], inputs)
            inputs = np.add(inputs, self.biases[i])

        return inputs.flatten()

    def crossover(self, parent2: 'GeneticAlgorithm') -> 'GeneticAlgorithm':
        child = GeneticAlgorithm(self.network)
        child.weights = self.weights
        child.biases = self.biases

        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random(0, 1) <= 0.5:
                        child.weights[i][j][k] = parent2.weights[i][j][k]

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                for k in range(len(child.biases[i][j])):
                    if random(0, 1) <= 0.5:
                        child.weights[i][j][k] = parent2.biases[i][j][k]

        return child

    def mutate(self) -> None:
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += random(-1, 1)

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                for k in range(len(self.biases[i][j])):
                    self.biases[i][j][k] += random(-1, 1)