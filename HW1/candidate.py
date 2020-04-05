import random

import numpy as np

from HW1.params import FunctionParam


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


class CandidateSelector:
    def __init__(self, function_param: FunctionParam):
        self.number_variables = function_param.number_variables
        self.n = function_param.n

    def generate_initial_candidate(self):
        candidate = np.zeros(shape=self.number_variables * self.n)
        for i in range(0, self.number_variables * self.n):
            bit = np.round(np.random.uniform(0, 1))
            candidate[i] = bit
        return candidate

    def generate_HammingNeighbours(self, candidate):
        candidate_neighbours = np.zeros(
            shape=(self.n * self.number_variables, self.n * self.number_variables)
        )
        for index in range(0, self.n * self.number_variables):
            candidate_neighbours[index] = switch_bit(candidate, index)
        return candidate_neighbours

    def cross_over(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        crossover_point = random.randint(1, self.number_variables * self.n - 1)

        for i in range(0, offspring_size[0]):
            parent1_idx = i % parents.shape[0]
            parent2_idx = (i + 1) % parents.shape[0]
            offspring[i, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[i, crossover_point:] = parents[parent2_idx, crossover_point:]

        return offspring

    def mutation(self, population, probability):
        for i in range(len(population)):
            population[i] = self._mutation(population[i], probability)

    def _mutation(self, candidate, probability):
        _candidate = candidate[:]
        for i in range(0, len(candidate)):
            if random.random() < probability:
                _candidate = switch_bit(_candidate, i)
        return _candidate
