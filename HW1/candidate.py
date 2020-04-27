import random

import numpy as np

from HW1.params import FunctionParam


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


def get_parents_pairs(parents):
    _parents = parents[:]
    parents_pairs = []
    parents_pairs_size = len(parents) / 2

    while len(parents_pairs) < parents_pairs_size:
        select_p1 = random.randint(0, len(_parents) - 1)
        select_p2 = random.randint(0, len(_parents) - 1)
        if select_p1 != select_p2:
            parents_pairs.append((_parents[select_p1], _parents[select_p2]))
            if select_p1 > select_p2:
                _parents.pop(select_p1)
                _parents.pop(select_p2)
            else:
                _parents.pop(select_p2)
                _parents.pop(select_p1)
    return parents_pairs


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

    def cross_over(self, parents, offspring_size, crossover_probability):
        offspring = np.empty(offspring_size)

        parents_pairs = get_parents_pairs(parents)
        parents_probabilities = [random.uniform(0, 1) for _ in range(len(parents_pairs))]

        parent_pairs_probabilities = zip(parents_pairs, parents_probabilities)

        index = 0
        for parent_pair, probability in parent_pairs_probabilities:
            if probability > crossover_probability:
                crossover_point = random.randint(1, self.number_variables * self.n - 2)
                offspring[index, 0:crossover_point] = parent_pair[0][0:crossover_point]
                offspring[index, crossover_point:] = parent_pair[1][crossover_point:]
                index += 1
                offspring[index, 0:crossover_point] = parent_pair[1][0:crossover_point]
                offspring[index, crossover_point:] = parent_pair[0][crossover_point:]
            else:
                offspring[index] = parent_pair[0]
                index += 1
                offspring[index] = parent_pair[1]
            index += 1

        return offspring

    def mutation(self, population, probability, chromosome_mutation_probability):
        for i in range(len(population)):
            if random.uniform(0, 1) < chromosome_mutation_probability:
                population[i] = self._mutation(population[i], probability)
        return population

    def _mutation(self, candidate, probability):
        _candidate = candidate[:]
        for i in range(0, len(candidate)):
            if random.random() < probability:
                _candidate = switch_bit(_candidate, i)
        return _candidate
