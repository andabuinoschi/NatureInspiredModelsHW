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
