import numpy as np
from math import sqrt, cos

MAX = 200


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


def binary_decoding(binary_point, a, b):
    decimal = 0
    for index in range(0, len(binary_point)):
        decimal += binary_point[index] * pow(2, len(binary_point) - index - 1)
    real = a + decimal * (b - a) / (pow(2, len(binary_point)) - 1)
    return real


def return_float_point(candidate, n, a, b, number_variables):
    candidate_array = [candidate[i * int(n):(i + 1) * int(n)] for i in range(0, number_variables)]
    candidate_np_array = np.array(candidate_array)
    # print(candidate_np_array, candidate_np_array.shape)
    float_points = np.zeros(number_variables)
    for index in range(0, number_variables):
        float_points[index] = binary_decoding(candidate_np_array[index], a, b)
    return float_points


def generate_initial_candidate(n, number_variables):
    candidate = np.zeros(shape=number_variables * n)
    for i in range(0, number_variables * n):
        bit = np.round(np.random.uniform(0, 1))
        candidate[i] = bit
    return candidate


def generate_HammingNeighbours(candidate, n, number_variables):
    candidate_neighbours = np.zeros(shape=(n * number_variables, n * number_variables))
    for index in range(0, n * number_variables):
        candidate_neighbours[index] = switch_bit(candidate, index)
    return candidate_neighbours

# garbage code
class HillClimbing:
    def __init__(self):
        self.precision = 4

    def return_N_n(self, a, b):
        N = (b - a) * pow(10, self.precision)
        n = np.ceil(np.log2(N))
        return int(N), int(n)

    def test_Griewangks(self, candidate, number_variables):
        a = -600.0
        b = 600.0
        N, n = self.return_N_n(a, b)
        # print(N, n)
        float_points = return_float_point(candidate, n, a, b, number_variables)
        # print(float_points)
        sum = 0
        for float_point in float_points:
            sum += pow(float_point, 2)
        function_termen_1 = sum / 4000

        function_termen_2 = 1
        for index in range(0, number_variables):
            function_termen_2 *= cos(float_points[index] / sqrt(index + 1))

        result = function_termen_1 - function_termen_2 + 1
        return result

    def select_v_n_from_neighbours(self, val_v_c, neighbours, number_variables, method="first_improvement"):
        if method == "first_improvement":
            for index in range(0, neighbours.shape[0]):
                if self.test_Griewangks(neighbours[index], number_variables) < val_v_c:
                    return neighbours[index]
        elif method == "best_improvement":
            val_neighbours = []
            for neighbour in neighbours:
                val_neighbours.append(self.test_Griewangks(neighbour, number_variables))
            index = np.argmin(val_neighbours)
            return neighbours[index]

    def HillClimbingAlgorithm(self, n, number_variables):
        t = 0
        best = [1 for index in range(0, n * number_variables)]
        best_value = self.test_Griewangks(best, number_variables)
        while t < MAX:
            local = False
            v_c = generate_initial_candidate(n, number_variables)
            val_v_c = self.test_Griewangks(v_c, number_variables)
            while not local:
                neighbours = generate_HammingNeighbours(v_c, n, number_variables)
                # v_n = self.select_v_n_from_neighbours(val_v_c,neighbours, number_variables, method="first_improvement")
                v_n = self.select_v_n_from_neighbours(val_v_c, neighbours, number_variables, method="best_improvement")
                val_v_n = self.test_Griewangks(v_n, number_variables)
                if val_v_n < val_v_c:
                    v_c = v_n
                    val_v_c = val_v_n
                    print("Am gasit o valoare mai buna: {0}".format(self.test_Griewangks(v_n, number_variables)))
                else:
                    local = True
            t += 1
            if val_v_c < best_value:
                best = v_c
                best_value = val_v_c
            print("Iteratia {0}, cea mai buna valoare gasita: {1}".format(t, best_value))
        return best


if __name__ == "__main__":
    h = HillClimbing()
    print(h.return_N_n(-600, 600))
    print(h.HillClimbingAlgorithm(24, 30))
    # print(h.generate_HammingNeighbours([1, 0,0,0,1,1],3,2))
