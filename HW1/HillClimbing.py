import numpy as np

MAX = 2000


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


class HillClimbing:
    def __init__(self):
        self.precision = 3

    def return_N_n(self, a, b):
        N = (b - a) * np.power(10, self.precision)
        n = np.ceil(np.log2(N))
        return int(N), int(n)

    def binary_decoding(self, binary_point, a, b):
        decimal = 0
        for index in range(0, len(binary_point)):
            decimal += binary_point[index] * np.power(2, len(binary_point) - index - 1)
        real = a + decimal * (b - a) / (np.power(2, len(binary_point)) - 1)
        return real

    def return_float_point(self, candidate, n, a, b, number_variables):
        candidate_array = [candidate[i * int(n):(i + 1) * int(n)] for i in range(0, number_variables)]
        candidate_np_array = np.array(candidate_array)
        # print(candidate_np_array, candidate_np_array.shape)
        float_points = np.zeros(number_variables)
        for index in range(0, number_variables):
            float_points[index] = self.binary_decoding(candidate_np_array[index], a, b)
        return float_points

    def generate_initial_candidate(self, n, number_variables):
        candidate = np.zeros(shape=number_variables * n)
        for i in range(0, number_variables * n):
            bit = np.round(np.random.uniform(0, 1))
            candidate[i] = bit
        return candidate

    def test_Griewangks(self, candidate, number_variables):
        a = -600.0
        b = 600.0
        N, n = self.return_N_n(a, b)
        # print(N, n)
        float_points = self.return_float_point(candidate, n, a, b, number_variables)
        # print(float_points)
        function_termen_1 = np.sum(np.power(float_points, 2)) / 4000
        function_termen_2 = 1
        for index in range(0, number_variables):
            function_termen_2 *= np.cos(float_points[index] / (index + 1))
        function_termen_2 += 1
        result = function_termen_1 - function_termen_2
        return result

    def generate_HammingNeighbours(self, candidate, n, number_variables):
        candidate_neighbours = np.zeros(shape=(n * number_variables, n * number_variables))
        for index in range(0, n * number_variables):
            candidate_neighbours[index] = switch_bit(candidate, index)
        return candidate_neighbours

    def select_v_n_from_neighbours(self, val_v_c, neighbours, number_variables, method="first_improvement"):
        if method == "first_improvement":
            for index in range (0, neighbours.shape[0]):
                if self.test_Griewangks(neighbours[index], number_variables) - val_v_c <= pow(10, -3):
                    return neighbours[index]
        elif method == "best_improvement":
            val_neighbours = []
            for neighbour in neighbours:
                val_neighbours.append(self.test_Griewangks(neighbour, number_variables))
            index = np.argmin(val_neighbours)
            return neighbours[index]

    def HillClimbingAlgorithm(self, n, number_variables):
        t = 0
        best = [1 for index in range (0, n*number_variables)]
        best_value = self.test_Griewangks(best, number_variables)
        while t < MAX:
            local = False
            v_c = self.generate_initial_candidate(n, number_variables)
            val_v_c = self.test_Griewangks(v_c, number_variables)
            while not local:
                neighbours = self.generate_HammingNeighbours(v_c, n, number_variables)
                # v_n = self.select_v_n_from_neighbours(val_v_c,neighbours, number_variables, method="first_improvement")
                v_n = self.select_v_n_from_neighbours(val_v_c, neighbours, number_variables, method="best_improvement")
                if self.test_Griewangks(v_n, number_variables) - val_v_c <= pow(10,-3):
                    v_c = v_n
                    print("Am gasit o valoare mai buna: {0}".format(self.test_Griewangks(v_n, number_variables)))
                else:
                    local = True
            t += 1
            if val_v_c - best_value <= pow(10,-self.precision):
                best = v_c
            print("Iteratia {0}, cea mai buna valoare gasita: {1}".format(t,best_value))
        return best


if __name__ == "__main__":
    h = HillClimbing()
    print(h.HillClimbingAlgorithm(21, 2))
    # print(h.generate_HammingNeighbours([1, 0,0,0,1,1],3,2))
