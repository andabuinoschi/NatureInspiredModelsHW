import numpy as np
from math import sqrt, cos, pi

MAX = 50


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


class HillClimbing:
    def return_N_n(self):
        N = (self.b1 - self.a1) * pow(10, self.precision)
        n = np.ceil(np.log2(N))
        return int(N), int(n)

    def __init__(self, function_type="Griewangk"):
        self.precision = 4
        if function_type == "Griewangk":
            self.name = "Griewangk"
            self.number_variables = 30
            self.a1 = -600
            self.b1 = 600
        elif function_type == "Rastrigin":
            self.name = "Rastrigin"
            self.number_variables = 30
            self.a1 = -5.12
            self.b1 = 5.12
        elif function_type == "Rosenbrock":
            self.name = "Rosenbrock"
            self.number_variables = 30
            self.a1 = -2.048
            self.b1 = 2.048
        self.N, self.n = self.return_N_n()

    def binary_decoding(self, binary_point):
        decimal = 0
        for index in range(0, len(binary_point)):
            decimal += binary_point[index] * pow(2, len(binary_point) - index - 1)
        real = self.a1 + decimal * (self.b1 - self.a1) / (pow(2, len(binary_point)) - 1)
        return real

    def generate_initial_candidate(self):
        candidate = np.zeros(shape=self.number_variables * self.n)
        for i in range(0, self.number_variables * self.n):
            bit = np.round(np.random.uniform(0, 1))
            candidate[i] = bit
        return candidate

    def return_float_point(self, candidate):
        candidate_array = [
            candidate[i * int(self.n): (i + 1) * int(self.n)]
            for i in range(0, self.number_variables)
        ]
        candidate_np_array = np.array(candidate_array)
        # print(candidate_np_array, candidate_np_array.shape)
        float_points = np.zeros(self.number_variables)
        for index in range(0, self.number_variables):
            float_points[index] = self.binary_decoding(candidate_np_array[index])
        return float_points

    def generate_HammingNeighbours(self, candidate):
        candidate_neighbours = np.zeros(
            shape=(self.n * self.number_variables, self.n * self.number_variables)
        )
        for index in range(0, self.n * self.number_variables):
            candidate_neighbours[index] = switch_bit(candidate, index)
        return candidate_neighbours

    def evaluate(self, candidate):
        float_points = self.return_float_point(candidate)
        # print(float_points)
        result = np.inf
        if self.name == "Griewangk":
            sum = 0
            for float_point in float_points:
                sum += pow(float_point, 2)
            function_termen_1 = sum / 4000

            function_termen_2 = 1
            for index in range(0, self.number_variables):
                function_termen_2 *= cos(float_points[index] / sqrt(index + 1))

            result = function_termen_1 - function_termen_2 + 1
        elif self.name == "Rastrigin":
            termen1 = 10 * self.number_variables
            termen2 = 0
            for index in range(0, self.number_variables):
                big_sum_termen_1 = pow(float_points[index], 2)
                big_sum_termen_2 = 10 * cos(2 * pi * float_points[index])
                big_sum_final_termen = big_sum_termen_1 - big_sum_termen_2
                termen2 += big_sum_final_termen
            result = termen1 + termen2
        elif self.name == "Rosenbrock":
            result = 0
            for index in range(0, self.number_variables - 1):
                result += 100 * pow(float_points[index + 1] - pow(float_points[index], 2), 2) + pow(
                    1 - float_points[index], 2)
        return result

    def select_v_n_from_neighbours(
            self, val_v_c, neighbours, method="first_improvement"
    ):
        if method == "first_improvement":
            for index in range(0, neighbours.shape[0]):
                if self.evaluate(neighbours[index]) < val_v_c:
                    return neighbours[index]
            return None
        elif method == "best_improvement":
            val_neighbours = []
            for neighbour in neighbours:
                val_neighbours.append(self.evaluate(neighbour))
            index = np.argmin(val_neighbours)
            return neighbours[index]

    def HillClimbingAlgorithm(self):
        t = 0
        best = [1 for index in range(0, self.n * self.number_variables)]
        best_value = self.evaluate(best)
        while t < MAX:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )
            local = False
            v_c = self.generate_initial_candidate()
            val_v_c = self.evaluate(v_c)
            while not local:
                neighbours = self.generate_HammingNeighbours(v_c)
                # v_n = self.select_v_n_from_neighbours(
                #     val_v_c, neighbours, method="first_improvement"
                # )
                v_n = self.select_v_n_from_neighbours(val_v_c, neighbours, method="best_improvement")
                if v_n is not None:
                    val_v_n = self.evaluate(v_n)
                else:
                    val_v_n = val_v_c
                if val_v_n < val_v_c:
                    v_c = v_n
                    val_v_c = val_v_n
                    print(
                        "Am gasit o valoare mai buna: {0}".format(
                            self.evaluate(v_n)
                        )
                    )
                else:
                    local = True
            t += 1
            if val_v_c < best_value:
                best = v_c
                best_value = val_v_c
            print(
                "Iteratia {0}, cea mai buna valoare gasita: {1}".format(t, best_value)
            )
        print(
            "Cea mai buna valoare gasita dupa {0} rulari este: {1}.".format(
                MAX, best_value
            )
        )


if __name__ == "__main__":
    # h = HillClimbing("Rastrigin")
    h = HillClimbing("Rosenbrock")
    h.HillClimbingAlgorithm()
