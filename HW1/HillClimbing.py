import numpy as np
from math import sqrt, cos, pi

MAX = 50


def switch_bit(bitarray, position):
    temp = bitarray.tolist()
    temp[position] = 1 ^ int(temp[position])
    return np.array(temp)


class HillClimbing:
    def return_N_n(self):
        if self.name != 'SixHumpCamelBack':
            N = (self.b1 - self.a1) * pow(10, self.precision)
            n = np.ceil(np.log2(N))
            return int(N), int(n)
        else:
            N1 = (self.b1 - self.a1) * pow(10, self.precision)
            n1 = np.ceil(np.log2(N1))
            N2 = (self.b2 - self.a2) * pow(10, self.precision)
            n2 = np.ceil(np.log2(N2))
            return (int(N1), int(N2)), (int(n1), int(n2))

    def __init__(self, function_type="Griewangk", initial_candidate_solution = None):
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
        elif function_type == "SixHumpCamelBack":
            self.name = "SixHumpCamelBack"
            # pentru prima variabila x_1
            self.a1 = -3
            self.b1 = 3
            # pentru a doua variabila x_2
            self.a2 = -2
            self.b2 = 2
            self.number_variables = 2
        self.N, self.n = self.return_N_n()
        if self.name != "SixHumpCamelBack":
            self.candidate_solution_shape = self.number_variables * self.n
        else:
            self.candidate_solution_shape = self.n[0] + self.n[1]
        self.initial_candidate_solution = initial_candidate_solution

    def binary_decoding(self, binary_point, index):
        decimal = 0
        for idx in range(0, len(binary_point)):
            decimal += binary_point[idx] * pow(2, len(binary_point) - idx - 1)
        if self.name != 'SixHumpCamelBack' or (self.name == 'SixHumpCamelBack' and index == 0):
            real = self.a1 + decimal * (self.b1 - self.a1) / (pow(2, len(binary_point)) - 1)
        else:
            real = self.a2 + decimal * (self.b2 - self.a2) / (pow(2, len(binary_point)) - 1)
        return real

    def generate_initial_candidate(self):
        candidate = np.zeros(shape=self.candidate_solution_shape)
        for i in range(0, self.candidate_solution_shape):
            bit = np.round(np.random.uniform(0, 1))
            candidate[i] = bit
        return candidate

    def return_float_point(self, candidate):
        if self.name != 'SixHumpCamelBack':
            candidate_array = [
                candidate[i * int(self.n): (i + 1) * int(self.n)]
                for i in range(0, self.number_variables)
            ]
        else:
            candidate_array = [candidate[0:self.n[0]], candidate[self.n[0]: self.n[0] + self.n[1]]]
        candidate_np_array = np.array(candidate_array)
        # print(candidate_np_array, candidate_np_array.shape)
        float_points = np.zeros(self.number_variables)
        for index in range(0, self.number_variables):
            float_points[index] = self.binary_decoding(candidate_np_array[index], index)
        return float_points

    def generate_HammingNeighbours(self, candidate):
        candidate_neighbours = np.zeros(
            shape=(self.candidate_solution_shape, self.candidate_solution_shape)
        )
        for index in range(0, self.candidate_solution_shape):
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
        elif self.name == "SixHumpCamelBack":
            termen1 = (4 - 2.1 * pow(float_points[0], 2) + pow(float_points[0], 4)/3) * pow(float_points[0], 2)
            termen2 = float_points[0] * float_points[1]
            termen3 = (-4 + 4 * pow(float_points[1], 2)) * pow(float_points[1], 2)
            result = termen1 + termen2 + termen3
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
        best = [1 for index in range(0, self.candidate_solution_shape)]
        best_value = self.evaluate(best)
        while t < MAX:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )
            local = False
            if self.initial_candidate_solution: # util pentru pasul hibridizarii
                v_c = self.initial_candidate_solution
            else:
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
            "Cea mai buna valoare gasita dupa {0} rulari este: {1} in punctul de inflexiune {2}.".format(
                MAX, best_value, self.return_float_point(best)
            )
        )


if __name__ == "__main__":
    # h = HillClimbing("Griewangk")
    # h = HillClimbing("Rastrigin")
    # h = HillClimbing("Rosenbrock")
    h = HillClimbing("SixHumpCamelBack")
    h.HillClimbingAlgorithm()
