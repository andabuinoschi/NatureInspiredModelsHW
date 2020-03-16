import numpy as np

from HW1.candidate import CandidateSelector
from HW1.params import FunctionFactory, FunctionParam

MAX = 50


class HillClimbing:
    def __init__(self, function):
        self.params: FunctionParam = function.function_param
        self.function = function
        self.candidate_selector = CandidateSelector(function.function_param)

    def binary_decoding(self, binary_point):
        decimal = 0
        for index in range(0, len(binary_point)):
            decimal += binary_point[index] * pow(2, len(binary_point) - index - 1)
        real = self.params.a1 + decimal * (self.params.b1 - self.params.a1) / (
            pow(2, len(binary_point)) - 1
        )
        return real

    def return_float_point(self, candidate):
        candidate_array = [
            candidate[i * int(self.params.n) : (i + 1) * int(self.params.n)]
            for i in range(0, self.params.number_variables)
        ]
        candidate_np_array = np.array(candidate_array)
        # print(candidate_np_array, candidate_np_array.shape)
        float_points = np.zeros(self.params.number_variables)
        for index in range(0, self.params.number_variables):
            float_points[index] = self.binary_decoding(candidate_np_array[index])
        return float_points

    def evaluate(self, candidate):
        float_points = self.return_float_point(candidate)
        return self.function.evaluate(float_points)

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

    def hillClimbingAlgorithm(self):
        t = 0
        best = [1 for index in range(0, self.params.n * self.params.number_variables)]
        best_value = self.evaluate(best)
        while t < MAX:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )
            local = False
            v_c = self.candidate_selector.generate_initial_candidate()
            val_v_c = self.evaluate(v_c)
            while not local:
                neighbours = self.candidate_selector.generate_HammingNeighbours(v_c)
                # v_n = self.select_v_n_from_neighbours(
                #     val_v_c, neighbours, method="first_improvement"
                # )
                v_n = self.select_v_n_from_neighbours(
                    val_v_c, neighbours, method="best_improvement"
                )
                if v_n is not None:
                    val_v_n = self.evaluate(v_n)
                else:
                    val_v_n = val_v_c
                if val_v_n < val_v_c:
                    v_c = v_n
                    val_v_c = val_v_n
                    print("Am gasit o valoare mai buna: {0}".format(self.evaluate(v_n)))
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
    h = HillClimbing(FunctionFactory.create("Griewangk"))
    h.hillClimbingAlgorithm()
