import numpy as np

from HW1.candidate import CandidateSelector
from HW1.genetic_algorithm import GeneticAlgorithm
from HW1.helper import BinaryHelper
from HW1.params import FunctionParam

MAX = 3000


class HillClimbing:
    def __init__(self, function):
        self.params: FunctionParam = function.function_param
        self.function = function
        self.candidate_selector = CandidateSelector(function.function_param)
        self.binary_helper = BinaryHelper(function)

    def select_v_n_from_neighbours(
            self, val_v_c, neighbours, method="first_improvement"
    ):
        if method == "first_improvement":
            for index in range(0, neighbours.shape[0]):
                if self.binary_helper.evaluate(neighbours[index]) < val_v_c:
                    return neighbours[index]
            return None
        elif method == "best_improvement":
            val_neighbours = []
            for neighbour in neighbours:
                val_neighbours.append(self.binary_helper.evaluate(neighbour))
            index = np.argmin(val_neighbours)
            return neighbours[index]

    def hillClimbingAlgorithm(self, hybrid=False):
        t = 0
        best = [1 for _ in range(0, self.params.n * self.params.number_variables)]
        best_value = self.binary_helper.evaluate(best)
        while t < MAX:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )
            local = False
            if hybrid:
                ga = GeneticAlgorithm(self.function)
                v_c = ga.run()
            else:
                v_c = self.candidate_selector.generate_initial_candidate()
            val_v_c = self.binary_helper.evaluate(v_c)
            while not local:
                neighbours = self.candidate_selector.generate_HammingNeighbours(v_c)
                # v_n = self.select_v_n_from_neighbours(
                #     val_v_c, neighbours, method="first_improvement"
                # )
                v_n = self.select_v_n_from_neighbours(
                    val_v_c, neighbours, method="best_improvement"
                )
                if v_n is not None:
                    val_v_n = self.binary_helper.evaluate(v_n)
                else:
                    val_v_n = val_v_c
                if val_v_n < val_v_c:
                    v_c = v_n
                    val_v_c = val_v_n
                    print("Am gasit o valoare mai buna: {0}".format(self.binary_helper.evaluate(v_n)))
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
