from HW1.candidate import CandidateSelector
import numpy as np
import random

from HW1.helper import BinaryHelper

MAX = 500


class GeneticAlgorithm:
    def __init__(self, function):
        self.binary_helper = BinaryHelper(function)
        self.candidate_selector = CandidateSelector(function.function_param)

    def run(self):
        t = 0
        current_population = [
            self.candidate_selector.generate_initial_candidate() for _ in range(10)
        ]

        while t < MAX:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )

            min_f = 1e9
            max_f = -1e9

            eval = np.array([_ for _ in range(len(current_population))])
            for i in range(len(current_population)):
                f = self.binary_helper.evaluate(current_population[i])
                eval[i] = f
                if f < min_f:
                    min_f = f
                if f > max_f:
                    max_f = f

            fitness = 1.1 * max_f - eval

            print(max(eval))

            # selection
            population = self.selection(current_population, fitness)

            # cross over
            self.candidate_selector.cross_over(population)

            # mutation
            self.candidate_selector.mutation(population)

            current_population = population

            t += 1

        return current_population[np.argmax(eval)]

    def selection(self, population, fitness):
        fs = [_ for _ in range(len(population))]
        new_pop = []

        fs[0] = fitness[0]

        for i in range(1, len(population)):
            fs[i] = fs[i - 1] + fitness[i]

        pos = random.random() * fs[-1]
        for i in range(len(population)):
            new_pop.append(population[self.select(fs, pos)])
        return new_pop

    def select(self, fs, pos):
        for i in range(len(fs)):
            if pos < fs[i]:
                return i
