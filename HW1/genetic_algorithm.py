import random

import matplotlib.pyplot as plt
import numpy as np

from HW1.candidate import CandidateSelector
from HW1.helper import BinaryHelper

POP_SIZE = 100
FIG_ID = 0
MIN_EVAL_VALUES = []


def select(q, pick):
    for i in range(0, len(q) - 1):
        if q[i] <= pick < q[i + 1]:
            return i


def roulette_selection(population, fitness):
    _max = sum(fitness)
    new_pop = []
    p = []
    q = [0]

    for i in range(POP_SIZE):
        p.append(fitness[i] / _max)

    for i in range(1, POP_SIZE + 1):
        q.append(q[i - 1] + p[i - 1])
    q.append(1.1)
    # plt.plot(q)
    # plt.show()

    for i in range(POP_SIZE):
        pos = random.uniform(0, 1)
        new_pop.append(population[select(q, pos)])
    return new_pop


def rank_selection(population, fitness):
    pop_fit = zip(population, fitness)
    sorted_pop_fit = sorted(pop_fit, key=lambda t: t[1])

    pop, fit = zip(*sorted_pop_fit)
    return roulette_selection(pop, fit)


class GeneticAlgorithm:
    def __init__(self, function, number_of_iterations=300):
        self.binary_helper = BinaryHelper(function)
        self.candidate_selector = CandidateSelector(function.function_param)
        self.mutation_probability = .01
        self.crossover_probability = .1
        self.chromosome_mutation_probability = .1
        self.number_of_iterations = number_of_iterations
        self.function = function

    def run(self):
        global FIG_ID
        t = 0
        current_population = [
            self.candidate_selector.generate_initial_candidate() for _ in range(POP_SIZE)
        ]

        max_fitness_values = []
        min_eval_values = []

        while t < self.number_of_iterations:
            print(
                "---------------------------- ITERATIA {0} ----------------------------".format(
                    t
                )
            )

            min_f = 1e9
            max_f = -1e9

            eval = np.array([_ for _ in range(POP_SIZE)], dtype=np.float)
            for i in range(POP_SIZE):
                f = self.binary_helper.evaluate(current_population[i])
                eval[i] = f
                if f < min_f:
                    min_f = f
                if f > max_f:
                    max_f = f

            fitness = 1.1 * max_f - eval

            max_fitness_values.append(max(fitness))
            min_eval_values.append(min(eval))
            print(min(eval))

            if t == self.number_of_iterations - 1:
                plt.plot(max_fitness_values)
                plt.xlabel("Iteration")
                plt.ylabel("Fitness value")
                plt.savefig(f'../raport/eps/{self.function.__class__.__name__}_ga_max_fitness_{FIG_ID}.eps', format='eps')
                plt.close()

                plt.plot(min_eval_values)
                plt.xlabel("Iteration")
                plt.ylabel("Minimum function value")
                plt.savefig(f'../raport/eps/{self.function.__class__.__name__}_ga_min_eval_{FIG_ID}.eps', format='eps')
                plt.close()
                FIG_ID += 1

                _argmin = np.argmin(eval)
                MIN_EVAL_VALUES.append(float(min(eval)))
                ind = 0
                if isinstance(_argmin, np.ndarray):
                    ind = _argmin[0]
                elif isinstance(_argmin, int):
                    ind = _argmin
                return current_population[ind], min(eval)

            # selection
            parents = roulette_selection(current_population, fitness)

            # cross over
            offsprings = self.candidate_selector.cross_over(parents, (POP_SIZE, len(parents[0])),
                                                            self.crossover_probability)

            # mutation
            offsprings = self.candidate_selector.mutation(offsprings, self.mutation_probability,
                                                          self.chromosome_mutation_probability)

            current_population = offsprings

            t += 1
