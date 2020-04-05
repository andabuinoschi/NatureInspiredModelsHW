import random

import numpy as np
import matplotlib.pyplot as plt


from HW1.candidate import CandidateSelector
from HW1.helper import BinaryHelper

MAX = 1000
POP_SIZE = 20
NUM_PARENTS = POP_SIZE//2


def select(q, pick):
    for i in range(len(q) - 1):
        if q[i] <= pick < q[i + 1]:
            # print(i)
            return i


def roulette_selection(population, fitness):
    _max = sum(fitness)
    new_pop = []
    p = []
    q = [0]

    for i in range(len(population)):
        p.append(fitness[i] / _max)

    for i in range(1, len(population) + 1):
        q.append(q[i - 1] + p[i - 1])
    q.append(1.1)

    for i in range(NUM_PARENTS):
        pos = random.uniform(0, 1)
        new_pop.append(population[select(q, pos)])
    return new_pop


def rank_selection(population, fitness):
    pop_fit = zip(population, fitness)
    sorted_pop_fit = sorted(pop_fit, key=lambda t: t[1])

    pop, fit = zip(*sorted_pop_fit)
    return roulette_selection(pop, fit)


class GeneticAlgorithm:
    def __init__(self, function, population_size=10):
        self.binary_helper = BinaryHelper(function)
        self.candidate_selector = CandidateSelector(function.function_param)
        self.mutation_probability = .01
        self.population_size = population_size
        self.num_parents = population_size / 2

    def run(self):
        t = 0
        current_population = [
            self.candidate_selector.generate_initial_candidate() for _ in range(self.population_size)
        ]

        max_fitness_values = []

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

            max_fitness_values.append(max(fitness))
            print(max(fitness))

            if t == MAX - 1:
                plt.plot(max_fitness_values)
                plt.show()
                _argmax = np.argmax(fitness)
                ind = 0
                if isinstance(_argmax, np.ndarray):
                    ind = _argmax[0]
                elif isinstance(_argmax, int):
                    ind = _argmax
                return current_population[ind]

            # selection
            parents = np.array(roulette_selection(current_population, fitness))

            # cross over
            offsprings = self.candidate_selector.cross_over(parents, (POP_SIZE - NUM_PARENTS, parents.shape[1]))

            # mutation
            self.candidate_selector.mutation(offsprings, self.mutation_probability)

            population = np.concatenate((parents, offsprings), axis=0)

            current_population = population

            t += 1
