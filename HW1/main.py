import os
import sys

sys.path.append(os.path.abspath('../'))

from HW1.genetic_algorithm import GeneticAlgorithm
from HW1.hill_climbing import HillClimbing
from HW1.params import FunctionFactory
import numpy as np
import json


def hc_experiment(function):
    h = HillClimbing(FunctionFactory.create(function))
    local_iteration = h.hillClimbingAlgorithm(hybrid=False)
    print(f"average - {np.average(local_iteration)}", "min", str(min(local_iteration)), "max", str(max(local_iteration)
                                                                                                   ))


def ga_experiment(function):
    acc = []
    f = FunctionFactory.create(function)
    ga = GeneticAlgorithm(f, number_of_iterations=2000)
    for i in range(30):
        _, best_value = ga.run()
        acc.append(best_value)

    best_values_string = json.dumps(acc)
    json.dump(best_values_string, open(f"../raport/values/{f.__class__.__name__}best_values_ga.json", "w"))


def hybrid_experiment(function, iterations=750):
    f = FunctionFactory.create(function)

    ga = GeneticAlgorithm(f, iterations)
    h = HillClimbing(f, ga)

    local_iteration = h.hillClimbingAlgorithm(hybrid=True)
    print(f"average - {np.average(local_iteration)}", "min", str(min(local_iteration)), "max",
          str(max(local_iteration)))


def t_prim():
    h = HillClimbing(FunctionFactory.create("tprim"))
    h.hillClimbingAlgorithm()


if __name__ == "__main__":
    # hc_experiment("Rosenbrock")
    # ga_experiment("Rosenbrock")
    hybrid_experiment("Rosenbrock", iterations=2000)
