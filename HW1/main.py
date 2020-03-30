from HW1.hill_climbing import HillClimbing
from HW1.params import FunctionFactory

if __name__ == "__main__":
    h = HillClimbing(FunctionFactory.create("Rastrigin"))
    h.hillClimbingAlgorithm(hybrid=True)

    # h = HillClimbing(FunctionFactory.create("tprim"))
    # h.hillClimbingAlgorithm()
