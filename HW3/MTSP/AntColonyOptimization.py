from MTSP.Graph import Graph
from MTSP.Ant import Ant


class AntColonyOptimizationMTSP:
    def __init__(self, tsp_file, number_of_ants: int, generations: int, alpha: float, beta: float, rho: float):
        self.graph = Graph(tsp_file=tsp_file)
        self.alpha = alpha  # 1.0
        self.beta = beta  # 10.0
        self.number_of_ants = number_of_ants
        self.generations = generations
        self.rho = rho  # evaporation rate

    def pheromone_update(self, ants_list: list):
        # each ant leaves pheromone along its tour
        for i in range(0, len(self.graph.pheromone_matrix)):
            for j in range(0, len(self.graph.pheromone_matrix)):
                # evaporate some pheromone
                self.graph.pheromone_matrix[i][j] = (1.0 - self.rho) * self.graph.pheromone_matrix[i][j]
                for ant in ants_list:
                    self.graph.pheromone_matrix[i][j] += ant.delta_pheromone[i][j]

    def AntColonyOptimizationAlg(self):
        best_cost = float('inf')
        best_solution = []
        for generation in range(0, self.generations):
            ants = [Ant(self) for i in range(0, self.number_of_ants)]
            for ant in ants:
                for i in range(0, self.graph.nodes_number):
                    ant.select_next_node()
                ant.total_cost += self.graph.distance_matrix[ant.tour[-1]][ant.tour[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.tour
                ant.update_delta_pheromone()
            if generation != 0:
                print("------------------------------------------------------------------")
                print("Generation {0}".format(generation))
                print("best cost {0} with solution {1}".format(best_cost, best_solution))
            self.pheromone_update(ants)
        return best_solution, best_cost


if __name__ == "__main__":
    aco = AntColonyOptimizationMTSP('Datasets/berlin52.tsp', 100, 100, 0.8, 2.5, 0.9)
    print(aco.AntColonyOptimizationAlg())
