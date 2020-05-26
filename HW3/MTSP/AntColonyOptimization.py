from MTSP.Graph import Graph
from MTSP.Ant import Ant
from sklearn.cluster import KMeans
import numpy as np


class AntColonyOptimizationMTSP:
    def __init__(self, tsp_file, number_of_salesman: int, number_of_ants: int, generations: int, alpha: float, beta: float,
                 rho: float):
        self.graph = Graph(tsp_file=tsp_file)
        self.alpha = alpha  # for pheromone
        self.beta = beta  # for heuristic
        self.number_of_ants = number_of_ants
        self.generations = generations
        self.rho = rho  # evaporation rate
        self.number_of_salesmen = number_of_salesman
        self.total_nodes_labels = [node.label for node in self.graph.nodes if node.label != 0]
        self.total_nodes_coordinates = [node.coordinates for node in self.graph.nodes if node.label != 0]
        self.nodes_kmeans = KMeans(n_clusters=self.number_of_salesmen).fit(self.total_nodes_coordinates)
        self.nodes_to_visit_salesmen = [[0] for i in range(0, self.number_of_salesmen)]
        for index, label in enumerate(self.nodes_kmeans.labels_):
            self.nodes_to_visit_salesmen[label].append(self.total_nodes_labels[index])

    def pheromone_update(self, ants_list: list):
        # each ant leaves pheromone along its tour
        for i in range(0, len(self.graph.pheromone_matrix)):
            for j in range(0, len(self.graph.pheromone_matrix)):
                # evaporate some pheromone
                self.graph.pheromone_matrix[i][j] = (1.0 - self.rho) * self.graph.pheromone_matrix[i][j]
                for ant in ants_list:
                    self.graph.pheromone_matrix[i][j] += ant.delta_pheromone[i][j]

    def AntColonyOptimizationAlg(self):
        total_cost_salesmen = 0
        best_cost_salesmen = []
        best_solution_salesmen = []
        for salesman in range(0, self.number_of_salesmen):
            best_cost = float('inf')
            best_solution = []
            for generation in range(0, self.generations):
                ants = [Ant(self, salesman) for i in range(0, self.number_of_ants)]
                for ant in ants:
                    for i in range(0, len(self.nodes_to_visit_salesmen[salesman])):
                        ant.select_next_node()
                    ant.total_cost += self.graph.distance_matrix[ant.tour[-1]][ant.tour[0]]
                    if ant.total_cost < best_cost:
                        best_cost = ant.total_cost
                        best_solution = ant.tour
                    ant.update_delta_pheromone()
                self.pheromone_update(ants)
            total_cost_salesmen += best_cost
            best_cost_salesmen.append(best_cost)
            best_solution_salesmen.append(best_solution)
            print("Best solution for salesman {0} is with cost {1} and tour {2}.".format(salesman + 1, best_cost, best_solution))
        return total_cost_salesmen, best_cost_salesmen, best_solution_salesmen


if __name__ == "__main__":
    aco = AntColonyOptimizationMTSP('Datasets/eil51.tsp', 5, 100, 100, 0.8, 2.5, 0.5)
    print(aco.AntColonyOptimizationAlg())
