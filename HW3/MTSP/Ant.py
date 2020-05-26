# from MTSP.AntColonyOptimization import AntColonyOptimizationMTSP
import numpy as np
import math
import random
import copy


class Ant:
    def __init__(self, colony, salesman):
        self.colony = colony
        self.graph = colony.graph
        self.total_cost = 0
        self.tour = []
        self.delta_pheromone = [[0 for i in range(0, self.graph.nodes_number)] for j in
                                range(0, self.graph.nodes_number)]
        self.nodes_to_visit = copy.copy(self.colony.nodes_to_visit_salesmen[salesman])
        self.heuristic_information = np.zeros(shape=(self.graph.nodes_number, self.graph.nodes_number))
        for i in range(0, self.graph.nodes_number):
            for j in range(0, self.graph.nodes_number):
                if i == j:
                    self.heuristic_information[i][j] = math.pow(10, -2)
                else:
                    self.heuristic_information[i][j] = 1 / self.graph.distance_matrix[i][j]
        node_start = 0
        self.tour.append(node_start)
        self.current_node = node_start
        self.nodes_to_visit.remove(node_start)

    def select_next_node(self):
        denominator = 0
        for i in self.nodes_to_visit:
            denominator += math.pow(self.graph.pheromone_matrix[self.current_node][i], self.colony.alpha) * \
                           math.pow(self.heuristic_information[self.current_node][i], self.colony.beta)
        probabilities_to_next_node = np.zeros(shape=self.graph.nodes_number)
        for index in range(0, self.graph.nodes_number):
            if index in self.nodes_to_visit:
                    probabilities_to_next_node[index] = math.pow(self.graph.pheromone_matrix[self.current_node][index],
                                                                 self.colony.alpha) * \
                                                        math.pow(self.heuristic_information[self.current_node][index],
                                                                 self.colony.beta) / denominator
        next_node = -1
        node_chosen = False
        while not node_chosen:
            random_probability = random.random()
            for node, probability in enumerate(probabilities_to_next_node):
                if random_probability <= probability:
                    next_node = node
                    node_chosen = True
                    self.tour.append(next_node)
                    self.nodes_to_visit.remove(next_node)
                    break
            if not self.nodes_to_visit:
                next_node = 0
                break
        self.total_cost += self.graph.distance_matrix[self.current_node][next_node]
        self.current_node = next_node

    def update_delta_pheromone(self):
        delta_pheromone = 1 / self.total_cost
        for i in range(0, len(self.tour) - 1):
            node_j = self.tour[i]
            node_l = self.tour[i + 1]
            self.delta_pheromone[node_j][node_l] = delta_pheromone
            self.delta_pheromone[node_l][node_j] = self.delta_pheromone[node_j][node_l]
