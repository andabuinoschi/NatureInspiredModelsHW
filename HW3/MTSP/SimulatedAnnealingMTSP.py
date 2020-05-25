from MTSP.Graph import Graph
import math
import random


class SimulatedAnnealingMTSP:
    def __init__(self, temperature, alpha, tsp_file, stopping_iteration, stopping_temperature,
                 number_travel_salesmen=None):
        self.number_salesman = number_travel_salesmen
        self.t = 0
        self.T = temperature
        self.alpha = alpha
        self.stop_iter = stopping_iteration
        self.stop_temp = stopping_temperature

        graph = Graph(tsp_file)
        self.distance_matrix = graph.get_distance_matrix()
        self.v_c = self.init_first_solution(self.distance_matrix)
        self.best_solution = self.v_c

        self.solution_history = [self.v_c]
        self.current_value = self.compute_solution_quality(self.v_c)
        self.best_value = self.current_value

    def init_first_solution(self, distance_matrix):
        node = 0
        track = [node]
        all_visiting_nodes = [i for i in range(0, distance_matrix.shape[0])]
        all_visiting_nodes.remove(node)
        while all_visiting_nodes:
            # initialization with minimum, driving to local optima too quick
            # nearest_node = min([(distance_matrix[node][j], j) for j in all_visiting_nodes], key=lambda x: x[0])
            # node = nearest_node[1]
            # track.append(node)
            # all_visiting_nodes.remove(node)

            # random initialization
            index = random.randint(0, len(all_visiting_nodes) - 1)
            node = all_visiting_nodes[index]
            track.append(node)
            all_visiting_nodes.remove(node)
        return track

    def compute_solution_quality(self, solution):
        return sum([self.distance_matrix[i, j] for i, j in zip(solution, solution[1:] + [solution[0]])])

    def twoExchangeNeighbourhood(self, candidate, i, k):
        neighbour = candidate[0: i]
        neighbour += reversed(candidate[i: k + 1])
        neighbour += candidate[k + 1:]
        return neighbour

    def acceptance_probability(self, candidate_solution_quality):
        return math.exp(-abs(candidate_solution_quality - self.current_value) / self.T)

    def accept(self, candidate):
        candidate_quality = self.compute_solution_quality(candidate)
        if candidate_quality < self.current_value:
            self.current_value = candidate_quality
            self.v_c = candidate
        else:
            if random.random() <= self.acceptance_probability(candidate_quality):
                self.current_value = candidate_quality
                self.v_c = candidate

    def SimulatedAnnealingAlg(self):
        while self.T >= self.stop_temp and self.t < self.stop_iter:
            candidate = self.v_c
            i = random.randint(1, self.distance_matrix.shape[0] - 2)
            k = random.randint(i + 1, self.distance_matrix.shape[0] - 1)
            v_n = self.twoExchangeNeighbourhood(candidate, i, k)
            self.accept(v_n)
            if self.current_value < self.best_value:
                self.best_value = self.current_value
                self.best_solution = self.v_c
            self.t += 1
            self.T *= self.alpha
            print(self.best_value)
            print(self.best_solution)


if __name__ == "__main__":
    sa = SimulatedAnnealingMTSP(1200, alpha=0.9995, tsp_file='Datasets/berlin52.tsp', stopping_iteration=1000000,
                                stopping_temperature=math.pow(10, -5), number_travel_salesmen=2)
    sa.SimulatedAnnealingAlg()
