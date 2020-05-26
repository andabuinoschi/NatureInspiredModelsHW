from MTSP.Graph import Graph
import math
import random
from sklearn.cluster import KMeans


class SimulatedAnnealingMTSP:
    def __init__(self, temperature, alpha, tsp_file, stopping_iteration, stopping_temperature,
                 number_travel_salesmen=None):
        self.number_salesman = number_travel_salesmen
        self.t = 0
        self.T = temperature
        self.alpha = alpha
        self.stop_iter = stopping_iteration
        self.stop_temp = stopping_temperature

        self.graph = Graph(tsp_file)
        self.distance_matrix = self.graph.get_distance_matrix()
        self.v_c = self.init_first_solution(self.distance_matrix)
        self.best_solution = self.v_c

        self.solution_history = [self.v_c]
        self.current_value = sum(self.compute_solution_quality(self.v_c))
        self.best_value = self.current_value
        self.min_max_qualities = None

    def init_first_solution(self, distance_matrix):
        start_node = 0
        track = [[start_node] for _ in range(self.number_salesman)]

        nodes_to_visit = [node.coordinates for node in self.graph.nodes if node.label != start_node]

        kmeans = KMeans(n_clusters=self.number_salesman)
        kmeans.fit(nodes_to_visit)

        node_clusters = [[] for _ in range(self.number_salesman)]

        for node in self.graph.nodes:
            if node.label != start_node:
                cluster = kmeans.predict([node.coordinates])[0]
                node_clusters[cluster].append(node)

        for c in range(self.number_salesman):
            current_cluster = node_clusters[c]
            while current_cluster:
                index = random.randint(0, len(current_cluster) - 1)
                node = current_cluster[index]
                track[c].append(node.label)
                current_cluster.remove(node)
        return track

    def compute_solution_quality(self, solution):
        sums = []
        for sol in solution:
            sums.append(sum([self.distance_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])]))
        return sums

    def twoExchangeNeighbourhood(self, candidate, i, k):
        neighbour = candidate[0: i]
        neighbour += reversed(candidate[i: k + 1])
        neighbour += candidate[k + 1:]
        return neighbour

    def acceptance_probability(self, candidate_solution_quality):
        return math.exp(-abs(candidate_solution_quality - self.current_value) / self.T)

    def accept(self, candidate):
        candidate_quality_per_salesman = self.compute_solution_quality(candidate)

        candidate_quality = sum(candidate_quality_per_salesman)
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

            v_n = []

            for candidate_salesman in candidate:
                i = random.randint(1, len(candidate_salesman) - 2)
                k = random.randint(i + 1, len(candidate_salesman) - 1)
                v_n_salesman = self.twoExchangeNeighbourhood(candidate_salesman, i, k)
                v_n.append(v_n_salesman)
            self.accept(v_n)
            if self.current_value < self.best_value:
                self.best_value = self.current_value
                self.best_solution = self.v_c

                best_values = self.compute_solution_quality(self.best_solution)
                self.min_max_qualities = [min(best_values), max(best_values)]
            self.t += 1
            self.T *= self.alpha
            if self.min_max_qualities is not None:
                print("[%.3f, %.3f] & %.3f" % (self.min_max_qualities[0], self.min_max_qualities[1], self.best_value))
            print(self.best_solution)


if __name__ == "__main__":
    sa = SimulatedAnnealingMTSP(1200, alpha=0.9995, tsp_file='Datasets/eil76.tsp', stopping_iteration=1000000,
                                stopping_temperature=math.pow(10, -5), number_travel_salesmen=2)
    sa.SimulatedAnnealingAlg()
