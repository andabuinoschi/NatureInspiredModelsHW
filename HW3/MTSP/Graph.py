from MTSP.Node import Node
from MTSP.Edge import Edge
import numpy as np


class Graph:
    def __init__(self, tsp_file: str = None):
        self.nodes = []
        self.edges = []
        self.edges_dict = {}
        if tsp_file:
            ignore_lines = True
            file_handle = open(tsp_file, 'r')
            for line in file_handle:
                line = line.strip("\n")
                line = line.lstrip(" ")
                if line == "NODE_COORD_SECTION":
                    ignore_lines = False
                    continue
                elif line == "EOF":
                    ignore_lines = True
                if ignore_lines:
                    pass
                else:
                    coords_splited = line.split(" ")
                    # if "" in coords_splited:
                    #     coords_splited.remove("")
                    coords_splited = [element for element in coords_splited if element != ""]
                    label = coords_splited[0]
                    coord_x = float(coords_splited[1])
                    coord_y = float(coords_splited[2])
                    node = Node(label=int(label) - 1, coordinates=(float(coord_x), float(coord_y)))
                    self.nodes.append(node)
        self.nodes_number = len(self.nodes)
        for i in range(0, self.nodes_number):
            for j in range(0, self.nodes_number):
                if i != j:
                    edge = Edge(self.nodes[i], self.nodes[j])
                    self.edges.append(edge)
                    self.edges_dict[(self.nodes[i], self.nodes[j])] = edge.get_distance()
        self.nodes = np.array(self.nodes)
        self.distance_matrix = np.zeros(shape=(len(self.nodes), len(self.nodes)))
        self.construct_distance_matrix()
        # self.pheromone_matrix = [[1 / (self.nodes_number * self.nodes_number) for i in range(0, self.nodes_number)] for j in range(0, self.nodes_number)]
        self.pheromone_matrix = np.random.uniform(size=(self.nodes_number, self.nodes_number))

    def construct_distance_matrix(self):
        for line in range(0, len(self.nodes)):
            for column in range(0, len(self.nodes)):
                self.distance_matrix[line][column] = Edge(self.nodes[line], self.nodes[column]).get_distance()

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges_dict

    def __str__(self):
        representation_string = "NODES \n"
        for node in self.nodes:
            representation_string += str(node) + "\n"
        representation_string += "\n"
        representation_string += "EDGES \n"
        for edge in self.edges:
            representation_string += str(edge) + "\n"
        return representation_string


if __name__ == "__main__":
    graph = Graph('Datasets/berlin52.tsp')
    print(graph)
