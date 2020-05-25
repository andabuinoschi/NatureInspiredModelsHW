from MTSP.Node import Node
from math import sqrt, pow


def compute_distance(point_A: Node, point_B: Node):
    coords_A = point_A.get_coordinates()
    coords_B = point_B.get_coordinates()
    distance = sqrt(pow(coords_A[0] - coords_B[0], 2) + pow(coords_A[1] - coords_B[1], 2))
    return distance


class Edge:
    def __init__(self, start: Node = None, destination: Node = None):
        self.start = start
        self.destination = destination
        self.distance = compute_distance(start, destination)

    def get_distance(self):
        return self.distance

    def __str__(self):
        return "Node {0} - Node {1} with distance {2}".format(self.start, self.destination, self.distance)


if __name__ == "__main__":
    a = Node(1, (565.0, 575.0))
    b = Node(2, (25.0, 185.0))
    edge = Edge(a, b)
    print(edge.get_distance())
