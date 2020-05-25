
class Node:
    def __init__(self, label=0, coordinates=(0, 0)):
        self.label = label
        self.coordinates = coordinates

    def get_coordinates(self):
        return self.coordinates

    def __str__(self):
        return "LABEL {0} COORDINATES ({1}, {2})".format(self.label, self.coordinates[0], self.coordinates[1])
