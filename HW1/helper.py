import numpy as np

from HW1.params import FunctionParam


class BinaryHelper:
    def __init__(self, function):
        self.params: FunctionParam = function.function_param
        self.function = function

    def binary_decoding(self, binary_point):
        decimal = 0
        for index in range(0, len(binary_point)):
            decimal += binary_point[index] * pow(2, len(binary_point) - index - 1)
        real = self.params.a1 + decimal * (self.params.b1 - self.params.a1) / (
                pow(2, len(binary_point)) - 1
        )
        return real

    def return_float_point(self, candidate):
        candidate_array = [
            candidate[i * int(self.params.n): (i + 1) * int(self.params.n)]
            for i in range(0, self.params.number_variables)
        ]
        candidate_np_array = np.array(candidate_array)
        # print(candidate_np_array, candidate_np_array.shape)
        float_points = np.zeros(self.params.number_variables)
        for index in range(0, self.params.number_variables):
            float_points[index] = self.binary_decoding(candidate_np_array[index])
        return float_points

    def evaluate(self, candidate):
        float_points = self.return_float_point(candidate)
        return self.function.evaluate(float_points)
