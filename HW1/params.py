from math import sqrt, cos, pi

import numpy as np


class FunctionParam:
    def return_N_n(self):
        N = (self.b1 - self.a1) * pow(10, self.precision)
        n = np.ceil(np.log2(N))
        return int(N), int(n)

    def __init__(self, a1, b1, precision=4, number_variables=30):
        self.precision = precision
        self.number_variables = number_variables
        self.a1 = a1
        self.b1 = b1
        self.N, self.n = self.return_N_n()


class Griewangk:
    def __init__(self, function_param):
        self.function_param = function_param

    def evaluate(self, float_points):
        sum = 0
        for float_point in float_points:
            sum += pow(float_point, 2)
        function_termen_1 = sum / 4000

        function_termen_2 = 1
        for index in range(0, self.function_param.number_variables):
            function_termen_2 *= cos(float_points[index] / sqrt(index + 1))

        return function_termen_1 - function_termen_2 + 1


class Rastrigin:
    def __init__(self, function_param):
        self.function_param = function_param

    def evaluate(self, float_points):
        termen1 = 10 * self.function_param.number_variables
        termen2 = 0
        for index in range(0, self.function_param.number_variables):
            big_sum_termen_1 = pow(float_points[index], 2)
            big_sum_termen_2 = 10 * cos(2 * pi * float_points[index])
            big_sum_final_termen = big_sum_termen_1 - big_sum_termen_2
            termen2 += big_sum_final_termen
        return termen1 + termen2


class Rosenbrock:
    def __init__(self, function_param):
        self.function_param = function_param

    def evaluate(self, float_points):
        result = 0
        for index in range(0, self.function_param.number_variables - 1):
            result += 100 * pow(
                float_points[index + 1] - pow(float_points[index], 2), 2
            ) + pow(1 - float_points[index], 2)
        return result


class TPrim:
    def __init__(self, function_param):
        self.function_param = function_param

    def evaluate(self, float_point):
        x = float_point[0]
        return pow(x, 3) - 60 * pow(x, 2) + 900 * x + 100


class FunctionFactory:
    @staticmethod
    def create(function_type):
        if function_type == "Griewangk":
            return Griewangk(FunctionParam(-600, 600))
        elif function_type == "Rastrigin":
            return Rastrigin(FunctionParam(-5.12, 5.12))
        elif function_type == "Rosenbrock":
            return Rosenbrock(FunctionParam(-2.048, 2.048))
        elif function_type == "tprim":
            return TPrim(FunctionParam(0, 31, precision=0, number_variables=1))
