import numpy as np

a = np.random.randint(5, 10)
print(a)


class HillClimbing:
    def __init__(self):
        self.t = 0
        self.v_c = 0
        self.best = self.v_c
        self.a = -5.0
        self.b = 1.0
        self.precision = 5
        N = (self.b - self.a) * np.power(10, 5)
        n = np.ceil(np.log2(N))

    def test_function(self):
        pass  # aici initializam self.a, self.b si valoarea functiei care trebuie returnata

    def binary_decoding(self, binary_point):
        decimal = 0
        for index in range(0, len(binary_point)):
            decimal += binary_point[index] * np.power(2, len(binary_point) - index - 1)
        real = self.a + decimal * (self.b - self.a) / (
            np.power(2, len(binary_point)) - 1
        )
        return real

    def search_neighbourhood(self):
        v_c = self.v_c


if __name__ == "__main__":
    print(HillClimbing().binary_decoding([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
