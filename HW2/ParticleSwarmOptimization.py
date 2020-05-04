import random

import numpy as np

from math import sqrt, cos, pi


class ParticleSwarmOptimization:
    def __init__(self, function_type="Griewangk"):
        self.maximum_lookback_steps = 50
        self.precision = 5
        self.number_particles = 60
        self.inertia_factor_w1 = 0.7
        self.particle_self_confidence_factor_w2 = 2
        self.swarm_confidence_factor_w3 = 2
        if function_type == "Griewangk":
            self.name = "Griewangk"
            self.number_variables = 30
            self.a1 = -600
            self.b1 = 600
        elif function_type == "Rastrigin":
            self.name = "Rastrigin"
            self.number_variables = 30
            self.a1 = -5.12
            self.b1 = 5.12
        elif function_type == "Rosenbrock":
            self.name = "Rosenbrock"
            self.number_variables = 30
            self.a1 = -2.048
            self.b1 = 2.048
        elif function_type == "SixHumpCamelBack":
            self.name = "SixHumpCamelBack"
            # pentru prima variabila x_1
            self.a1 = -3
            self.b1 = 3
            # pentru a doua variabila x_2
            self.a2 = -2
            self.b2 = 2
            self.number_variables = 2
        self.particles_best_position_p = np.zeros(shape=(self.number_particles, self.number_variables))
        self.particles_positions_x = np.zeros(shape=(self.number_particles, self.number_variables))
        self.particles_velocities_v = np.zeros(shape=(self.number_particles, self.number_variables))
        self.generate_initial_particles()
        self.best_particle_g = self.particles_positions_x[random.randint(0, self.number_particles)]

    def generate_initial_particles(self):
        if self.name != "SixHumpCamelBack":
            for index_particle in range(0, self.number_particles):
                for index_variable in range(0, self.number_variables):
                    self.particles_positions_x[index_particle][index_variable] = random.uniform(self.a1, self.b1)
                    self.particles_velocities_v[index_particle][index_variable] = random.uniform(
                        - abs(self.b1 - self.a1), abs(self.b1 - self.a1))
        elif self.name == "SixHumpCamelBack":
            for index_particle in range(0, self.number_particles):
                self.particles_positions_x[index_particle][0] = random.uniform(self.a1, self.b1)
                self.particles_velocities_v[index_particle][0] = random.uniform(self.a1, self.b1)
                self.particles_positions_x[index_particle][1] = random.uniform(self.a2, self.b2)
                self.particles_velocities_v[index_particle][1] = random.uniform(self.a2, self.b2)
        self.particles_best_position_p = self.particles_positions_x

    def evaluate(self, particle):
        result = np.inf
        if self.name == "Griewangk":
            sum = 0
            for float_point in particle:
                sum += pow(float_point, 2)
            function_termen_1 = sum / 4000

            function_termen_2 = 1
            for index in range(0, self.number_variables):
                function_termen_2 *= cos(particle[index] / sqrt(index + 1))

            result = function_termen_1 - function_termen_2 + 1
        elif self.name == "Rastrigin":
            termen1 = 10 * self.number_variables
            termen2 = 0
            for index in range(0, self.number_variables):
                big_sum_termen_1 = pow(particle[index], 2)
                big_sum_termen_2 = 10 * cos(2 * pi * particle[index])
                big_sum_final_termen = big_sum_termen_1 - big_sum_termen_2
                termen2 += big_sum_final_termen
            result = termen1 + termen2
        elif self.name == "Rosenbrock":
            result = 0
            for index in range(0, self.number_variables - 1):
                result += 100 * pow(particle[index + 1] - pow(particle[index], 2), 2) + pow(
                    1 - particle[index], 2)
        elif self.name == "SixHumpCamelBack":
            termen1 = (4 - 2.1 * pow(particle[0], 2) + pow(particle[0], 4) / 3) * pow(particle[0], 2)
            termen2 = particle[0] * particle[1]
            termen3 = (-4 + 4 * pow(particle[1], 2)) * pow(particle[1], 2)
            result = termen1 + termen2 + termen3
        return result

    def PSOAlgorithm(self):
        t = 0
        best_particle_value_history = []
        while True:  # conditie de oprire
            t = t + 1
            random_p = random.uniform(0, 1)
            random_g = random.uniform(0, 1)
            self.particles_velocities_v = self.inertia_factor_w1 * self.particles_velocities_v + \
                self.particle_self_confidence_factor_w2 * random_p * (self.particles_best_position_p - self.particles_positions_x) + \
                self.swarm_confidence_factor_w3 * random_g * (self.best_particle_g - self.particles_positions_x)
            self.particles_positions_x += self.particles_velocities_v
            for index_particle in range (0, self.number_particles):
                if self.evaluate(self.particles_positions_x[index_particle]) < self.evaluate(self.particles_best_position_p[index_particle]):
                    self.particles_best_position_p[index_particle] = self.particles_positions_x[index_particle]
            for index_particle in range(0, self.number_particles):
                if self.evaluate(self.particles_best_position_p[index_particle]) < self.evaluate(self.best_particle_g):
                    self.best_particle_g = self.particles_best_position_p[index_particle]
                    # best_particle_value_history.append(self.best_particle_g)
                    # if len(best_particle_value_history) == self.maximum_lookback_steps:
                    #     for index in range (0, self.maximum_lookback_steps):
            print("Cea mai buna valoare gasita pana la iteratia t = {0} este {1} si are coordonatele {2}".format(t,
                self.evaluate(self.best_particle_g), self.best_particle_g))


if __name__ == "__main__":
    # pso = ParticleSwarmOptimization("Griewangk")
    pso = ParticleSwarmOptimization("Rastrigin")
    # pso = ParticleSwarmOptimization("Rosenbrock")
    # pso = ParticleSwarmOptimization("SixHumpCamelBack")
    # pso.PSOAlgorithm()
    # print(pso.particles_best_position_p)
    pso.PSOAlgorithm()
