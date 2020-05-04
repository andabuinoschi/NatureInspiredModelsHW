import random

import numpy as np
from copy import copy

from math import sqrt, cos, pi


class ParticleSwarmOptimization:
    def __init__(self, function_type="Griewangk"):
        self.maximum_lookback_steps = 50
        self.precision = 9
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
        self.best_particle_g = copy(self.particles_positions_x[random.randint(0, self.number_particles)])

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
        self.particles_best_position_p = copy(self.particles_positions_x)

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

    def update_velocities(self):
        for i in range(0, self.number_particles):
            for j in range(0, self.number_variables):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                cognitive_factor = self.inertia_factor_w1 * r1 * (self.particles_best_position_p[i][j] - self.particles_positions_x[i][j])
                social_factor = self.swarm_confidence_factor_w3 * r2 * (self.best_particle_g[j] - self.particles_positions_x[i][j])
                self.particles_velocities_v[i][j] = self.inertia_factor_w1 * self.particles_velocities_v[i][j] + cognitive_factor + social_factor

    def update_position(self):
        for i in range(0, self.number_particles):
            self.particles_positions_x[i] = self.particles_positions_x[i] + self.particles_velocities_v[i]
            if self.name != "SixHumpCamelBack":
                for j in range(0, self.number_variables):
                    # adjust maximum position if necessary
                    if self.particles_positions_x[i][j] > self.b1:
                        self.particles_positions_x[i][j] = self.b1

                    # adjust minimum position if neseccary
                    if self.particles_positions_x[i][j] < self.a1:
                        self.particles_positions_x[i][j] = self.a1
            elif self.name == "SixHumpCamelBack":
                if self.particles_positions_x[i][0] > self.b1:
                    self.particles_positions_x[i][0] = self.b1
                if self.particles_positions_x[i][0] < self.a1:
                    self.particles_positions_x[i][0] = self.a1
                if self.particles_positions_x[i][1] > self.b2:
                    self.particles_positions_x[i][1] = self.b2
                if self.particles_positions_x[i][1] < self.a2:
                    self.particles_positions_x[i][1] = self.a2

    def PSOAlgorithm(self):
        t = 0
        best_particle_value_history = []
        local = False
        while not local:  # conditie de oprire
            t = t + 1
            self.update_velocities()
            self.update_position()
            for index_particle in range(0, self.number_particles):
                if self.evaluate(self.particles_positions_x[index_particle]) < self.evaluate(
                        self.particles_best_position_p[index_particle]):
                    self.particles_best_position_p[index_particle] = copy(self.particles_positions_x[index_particle])
            for index_particle in range(0, self.number_particles):
                if self.evaluate(self.particles_best_position_p[index_particle]) < self.evaluate(self.best_particle_g):
                    self.best_particle_g = copy(self.particles_best_position_p[index_particle])
                    # if len(best_particle_value_history) != self.maximum_lookback_steps:
                    #     best_particle_value_history.append(self.best_particle_g)
                    # else:
                    #     deltas = [abs(self.evaluate(self.best_particle_g) - self.evaluate(best_particle_value_history[index])) for index in range (0, self.maximum_lookback_steps)]
                    #     best_particle_value_history.pop(0)
                    #     if all(delta <= pow(10, - self.precision) for delta in deltas):
                    #         print("Cea mai buna valoare gasita pana la iteratia t = {0} este {1} si are coordonatele {2}".format(t,self.evaluate(self.best_particle_g),self.best_particle_g))
                    #         local = True
            print("Cea mai buna valoare gasita pana la iteratia t = {0} este {1} si are coordonatele {2}".format(t,
                                                                                                                 self.evaluate(
                                                                                                                     self.best_particle_g),
                                                                                                                 self.best_particle_g))


if __name__ == "__main__":
    # pso = ParticleSwarmOptimization("Griewangk")
    # pso = ParticleSwarmOptimization("Rastrigin")
    pso = ParticleSwarmOptimization("Rosenbrock")
    # pso = ParticleSwarmOptimization("SixHumpCamelBack")
    pso.PSOAlgorithm()
    # last_elements = [5, 2, 1, 6, 7, 7, 6]
    # last_elements.pop(1)
    # print(last_elements)
