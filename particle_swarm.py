import numpy as np
import matplotlib.pyplot as plt
from objective_function import ObjectiveFunction
from typing import Union

class Particle:
    """PSO algorithm Particle class"""

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        position: np.ndarray,
        bounds: Union[np.ndarray, None],
        load: Union[np.ndarray, None],
    ) -> None:
        self.dim = bounds.shape[1]
        self.bounds = bounds
        self.load = load
        self.objective_function = objective_function
        self.position = position
        self.adjust_to_constraints()
        self.fitness = self.evaluate_fitness()
        self.velocity = np.random.uniform(-1, 1, self.dim)
        self.pbest = np.copy(self.position)

    def evaluate_fitness(self):
        """Calculates the fitness of a a particle depending on its posistion and an objective function"""
        return self.objective_function.evaluate(self.position)

    def update_velocity(self, w, c1, c2, gbest):
        """Updates the velocity of the current particle."""
        r1 = np.random.random()
        r2 = np.random.random()
        cogn_velocity = c1 * r1 * (self.pbest - self.position)
        social_velocity = c2 * r2 * (gbest - self.position)
        self.velocity = w * self.velocity + cogn_velocity + social_velocity

    def adjust_to_constraints(self):
        if self.bounds is not None:
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        if self.load is not None:
            self.position[:-3] = self.load - (
                self.position[-3] + self.position[-2] + self.position[-1]
            )
            self.position[self.position < 0] = 0

    def update_position(self):
        """Updates the positions of the current particle while respecting load and bounds constraint."""
        self.position = self.position + self.velocity
        self.adjust_to_constraints()

    


class PSO:
    """PSO Algorithm Class Definition"""

    def __init__(
        self,
        n_particles: int,
        max_iters: int,
        objective_function: ObjectiveFunction,
        bounds: Union[np.ndarray, None],
        load: Union[np.ndarray, None],
        minimize: bool = True,
    ) -> None:
        self.minimize = minimize
        self.n_particles = n_particles
        self.bounds = bounds
        self.load = load
        self.max_iters = max_iters
        self.objective_function = objective_function
        self.particles = self._init_particles()
        self.gbest = min(self.particles, key=lambda particle: particle.fitness).pbest
        self.run_results = []
        self.history = []

    def _init_particles(self):
        """Initializes the positions of the particles in the PSO algorithm by sampling from a uniform distribution"""
        particles = []
        for _ in range(self.n_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1])
            particle = Particle(
                self.objective_function, position, self.bounds, self.load
            )
            particles.append(particle)
        return particles

    def _step(self, w, c1, c2):
        """Computes ones step of the SPO algorithm for n particles: evalute -> check -> update"""
        for idx in range(self.n_particles):
            self.particles[idx].update_velocity(w, c1, c2, self.gbest)
            self.particles[idx].update_position()

            new_fitness = self.particles[idx].evaluate_fitness()

            if self.minimize is False:
                if new_fitness > self.particles[idx].fitness:
                    self.particles[idx].fitness = new_fitness
                    self.particles[idx].pbest = np.copy(self.particles[idx].position)
                    if new_fitness > self.objective_function.evaluate(self.gbest):
                        self.gbest = np.copy(self.particles[idx].position)
            else:
                if new_fitness < self.particles[idx].fitness:
                    self.particles[idx].fitness = new_fitness
                    self.particles[idx].pbest = np.copy(self.particles[idx].position)
                    if new_fitness < self.objective_function.evaluate(self.gbest):
                        self.gbest = np.copy(self.particles[idx].position)
        self.history.append(
            {
                "gbest": self.gbest,
                "particles": self.particles,
                "fitness": self.objective_function.evaluate(self.gbest),
            }
        )

    def run(self, w, c1, c2):
        self.run_results = []
        self.history = []
        """Executes a full run of the PSO algorithm on all particles for a specified number of iterations and prints resutls."""
        print("Executing PSO algorithm")
        print(f"Number of particles: \t{self.n_particles}")
        print(f"Number of Iterations: \t{self.max_iters}")
        print(f"w:\t{w}")
        print(f"c1:\t{c1}")
        print(f"c2:\t{c2}")
        print("\n")
        for idx in range(self.max_iters):
            self._step(w, c1, c2)
            print(
                f"\tStep {idx}/{self.max_iters}:\t gbest: [{self.gbest[0]:.2f}...{self.gbest[30]:.2f}...{self.gbest[-1]:.2f}] fitness: {self.objective_function.evaluate(self.gbest):.2f}"
            )
            self.run_results.append(self.objective_function.evaluate(self.gbest))
        print("\nPSO run finished.")

    def results(self):
        """Prints the results of the PSO algorithm after a successful run."""
        print(f"PSO Run Results:\n")
        print(
            f"\tOptimal PV size: {self.gbest[-2]:.2f}\t\t Interval: [{self.bounds[0][-2]:.2f},{self.bounds[1][-2]:.2f}]"
        )
        print(
            f"\tOptimal ESS Capacity: {self.gbest[-1]:.2f}\t Interval: [{self.bounds[0][-1]:.2f},{self.bounds[1][-1]:.2f}]"
        )
        print(
            f"\tOptimal SF Capacity: {self.gbest[-3]:.2f}\t Interval: [{self.bounds[0][-3]:.2f},{self.bounds[1][-3]:.2f}]"
        )
        print(
            f"\tOptimal values for PD: \t\tInterval: [{self.bounds[0][0]:.2f},{self.bounds[1][0]:.2f}]\n"
        )
        for idx, value in enumerate(self.gbest[:-3]):
            loads = f"\t\tLoad: {self.load[idx]:.4f}" if self.load is not None else ""
            msg = f"\t\t PD_{idx+1}: {value:.4f}" + loads
            print(msg)

    def plot_run(self):
        x = range(self.max_iters)
        y = self.run_results
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel("Step Number")
        plt.ylabel("Fitness of gbest")
        plt.title("PSO Run plot.")
        plt.show()

    def plot_particle_movement(self, particle_idx=Union[str, int]):
        """Plots the movement of a chosen particle during the run in a 3D plane"""
        # pour ajouter Parreto, tu changes les composants du vecteur position ([pd_1, pd_2, ..., sf, pv, ess]: 3843)
        # vect =  [pd_1, .., emission, cost]
        # x = vect[-1] cost
        # y = vect[-2] emission 
        if particle_idx == "gbest":
            history = [hist_dict["gbest"] for hist_dict in self.history]
        else:
            history = [hist_dict["particles"][particle_idx].position for hist_dict in self.history]

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        for idx, position in enumerate(history):
            x = position[-2]
            y = position[-1]
            z = self.objective_function.evaluate(position)
            # pour changer le type du graphe
            ax.scatter(
                x,
                y,
                z,
            )
            ax.text(x, y, z, f"{idx+1}", color="black", fontsize=8)
        ax.set_xlabel("PV")
        ax.set_ylabel("ESS")
        ax.set_zlabel("Fitness")

    def __str__(self):
        """Desciption of the PSO algorithm object."""
        msg = f"PSO Algorithm:\n"
        msg += f"\tNumber of particles: {self.n_particles}\n"
        msg += f"\tMaximum Number of Iterations: {self.max_iters}\n"
        msg += f"\tsolution Space dimension: {self.bounds.shape[1]}\n"
        msg += "\tObjective: Minimize\n" if self.minimize else "Objective: Maximize\n"
        msg += "\n\tParticles: \n"
        for idx, part in enumerate(self.particles):
            msg += f"\t\t|Particle {idx+1}:\tInitial Fitness: {part.fitness}\t Initial Position: [{part.position[0]:.4f} ... {part.position[-1]:.4f}]\n"
        return msg