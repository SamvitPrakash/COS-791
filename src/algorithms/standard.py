from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class StandardDEResult:
    best_solution: np.ndarray
    best_fitness: float
    function_evaluations: int


class StandardDE:
    """Standard Differential Evolution: DE/rand/1/bin with greedy selection."""

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        dimensions: int,
        lower_bound: float,
        upper_bound: float,
        population_size: int = 100,
        max_function_evaluations: int = 10_000,
        f: float = 0.5,
        cr: float = 0.9,
        seed: int | None = None,
    ) -> None:

        if dimensions <= 0:
            raise ValueError(
                "dimensions must be greater than zero."
            )

        if lower_bound >= upper_bound:
            raise ValueError(
                "lower_bound must be smaller than upper_bound."
            )

        if population_size < 4:
            raise ValueError(
                "population_size must be at least 4."
            )

        if max_function_evaluations < population_size:
            raise ValueError(
                "max_function_evaluations must be at least "
                "population_size."
            )

        if not 0 < f <= 2:
            raise ValueError(
                "f must be a positive scale factor."
            )

        if not 0 <= cr <= 1:
            raise ValueError(
                "cr must satisfy 0 <= cr <= 1."
            )

        self.objective_function = objective_function

        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.population_size = population_size
        self.max_function_evaluations = max_function_evaluations

        self.f = f
        self.cr = cr

        self.rng = np.random.default_rng(seed)

    def optimise(self) -> StandardDEResult:

        population = self._initialise_population()

        fitness = np.empty(
            self.population_size,
            dtype=float,
        )

        function_evaluations = 0

        for i in range(self.population_size):
            fitness[i] = self._evaluate(
                population[i]
            )

            function_evaluations += 1

        best_index = int(
            np.argmax(fitness)
        )

        best_solution = population[best_index].copy()
        best_fitness = float(fitness[best_index])

        while (
            function_evaluations + self.population_size
            <= self.max_function_evaluations
        ):

            population, fitness = self._generation(
                population,
                fitness,
            )

            function_evaluations += self.population_size

            generation_best_index = int(
                np.argmax(fitness)
            )

            generation_best_fitness = float(
                fitness[generation_best_index]
            )

            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_solution = (
                    population[generation_best_index].copy()
                )

        return StandardDEResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            function_evaluations=function_evaluations,
        )

    def _generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        population_size = len(population)

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(population_size):

            target = population[i]

            r1_index, r2_index, r3_index = self._select_three(
                i,
                population_size,
            )

            mutant = (
                population[r1_index]
                + self.f
                * (population[r2_index] - population[r3_index])
            )

            mutant = self._repair_bounds(mutant)

            trial = self._binomial_crossover(
                target,
                mutant,
                self.cr,
            )

            trial_fitness = self._evaluate(trial)

            if trial_fitness >= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        return new_population, new_fitness

    def _initialise_population(self) -> np.ndarray:

        return self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(
                self.population_size,
                self.dimensions,
            ),
        )

    def _evaluate(self, solution: np.ndarray) -> float:

        return float(
            self.objective_function(solution)
        )

    def _select_three(
        self,
        target_index: int,
        population_size: int,
    ) -> tuple[int, int, int]:

        candidates = np.arange(population_size)
        candidates = candidates[candidates != target_index]

        r1, r2, r3 = self.rng.choice(
            candidates,
            size=3,
            replace=False,
        )

        return int(r1), int(r2), int(r3)

    def _binomial_crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        cr: float,
    ) -> np.ndarray:

        trial = target.copy()

        random_dimension = int(
            self.rng.integers(0, self.dimensions)
        )

        for j in range(self.dimensions):
            if (
                self.rng.random() <= cr
                or j == random_dimension
            ):
                trial[j] = mutant[j]

        return trial

    def _repair_bounds(self, vector: np.ndarray) -> np.ndarray:

        return np.clip(
            vector,
            self.lower_bound,
            self.upper_bound,
        )