from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class JADEResult:
    best_solution: np.ndarray
    best_fitness: float
    function_evaluations: int


class JADE:
    """Adaptive DE with Optional External Archive (JADE).

    Uses DE/current-to-pbest/1 mutation, per-individual F ~ Cauchy(muF, 0.1)
    and CR ~ N(muCR, 0.1), with muF/muCR adapted each generation via an
    exponential (Lehmer-mean for F) update, and an optional external
    archive of recently replaced parents.
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        dimensions: int,
        lower_bound: float,
        upper_bound: float,
        population_size: int = 100,
        max_function_evaluations: int = 10_000,
        p: float = 0.1,
        c: float = 0.1,
        use_archive: bool = True,
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

        if not 0 < p <= 1:
            raise ValueError(
                "p must satisfy 0 < p <= 1."
            )

        if not 0 < c <= 1:
            raise ValueError(
                "c must satisfy 0 < c <= 1."
            )

        self.objective_function = objective_function

        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.population_size = population_size
        self.max_function_evaluations = max_function_evaluations

        self.p = p
        self.c = c
        self.use_archive = use_archive

        self.rng = np.random.default_rng(seed)

        self.mu_f = 0.5
        self.mu_cr = 0.5

        self.archive: list[np.ndarray] = []

    def optimise(self) -> JADEResult:

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

            (
                population,
                fitness,
                generation_success_f,
                generation_success_cr,
            ) = self._generation(
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

            self._update_parameters(
                generation_success_f,
                generation_success_cr,
            )

        return JADEResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            function_evaluations=function_evaluations,
        )

    def _generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[float],
        list[float],
    ]:

        population_size = len(population)

        new_population = population.copy()
        new_fitness = fitness.copy()

        successful_f: list[float] = []
        successful_cr: list[float] = []

        sorted_indices = np.argsort(fitness)[::-1]

        p_count = max(
            2,
            int(np.ceil(self.p * population_size)),
        )

        p_best_indices = sorted_indices[:p_count]

        for i in range(population_size):

            target = population[i]

            f = self._sample_f(self.mu_f)
            cr = self._sample_cr(self.mu_cr)

            p_best_index = int(
                self.rng.choice(p_best_indices)
            )

            p_best = population[p_best_index]

            r1_index = self._select_r1(
                i,
                population_size,
            )

            r1 = population[r1_index]

            r2_is_population, r2_index = self._select_r2(
                i,
                r1_index,
                population_size,
            )

            if r2_is_population:
                r2 = population[r2_index]
            else:
                r2 = self.archive[r2_index]

            mutant = (
                target
                + f * (p_best - target)
                + f * (r1 - r2)
            )

            mutant = self._repair_bounds(mutant)

            trial = self._binomial_crossover(
                target,
                mutant,
                cr,
            )

            trial_fitness = self._evaluate(trial)

            if trial_fitness >= fitness[i]:

                if self.use_archive:
                    self.archive.append(target.copy())

                new_population[i] = trial
                new_fitness[i] = trial_fitness

                successful_f.append(f)
                successful_cr.append(cr)

        if self.use_archive:
            while len(self.archive) > population_size:
                remove_index = int(
                    self.rng.integers(0, len(self.archive))
                )
                self.archive.pop(remove_index)

        return (
            new_population,
            new_fitness,
            successful_f,
            successful_cr,
        )

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

    def _sample_f(self, mu_f: float) -> float:

        while True:
            f = (
                mu_f
                + 0.1 * self.rng.standard_cauchy()
            )

            if f > 0:
                break

        return float(min(f, 1.0))

    def _sample_cr(self, mu_cr: float) -> float:

        cr = mu_cr + 0.1 * self.rng.normal()

        return float(np.clip(cr, 0.0, 1.0))

    def _select_r1(
        self,
        target_index: int,
        population_size: int,
    ) -> int:

        candidates = np.arange(population_size)
        candidates = candidates[candidates != target_index]

        return int(self.rng.choice(candidates))

    def _select_r2(
        self,
        target_index: int,
        r1_index: int,
        population_size: int,
    ) -> tuple[bool, int]:

        archive_size = len(self.archive) if self.use_archive else 0

        population_candidates = [
            index
            for index in range(population_size)
            if index != target_index and index != r1_index
        ]

        total_choices = len(population_candidates) + archive_size

        if total_choices == 0:
            raise RuntimeError(
                "Unable to select r2."
            )

        selected = int(
            self.rng.integers(0, total_choices)
        )

        if selected < len(population_candidates):
            return True, population_candidates[selected]

        archive_index = selected - len(population_candidates)

        return False, archive_index

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

    def _update_parameters(
        self,
        successful_f: list[float],
        successful_cr: list[float],
    ) -> None:

        if not successful_f:
            return

        f_array = np.asarray(successful_f, dtype=float)
        cr_array = np.asarray(successful_cr, dtype=float)

        # Lehmer mean for F, arithmetic mean for CR.
        lehmer_mean_f = float(
            np.sum(f_array ** 2) / np.sum(f_array)
        )

        mean_cr = float(np.mean(cr_array))

        self.mu_f = (
            (1 - self.c) * self.mu_f + self.c * lehmer_mean_f
        )

        self.mu_cr = (
            (1 - self.c) * self.mu_cr + self.c * mean_cr
        )