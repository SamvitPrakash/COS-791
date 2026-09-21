from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class LSHADEResult:
    best_solution: np.ndarray
    best_fitness: float
    function_evaluations: int


class L_SHADE:

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        dimensions: int,
        lower_bound: float,
        upper_bound: float,
        initial_population_size: int = 100,
        minimum_population_size: int = 4,
        max_function_evaluations: int = 10_000,
        memory_size: int | None = None,
        p: float = 0.1,
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

        if initial_population_size < 4:
            raise ValueError(
                "initial_population_size must be at least 4."
            )

        if minimum_population_size < 4:
            raise ValueError(
                "minimum_population_size must be at least 4."
            )

        if minimum_population_size > initial_population_size:
            raise ValueError(
                "minimum_population_size cannot exceed "
                "initial_population_size."
            )

        if max_function_evaluations < initial_population_size:
            raise ValueError(
                "max_function_evaluations must be at least "
                "initial_population_size."
            )

        if memory_size is None:
            memory_size = initial_population_size

        if memory_size <= 0:
            raise ValueError(
                "memory_size must be greater than zero."
            )

        if not 0 < p <= 1:
            raise ValueError(
                "p must satisfy 0 < p <= 1."
            )

        self.objective_function = objective_function

        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.initial_population_size = (
            initial_population_size
        )

        self.minimum_population_size = (
            minimum_population_size
        )

        self.max_function_evaluations = (
            max_function_evaluations
        )

        self.memory_size = memory_size
        self.p = p

        self.rng = np.random.default_rng(
            seed
        )

        self.memory_f = np.full(
            self.memory_size,
            0.5,
            dtype=float,
        )

        self.memory_cr = np.full(
            self.memory_size,
            0.5,
            dtype=float,
        )

        self.memory_index = 0

        self.archive: list[np.ndarray] = []

    def optimise(self) -> LSHADEResult:

        population = self._initialise_population()

        population_size = len(
            population
        )

        fitness = np.empty(
            population_size,
            dtype=float,
        )

        function_evaluations = 0

        for i in range(population_size):

            fitness[i] = self._evaluate(
                population[i]
            )

            function_evaluations += 1

        best_index = int(
            np.argmax(fitness)
        )

        best_solution = (
            population[best_index].copy()
        )

        best_fitness = float(
            fitness[best_index]
        )

        while function_evaluations < self.max_function_evaluations:

            current_population_size = len(
                population
            )

            remaining_evaluations = (
                self.max_function_evaluations
                - function_evaluations
            )

            generation_size = min(
                current_population_size,
                remaining_evaluations,
            )

            (
                population,
                fitness,
                successful_f,
                successful_cr,
                improvements,
            ) = self._generation(
                population,
                fitness,
                generation_size,
            )

            function_evaluations += (
                generation_size
            )

            self._update_memory(
                successful_f,
                successful_cr,
                improvements,
            )

            current_best_index = int(
                np.argmax(fitness)
            )

            current_best_fitness = float(
                fitness[current_best_index]
            )

            if current_best_fitness > best_fitness:

                best_fitness = (
                    current_best_fitness
                )

                best_solution = (
                    population[
                        current_best_index
                    ].copy()
                )

            target_population_size = (
                self._calculate_population_size(
                    function_evaluations
                )
            )

            if (
                target_population_size
                < len(population)
            ):

                population, fitness = (
                    self._reduce_population(
                        population,
                        fitness,
                        target_population_size,
                    )
                )

        return LSHADEResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            function_evaluations=function_evaluations,
        )

    def _generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        generation_size: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[float],
        list[float],
        list[float],
    ]:

        population_size = len(
            population
        )

        new_population = population.copy()
        new_fitness = fitness.copy()

        successful_f: list[float] = []
        successful_cr: list[float] = []
        improvements: list[float] = []

        sorted_indices = np.argsort(
            fitness
        )[::-1]

        p_count = max(
            2,
            int(
                np.ceil(
                    self.p
                    * population_size
                )
            ),
        )

        p_best_indices = (
            sorted_indices[:p_count]
        )

        for i in range(
            generation_size
        ):

            target = population[i]

            memory_index = int(
                self.rng.integers(
                    0,
                    self.memory_size,
                )
            )

            f = self._sample_f(
                self.memory_f[
                    memory_index
                ]
            )

            cr = self._sample_cr(
                self.memory_cr[
                    memory_index
                ]
            )

            p_best_index = int(
                self.rng.choice(
                    p_best_indices
                )
            )

            p_best = population[
                p_best_index
            ]


            r1_index = self._select_r1(
                i,
                population_size,
            )

            r1 = population[
                r1_index
            ]


            r2_is_population, r2_index = (
                self._select_r2(
                    i,
                    r1_index,
                    population_size,
                )
            )

            if r2_is_population:

                r2 = population[
                    r2_index
                ]

            else:

                r2 = self.archive[
                    r2_index
                ]


            mutant = (
                target
                + f * (p_best - target)
                + f * (r1 - r2)
            )

            mutant = self._repair_bounds(
                mutant
            )


            trial = self._binomial_crossover(
                target,
                mutant,
                cr,
            )

            trial_fitness = self._evaluate(
                trial
            )

            if trial_fitness >= fitness[i]:

                self.archive.append(
                    target.copy()
                )

                new_population[i] = (
                    trial
                )

                new_fitness[i] = (
                    trial_fitness
                )

                improvement = (
                    trial_fitness
                    - fitness[i]
                )

                successful_f.append(
                    f
                )

                successful_cr.append(
                    cr
                )

                improvements.append(
                    improvement
                )

        while len(
            self.archive
        ) > population_size:

            remove_index = int(
                self.rng.integers(
                    0,
                    len(self.archive),
                )
            )

            self.archive.pop(
                remove_index
            )

        return (
            new_population,
            new_fitness,
            successful_f,
            successful_cr,
            improvements,
        )

    def _calculate_population_size(
        self,
        function_evaluations: int,
    ) -> int:

        progress = (
            function_evaluations
            / self.max_function_evaluations
        )

        progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

        population_size = (
            self.initial_population_size
            - (
                self.initial_population_size
                - self.minimum_population_size
            )
            * progress
        )

        return max(
            self.minimum_population_size,
            int(
                round(
                    population_size
                )
            ),
        )

    def _reduce_population(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        target_size: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        current_size = len(
            population
        )

        if target_size >= current_size:
            return (
                population,
                fitness,
            )

        sorted_indices = np.argsort(
            fitness
        )[::-1]

        keep_indices = sorted_indices[
            :target_size
        ]

        population = (
            population[
                keep_indices
            ].copy()
        )

        fitness = (
            fitness[
                keep_indices
            ].copy()
        )

        return (
            population,
            fitness,
        )

    def _initialise_population(
        self,
    ) -> np.ndarray:

        return self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(
                self.initial_population_size,
                self.dimensions,
            ),
        )

    def _evaluate(
        self,
        solution: np.ndarray,
    ) -> float:


        return float(
            self.objective_function(
                solution
            )
        )

    def _sample_f(
        self,
        memory_f: float,
    ) -> float:

        while True:

            f = (
                memory_f
                + 0.1
                * self.rng.standard_cauchy()
            )

            if f > 0:
                break

        return float(
            min(
                f,
                1.0,
            )
        )

    def _sample_cr(
        self,
        memory_cr: float,
    ) -> float:

        cr = (
            memory_cr
            + 0.1
            * self.rng.normal()
        )

        return float(
            np.clip(
                cr,
                0.0,
                1.0,
            )
        )

    def _select_r1(
        self,
        target_index: int,
        population_size: int,
    ) -> int:

        candidates = np.arange(
            population_size
        )

        candidates = candidates[
            candidates != target_index
        ]

        return int(
            self.rng.choice(
                candidates
            )
        )

    def _select_r2(
        self,
        target_index: int,
        r1_index: int,
        population_size: int,
    ) -> tuple[bool, int]:

        population_candidates = [
            index
            for index in range(
                population_size
            )
            if (
                index != target_index
                and index != r1_index
            )
        ]

        archive_size = len(
            self.archive
        )

        total_choices = (
            len(
                population_candidates
            )
            + archive_size
        )

        if total_choices == 0:
            raise RuntimeError(
                "Unable to select r2."
            )

        selected = int(
            self.rng.integers(
                0,
                total_choices,
            )
        )

        if selected < len(
            population_candidates
        ):

            return (
                True,
                population_candidates[
                    selected
                ],
            )

        archive_index = (
            selected
            - len(
                population_candidates
            )
        )

        return (
            False,
            archive_index,
        )

    def _binomial_crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        cr: float,
    ) -> np.ndarray:

        trial = target.copy()

        forced_dimension = int(
            self.rng.integers(
                0,
                self.dimensions,
            )
        )

        for j in range(
            self.dimensions
        ):

            if (
                self.rng.random()
                <= cr
                or j == forced_dimension
            ):
                trial[j] = mutant[j]

        return trial

    def _repair_bounds(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:


        return np.clip(
            vector,
            self.lower_bound,
            self.upper_bound,
        )

    def _update_memory(
        self,
        successful_f: list[float],
        successful_cr: list[float],
        improvements: list[float],
    ) -> None:
        if not successful_f:
            return

        improvements_array = np.asarray(
            improvements,
            dtype=float,
        )

        total_improvement = float(
            np.sum(
                improvements_array
            )
        )

        if total_improvement <= 0:
            return

        weights = (
            improvements_array
            / total_improvement
        )

        f_array = np.asarray(
            successful_f,
            dtype=float,
        )

        cr_array = np.asarray(
            successful_cr,
            dtype=float,
        )

        f_denominator = float(
            np.sum(
                weights * f_array
            )
        )

        if f_denominator > 0:

            self.memory_f[
                self.memory_index
            ] = (
                np.sum(
                    weights
                    * f_array
                    * f_array
                )
                / f_denominator
            )

        self.memory_cr[
            self.memory_index
        ] = float(
            np.sum(
                weights
                * cr_array
            )
        )

        self.memory_index = (
            self.memory_index + 1
        ) % self.memory_size