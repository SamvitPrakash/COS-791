import numpy as np

class LateAcceptanceDE:
    def __init__(self, objective_func, K, max_fes=10000, pop_size=30, F=0.5, CR=0.8, L=50):
        self.objective_func = objective_func
        self.D = K
        self.max_fes = max_fes
        self.NP = pop_size
        self.F = F
        self.CR = CR
        self.L = L
        self.lb = 1.0
        self.ub = 254.0

    def optimize(self):
        # 1. Initialize population uniformly within boundaries
        pop = np.zeros((self.NP, self.D))
        for i in range(self.NP):
            pop[i] = np.sort(np.random.uniform(self.lb, self.ub, self.D))

        fitness = np.array([self.objective_func(ind) for ind in pop])
        fes = self.NP

        best_idx = np.argmax(fitness)
        best_sol = np.copy(pop[best_idx])
        best_fit = fitness[best_idx]

        # 2. Late Acceptance Buffer
        late_fitness = np.full(self.L, best_fit, dtype=np.float64)
        v = 0

        # 3. Evolution Loop
        while fes < self.max_fes:
            for i in range(self.NP):
                if fes >= self.max_fes:
                    break

                # Mutation: DE/rand/1
                candidates = [idx for idx in range(self.NP) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                v_donor = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Boundary handling
                mask = (v_donor < self.lb) | (v_donor > self.ub)
                v_donor[mask] = np.random.uniform(self.lb, self.ub, np.sum(mask))

                # Crossover: Binomial
                j_rand = np.random.randint(0, self.D)
                u_trial = np.empty(self.D)
                for j in range(self.D):
                    if np.random.rand() < self.CR or j == j_rand:
                        u_trial[j] = v_donor[j]
                    else:
                        u_trial[j] = pop[i, j]

                u_trial = np.sort(u_trial)

                # Evaluation
                trial_fit = self.objective_func(u_trial)
                fes += 1

                # Late Acceptance Selection (Maximization)
                if (trial_fit >= fitness[i]) or (trial_fit >= late_fitness[v]):
                    pop[i] = u_trial
                    fitness[i] = trial_fit

                late_fitness[v] = fitness[i]
                v = (v + 1) % self.L

                if fitness[i] > best_fit:
                    best_fit = fitness[i]
                    best_sol = np.copy(pop[i])

        return np.sort(np.rint(best_sol).astype(int)), best_fit
