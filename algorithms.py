import time
import os
from typing import Any
import numpy as np
from dataclasses import dataclass, field

from objective import Objective


@dataclass
class Solution:
    y: float = float("inf")
    x: np.ndarray = field(default=None, repr=None)


@dataclass
class Logger:
    folder_name: str

    def __post_init__(self):
        if self.folder_name is not None:
            self.writer = open(os.path.join(self.folder_name, "stats.csv"), "w")
            self.columns = (
                "generation",
                "dt",
                "n_evals",
                "n_train_episodes",
                "n_train_timesteps",
                "best",
                "current",
                "sigma",
                "best_test",
                "current_test",
                "population_mean",
                "population_std",
                "best_median",
                "best_std",
                "current_median",
                "current_std",
            )
            self.writer.write(f'{",".join(self.columns)}\n')

    def write(self, x) -> Any:
        if self.folder_name is not None:
            assert len(x) == len(self.columns)
            self.writer.write(f'{",".join(map(str, x))}\n')
            self.writer.flush()

    def close(self):
        if self.folder_name is not None:
            self.writer.close()



class State:
    def __init__(self, data_folder, test_gen, lamb, revaluate_best_after: int = None):
        self.counter = 0
        self.best = Solution()
        self.mean = Solution()
        self.best_test = float("inf")
        self.mean_test = float("inf")
        self.tic = time.perf_counter()
        self.data_folder = data_folder
        self.logger = Logger(data_folder)
        self.test_gen: int = test_gen
        self.lamb: int = lamb
        self.mean_test = None
        self.best_test = None
        self.best_median = None
        self.best_std = None
        self.mean_median = None
        self.mean_std = None
        self.used_budget = 0
        self.time_since_best_update = 0
        self.revaluate_best_every = revaluate_best_after

    def update(
        self,
        problem: Objective, 
        best_offspring: Solution,
        mean: Solution,
        sigma: float,
        f: np.ndarray
    ) -> None:
        self.counter += 1
        self.time_since_best_update += 1
        self.used_budget += len(f)

        if self.revaluate_best_every is not None and self.revaluate_best_every < self.time_since_best_update:
            self.time_since_best_update = 0
            self.best.y = problem.eval_sequential(self.best.x)

        toc = time.perf_counter()
        dt = toc - self.tic
        self.tic = toc

        if best_offspring.y < self.best.y:
            self.best = best_offspring
            self.time_since_best_update = 0 
        self.mean = mean

        n_evals = self.counter * self.lamb
        print(
            f"counter: {self.counter}, dt: {dt:.3f} n_evals {n_evals}, best {-self.best.y}, mean: {-mean.y}, sigma: {sigma} ",
        )

        if self.counter % self.test_gen == 0:
            self.best_test, self.best_median, self.best_std = problem.test(
                self.best.x,
                "rgb_array_list",
                False,
                self.data_folder,
                name=f"t-{self.counter}-best",
            )
            print("Test with best x (max):", self.best_test)
            self.mean_test, self.mean_median, self.mean_std = problem.test(
                self.mean.x,
                "rgb_array_list",
                False,
                self.data_folder,
                name=f"t-{self.counter}-mean",
            )
            print("Test with mean x (max):", self.mean_test)


        self.logger.write(
            (
                self.counter,
                dt,
                n_evals,
                problem.n_train_episodes,
                problem.n_train_timesteps,
                -self.best.y,
                -self.mean.y,
                sigma,
                self.best_test,
                self.mean_test,
                np.mean(-f),
                np.std(-f),
                self.best_median,
                self.best_std,
                self.mean_median,
                self.mean_std,
            )
        )


@dataclass
class UncertaintyHandling:
    active: bool
    update_timer: int = 1
    averaging_f: float = 1.0
    averaging: int = 1
    max_averaging: float = 25.0
    targetnoise: float = 0.12
    S: float = 0.12

    def update(self, problem, f, X, n, lambda_, state: State):
        idx = np.argsort(f)
        self.update_timer -= 1
        if self.active and self.update_timer <= 0:
            fu = f.copy()
            self.update_timer = int(np.ceil(40 / lambda_))
            # find two random individuals for re-evaluation
            i1, i2 = np.random.choice(lambda_, size=2, replace=False)
            # re-evaluate
            fu[i1] = np.median(
                [problem.eval_sequential(X[:, i1]) for _ in range(self.averaging)]
            )
            fu[i2] = np.median(
                [problem.eval_sequential(X[:, i1]) for _ in range(self.averaging)]
            )
            state.used_budget += self.averaging * 2

            idx2 = np.argsort(fu)

            # compute rank difference statistics (inspired by Hansen 2008, but simplified)
            self.S = abs(idx[i1] - idx2[i1]) + abs(idx[i2] - idx2[i2])
            self.S /= 2 * (lambda_ - 1)

            # accumulate
            c_uh = max(1.0, 10.0 * lambda_ / n)

            self.averaging_f *= np.exp(c_uh * (self.S - self.targetnoise))
            self.averaging_f = max(1.0, min(self.max_averaging, self.averaging_f))

            # adapt amount of averaging
            self.averaging = int(round(self.averaging_f))

            # incorporate additional fitness evaluation
            f[i1] = 0.5 * (f[i1] + fu[i1])
            f[i2] = 0.5 * (f[i2] + fu[i2])
            print(f"Updated UCH S: {self.S} n_avg {self.averaging}")

        return idx, f

    def should_update(self):
        return not self.active or self.S <= self.targetnoise


@dataclass
class WeightedRecombination:
    mu: int
    lambda_: int
    mueff: float = None

    def __post_init__(self):
        wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = wi_raw / np.sum(wi_raw)
        self.w_all = np.r_[self.w, -self.w[::-1]]
        self.mueff = 1 / np.sum(np.power(self.w, 2))

    
def init(dim, lb, ub, method="zero"):
    if method == "zero":
        return np.zeros((dim, 1))
    elif method == "uniform":
        return np.random.uniform(lb, ub, size=(dim, 1))
    elif method == "gauss":
        return np.random.normal(size=(dim, 1))
    raise ValueError()

def init_lambda(n, method="n/2"):
    """
        range:      2*mu < lambda < 2*n + 10 
        default:    4 + floor(3 * ln(n))     

    """
    if method == "default":
        return (4 + np.floor(3 * np.log(n))).astype(int) 
    
    elif method == "n/2":
        return np.floor(n / 2).astype(int)
    else:
        raise ValueError()


@dataclass
class DR1:
    n: int
    budget: int = 25_000
    mu: int = None
    lambda_: int = None
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False
    mirrored: bool = True
    revaluate_best_after: int = None

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.mu or self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")
    
    def __call__(self, problem: Objective):
        beta_scale = 1 / self.n
        beta = np.sqrt(beta_scale)
        zeta = np.array([5 / 7, 7 / 5])
        sigma = np.ones((self.n, 1)) * self.sigma0
        root_pi = np.sqrt(2 / np.pi)

        x_prime = init(self.n, problem.lb, problem.ub, self.initialization)

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
        uch = UncertaintyHandling(self.uncertainty_handling)
        weights = WeightedRecombination(self.mu, self.lambda_)

        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        
        try:
            while self.budget > state.used_budget:
                Z = np.random.normal(size=(self.n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])

                zeta_i = np.random.choice(zeta, (1, self.lambda_))
                Y = (zeta_i * (sigma * Z))
                X = x_prime + Y
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_, state)

                mu_best = idx[: self.mu]
                idx_min = idx[0]

                y_prime = np.sum(Y[:, mu_best] * weights.w, axis=1, keepdims=True)
                x_prime = x_prime + y_prime
                z_prime = np.sum(Z[:, mu_best] * weights.w, axis=1, keepdims=True) * np.sqrt(weights.mueff)
                if uch.should_update():
                    zeta_w = zeta_i[:, idx_min][0]
                    zeta_sel = np.exp(z_prime - root_pi)
                    sigma *= (
                        np.power(zeta_w, beta) * np.power(zeta_sel, beta_scale)
                    ).reshape(-1, 1)

                state.update(
                    problem,
                    Solution(f[idx_min], X[:, idx_min].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x_prime.copy()),
                    np.mean(sigma),
                    f,
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean
    

@dataclass
class DR2:
    n: int
    budget: int = 25_000
    mu: int = None
    lambda_: int = None
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False
    mirrored: bool = True
    revaluate_best_after: int = None

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.mu or self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")
    
    def __call__(self, problem: Objective):
        beta_scale = 1 / self.n
        beta = np.sqrt(beta_scale)
        c = beta

        zeta = np.zeros((self.n, 1))
        sigma_local = np.ones((self.n, 1)) * self.sigma0
        sigma = self.sigma0

        c1 = np.sqrt(c / (2 - c))
        c2 = np.sqrt(self.n) * c1
        c3 = 1 + (1 / (5 * self.n))

        weights = WeightedRecombination(self.mu, self.lambda_)
        x_prime = init(self.n, problem.lb, problem.ub, self.initialization)

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > state.used_budget:
                Z = np.random.normal(size=(self.n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                Y = sigma * (sigma_local * Z)
                X = x_prime + Y
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_, state)
                mu_best = idx[: self.mu]
                idx_min = idx[0]

                z_prime = np.sum(Z[:, mu_best] * weights.w, axis=1, keepdims=True) * np.sqrt(weights.mueff)
                y_prime = np.sum(Y[:, mu_best] * weights.w, axis=1, keepdims=True)
                x_prime = x_prime + y_prime
                zeta = ((1 - c) * zeta) + (c * z_prime)
                if uch.should_update():
                    sigma = sigma * np.power(
                        np.exp((np.linalg.norm(zeta) / c2) - c3), beta
                    )
                    sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)

                state.update(
                    problem,
                    Solution(f[idx_min], X[:, idx_min].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x_prime.copy()),
                    np.mean(sigma),
                    f,
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class CSA:
    n: int
    budget: int = 25_000
    lambda_: int = None
    mu: float = None
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False
    mirrored: bool = True
    revaluate_best_after: int = None

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")


    def __call__(self, problem: Objective):
        weights = WeightedRecombination(self.mu, self.lambda_)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))
        
        c_s = (weights.mueff + 2) / (self.n + weights.mueff + 5)
        d_s = 1 + c_s + 2 * max(0, np.sqrt((weights.mueff - 1) / (self.n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * weights.mueff)

        x_prime = init(self.n, problem.lb, problem.ub, self.initialization)
        sigma = self.sigma0
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > state.used_budget:
                Z = np.random.normal(size=(self.n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                X = x_prime + (sigma * Z)
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_, state)
                mu_best = idx[: self.mu]
                idx_min = idx[0]
                z_prime = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                x_prime = x_prime + (sigma * z_prime)
                s = ((1 - c_s) * s) + (sqrt_s * z_prime)

                if uch.should_update():
                    sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))

                state.update(
                    problem,
                    Solution(f[idx_min], X[:, idx_min].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x_prime.copy()),
                    np.mean(sigma),
                    f,
                )

        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class MAES:
    n: int
    budget: int = 25_000
    lambda_: int = None
    mu: float = None
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False
    mirrored: bool = True
    revaluate_best_after: int = None

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")

    def __call__(self, problem: Objective):
        weights = WeightedRecombination(self.mu, self.lambda_)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))
        c_s = (weights.mueff + 2) / (self.n + weights.mueff + 5)
        c_1 = 2 / (pow(self.n + 1.3, 2) + weights.mueff)
        c_mu = min(1 - c_1, 2 * (weights.mueff - 2 + (1 / weights.mueff)) / (pow(self.n + 2, 2) + weights.mueff))
        d_s = 1 + c_s + 2 * max(0, np.sqrt((weights.mueff - 1) / (self.n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * weights.mueff)

        x_prime = init(self.n, problem.lb, problem.ub, self.initialization)
        sigma = self.sigma0
        M = np.eye(self.n)
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > state.used_budget:
                Z = np.random.normal(size=(self.n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                D = M.dot(Z)
                X = x_prime + (sigma * D)
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_, state)
                mu_best = idx[: self.mu]
                idx_min = idx[0]
                z_prime = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                d_prime = np.sum(weights.w * D[:, mu_best], axis=1, keepdims=True)
                x_prime = x_prime + (sigma * d_prime)
                s = ((1 - c_s) * s) + (sqrt_s * z_prime)

                if uch.should_update():
                    M = (
                        ((1 - 0.5 * c_1 - 0.5 * c_mu) * M)
                        + ((0.5 * c_1) * M.dot(s).dot(s.T))
                        + ((0.5 * c_mu * weights.w_all) * D[:, idx]).dot(Z[:, idx].T)
                    )
                    sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))


                state.update(
                    problem,
                    Solution(f[idx_min], X[:, idx_min].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x_prime.copy()),
                    np.mean(sigma),
                    f,
                )

        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class ARSV1:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     # learning rate alpha
    lambda_: int = 16       # n offspring for each direction
    mu: int = 16            # best offspring
    eta: float = 0.03       # noise parameter

    def __call__(self, problem: Objective):
        m = np.zeros((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_ * 2)
        try:
            while self.budget > state.used_budget:
                delta = np.random.normal(size=(self.n, self.lambda_))

                neg = m - (self.eta * delta)
                pos = m + (self.eta * delta)

                neg_reward = -problem(neg)
                pos_reward = -problem(pos)
                
                best_rewards = np.maximum(neg_reward, pos_reward)
                idx = np.argsort(best_rewards)[::-1]

                f = np.r_[neg_reward, pos_reward]
                sigma_rewards = f.std()
                weight = self.sigma0 / (self.lambda_ * sigma_rewards)

                delta_rewards = pos_reward - neg_reward
                m += (weight * (delta_rewards[idx] * delta[:, idx]).sum(axis=1, keepdims=True))

                best_idx = idx[0]
                if neg_reward[best_idx] > pos_reward[best_idx]:
                    best = neg[:, best_idx]
                else:
                    best = pos[:, best_idx]

                state.update(
                    problem,
                    Solution(-best_rewards[best_idx],  best.copy()),
                    Solution(-np.mean(best_rewards), m.copy()),
                    sigma_rewards,
                    f
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean
