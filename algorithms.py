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


def init(dim, lb, ub, method="zero"):
    if method == "zero":
        return np.zeros(dim)
    elif method == "uniform":
        return np.random.uniform(lb, ub)
    elif method == "gauss":
        return np.random.normal(size=dim)
    raise ValueError()


class State:
    def __init__(self, data_folder, test_gen, lamb):
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

    def update(
        self,
        problem: Objective,
        best_offspring: Solution,
        mean: Solution,
        sigma: float,
        f: np.ndarray
    ) -> None:
        self.counter += 1
        toc = time.perf_counter()
        dt = toc - self.tic
        self.tic = toc

        if best_offspring.y < self.best.y:
            self.best = best_offspring
        self.mean = mean

        n_evals = self.counter * self.lamb
        print(
            f"counter: {self.counter}, dt: {dt:.3f} n_evals {n_evals}, best {-self.best.y}, mean: {-mean.y}, sigma: {sigma} ",
        )

        if self.counter % self.test_gen == 0:
            self.best_test = problem.test(
                self.best.x,
                "rgb_array_list",
                False,
                self.data_folder,
                name=f"t-{self.counter}-best",
            )
            print("Test with best x (max):", self.best_test)
            self.mean_test = problem.test(
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
                np.std(-f)
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

    def update(self, problem, f, X, n, lambda_):
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


@dataclass
class DR1:
    n: int
    budget: int = 25_000
    mu: int = None
    lambda_: int = 10
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.n))).astype(int)
        self.mu = 1
    
    def __call__(self, problem: Objective):
        beta_scale = 1 / self.n
        beta = np.sqrt(beta_scale)
        zeta = np.array([5 / 7, 7 / 5])
        sigma = np.ones((self.n, 1)) * self.sigma0
        root_pi = np.sqrt(2 / np.pi)

        x_prime = init(self.n, problem.lb, problem.ub, self.initialization)

        state = State(self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)

        try:
            while self.budget > state.counter * self.lambda_:
                z = np.random.normal(size=(self.n, self.lambda_))
                zeta_i = np.random.choice(zeta, self.lambda_)
                X = x_prime + (zeta_i * (sigma * z)).T
                f = problem(X.T)

                idx, f = uch.update(problem, f, X.T, self.n, self.lambda_)
                idx_min = idx[0]

                if uch.should_update():
                    x_prime = X[idx_min, :].copy()
                    zeta_sel = np.exp(np.abs(z[:, idx_min]) - root_pi)
                    sigma *= (
                        np.power(zeta_i[idx_min], beta) * np.power(zeta_sel, beta_scale)
                    ).reshape(-1, 1)

                state.update(
                    problem,
                    Solution(f[idx_min], x_prime.copy()),
                    Solution(np.mean(f), np.mean(X, axis=0)),
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
    lambda_: int = 10
    sigma0: float = .5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.n))).astype(int)
        self.mu = 1
    
    def __call__(self, problem: Objective):
        beta_scale = 1 / self.n
        beta = np.sqrt(beta_scale)
        c = beta

        zeta = np.zeros((self.n, 1))
        sigma_local = np.ones((self.n, 1)) * self.sigma0
        sigma = self.sigma0

        c1 = np.sqrt(c / (2 - c))
        c2 = np.sqrt(self.n) * c1

        x_prime = init(self.n, problem.lb, problem.ub, self.initialization).reshape(-1, 1)

        state = State(self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)

        try:
            while self.budget > state.counter * self.lambda_:
                Z = np.random.normal(size=(self.n, self.lambda_))
                Y = sigma * sigma_local * Z
                X = x_prime + Y
                f = problem(X)
                idx, f = uch.update(problem, f, X, self.n, self.lambda_)
                idx_min = idx[0]

                if uch.should_update():
                    x_prime = (x_prime.T + (Y[:, idx_min])).T
                    z_prime = Z[:, idx_min].reshape(-1, 1)

                    zeta = ((1 - c) * zeta) + (c * z_prime)
                    sigma *= np.power(
                        np.exp((np.linalg.norm(zeta) / c2) - 1 + (1 / (5 * self.n))), beta
                    )
                    sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)

                state.update(
                    problem,
                    Solution(f[idx_min], x_prime.ravel().copy()),
                    Solution(np.mean(f), np.mean(X, axis=1)),
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

    def __post_init__(self):
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.n))).astype(int)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")


    def __call__(self, problem: Objective):
        weights = WeightedRecombination(self.mu, self.lambda_)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))
        mueff = 1 / np.sum(np.power(weights.w, 2))
        c_s = (mueff + 2) / (self.n + mueff + 5)
        d_s = 1 + c_s + 2 * max(0, np.sqrt((mueff - 1) / (self.n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * mueff)

        x = init(self.n, problem.lb, problem.ub, self.initialization).reshape(-1, 1)
        sigma = self.sigma0
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        try:
            while self.budget > state.counter * self.lambda_:
                Z = np.random.normal(0, 1, (self.n, self.lambda_))
                X = x + (sigma * Z)
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_)
                mu_best = idx[: self.mu]

                if uch.should_update():
                    z = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                    x = x + (sigma * z)
                    s = ((1 - c_s) * s) + (sqrt_s * z)
                    sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))

                state.update(
                    problem,
                    Solution(f[idx[0]], X[:, idx[0]].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x.copy()),
                    sigma,
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

    def __post_init__(self):
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.n))).astype(int)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")

    def __call__(self, problem: Objective):
        weights = WeightedRecombination(self.mu, self.lambda_)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))
        mueff = 1 / np.sum(np.power(weights.w, 2))
        c_s = (mueff + 2) / (self.n + mueff + 5)
        c_1 = 2 / (pow(self.n + 1.3, 2) + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + (1 / mueff)) / (pow(self.n + 2, 2) + mueff))
        d_s = 1 + c_s + 2 * max(0, np.sqrt((mueff - 1) / (self.n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * mueff)

        x = init(self.n, problem.lb, problem.ub, self.initialization).reshape(-1, 1)
        sigma = self.sigma0
        M = np.eye(self.n)
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        try:
            while self.budget > state.counter * self.lambda_:
                Z = np.random.normal(0, 1, (self.n, self.lambda_))
                D = M.dot(Z)
                X = x + (sigma * D)
                f = problem(X)

                idx, f = uch.update(problem, f, X, self.n, self.lambda_)
                mu_best = idx[: self.mu]

                if uch.should_update():
                    z = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                    d = np.sum(weights.w * D[:, mu_best], axis=1, keepdims=True)
                    x = x + (sigma * d)
                    s = ((1 - c_s) * s) + (sqrt_s * z)
                    M = (
                        ((1 - 0.5 * c_1 - 0.5 * c_mu) * M)
                        + ((0.5 * c_1) * M.dot(s).dot(s.T))
                        + ((0.5 * c_mu * weights.w_all) * D[:, idx]).dot(Z[:, idx].T)
                    )
                    sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))

                state.update(
                    problem,
                    Solution(f[idx[0]], X[:, idx[0]].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x.copy()),
                    sigma,
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
            while self.budget > state.counter * self.lambda_ * 2:
                delta = np.random.normal(size=(self.n, self.lambda_))

                neg = m - (self.eta * delta)
                pos = m + (self.eta * delta)

                neg_reward = -problem(neg)
                pos_reward = -problem(pos)
                
                best_rewards = np.maximum(neg_reward, pos_reward)
                idx = np.argsort(best_rewards)[::-1]

                sigma_rewards = np.r_[neg_reward, pos_reward].std()
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
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean
