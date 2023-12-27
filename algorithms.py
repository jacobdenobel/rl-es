import warnings
import time
import os
from typing import Any
from dataclasses import dataclass, field


import numpy as np
from scipy.stats import qmc

from objective import Objective


SIGMA_MAX = 1e3

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
    def __init__(self, name, data_folder, test_gen, lamb, revaluate_best_after: int = 1):
        self.counter = 0
        self.best = Solution()
        self.mean = Solution()
        self.best_test = float("inf")
        self.mean_test = float("inf")
        self.tic = time.perf_counter()
        self.data_folder = data_folder
        self.logger = Logger(data_folder)
        self.test_gen: int = test_gen
        self.name = name
        self.lamb: int = lamb
        self.mean_test = None
        self.best_test = None
        self.best_median = None
        self.best_std = None
        self.mean_median = None
        self.mean_std = None
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

        if self.revaluate_best_every is not None and self.revaluate_best_every < self.time_since_best_update:
            self.time_since_best_update = 0
            best_old = self.best.y
            self.best.y = problem.eval_sequential(self.best.x, False)
            # got_test, got_mean, got_std = problem.test_policy(
            #     self.best.x,
            #     name=f"t-{self.counter}-best",
            #     save=False
            # )
            # print("reevaluating best, expected: ", -best_old, -self.best.y, -got_test, -got_mean, got_std)
            

        toc = time.perf_counter()
        dt = toc - self.tic
        self.tic = toc

        if best_offspring.y < self.best.y:
            self.best = best_offspring
            self.time_since_best_update = 0 
        self.mean = mean

        print(
            f"{self.name}, counter: {self.counter}, dt: {dt:.3f} n_evals {problem.n_evals}, n_episodes: {problem.n_train_episodes} "
            f"best (train): {-self.best.y}, mean (train): {-mean.y}, sigma: {sigma} "
            f"best (test): {self.best_test}, mean (test): {self.mean_test}"
        )

        if self.counter % self.test_gen == 0:
            self.best_test, self.best_median, self.best_std = problem.test_policy(
                self.best.x,
                name=f"t-{self.counter}-best",
                save=True
            )
            self.best_test, self.best_median = -self.best_test, -self.best_median
            print("Test with best x (max):", self.best_test)
            self.mean_test, self.mean_median, self.mean_std = problem.test_policy(
                self.mean.x,
                name=f"t-{self.counter}-mean",
                save=True
            )
            self.mean_test, self.mean_median = -self.mean_test, -self.mean_median
            print("Test with mean x (max):", self.mean_test)

        self.logger.write(
            (
                self.counter,
                dt,
                problem.n_evals,
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
class Weights:
    mu: int
    lambda_: int
    n: int
    method: str = "log"

    def __post_init__(self):
        self.set_weights()
        self.normalize_weights()

    def set_weights(self):
        if self.method == "log":
            self.wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        elif self.method == "linear":
            self.wi_raw = np.arange(1, self.mu + 1)[::-1]
        elif self.method == "equal":
            self.wi_raw = np.ones(self.mu)

    def normalize_weights(self):
        self.w = self.wi_raw / np.sum(self.wi_raw) 
        self.w_all = np.r_[self.w, -self.w[::-1]]

    @property
    def mueff(self):
        return 1 / np.sum(np.power(self.w, 2))

    @property
    def c_s(self):
        return (self.mueff + 2) / (self.n + self.mueff + 5)
    
    @property
    def d_s(self):
        return 1 + self.c_s + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1)
    
    @property
    def sqrt_s(self):
        return np.sqrt(self.c_s * (2 - self.c_s) * self.mueff)
    

def init_lambda(n, method="n/2"):
    """
        range:      2*mu < lambda < 2*n + 10 
        default:    4 + floor(3 * ln(n))     
    """

    if method == "default":
        return (4 + np.floor(3 * np.log(n))).astype(int) 
    
    elif method == "n/2":
        return min(128, max(32, np.floor(n / 2).astype(int)))
    elif isinstance(method, int):
        return method
    else:
        raise ValueError()
    


@dataclass
class Initializer:
    n: int 
    lb: float = -0.1
    ub: float =  0.1
    method: str = "lhs"
    fallback: str = "zero"
    n_evals: int = 0
    max_evals: int = 32 *5
    max_observed: float = -np.inf
    min_observed: float =  np.inf
    aggregate: bool = False

    def __post_init__(self):
        self.sampler = qmc.LatinHypercube(self.n)

    def static_init(self, method):
        if method == "zero":
            return np.zeros((self.n, 1))
        elif method == "uniform":
            return np.random.uniform(self.lb, self.ub, size=(self.n, 1))
        elif method == "gauss":
            return np.random.normal(size=(self.n, 1))
        raise ValueError()

    def get_x_prime(self, problem, samples_per_trial: int = 128) -> np.ndarray:
        if self.method != "lhs":
            return self.static_init(self.method)

        samples = None
        sample_values = np.array([])
        f = np.array([0])
        while self.n_evals < self.max_evals:
            X = qmc.scale(self.sampler.random(samples_per_trial), self.lb, self.ub).T
            f = problem(X)
            self.n_evals += samples_per_trial
            self.max_observed = max(self.max_observed, f.max())
            self.min_observed = min(self.min_observed, f.min())
            
            if f.std() > 0:
                idx = f != self.max_observed
                if samples is None:
                    samples = X[:, idx]
                else:
                    samples = np.c_[samples, X[:, idx]]
                sample_values = np.r_[sample_values, f[idx]]
        
        if not any(sample_values):
            warnings.warn(f"DOE did not find any variation after max_evals={self.max_evals}"
                          f", using fallback {self.fallback} intialization.")
            return self.static_init(self.fallback)

        idx = np.argsort(sample_values)
        if self.aggregate:
            w = np.log(len(sample_values) + 0.5) - np.log(np.arange(1, len(sample_values) + 1))
            w = w / w.sum()
            x_prime = np.sum(w * samples[:, idx], axis=1, keepdims=True)
        else:
            x_prime = samples[:, idx[0]].reshape(-1, 1)

        print("lhs:", problem.n_evals, self.min_observed, self.max_observed)
        return x_prime


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

        init = Initializer(self.n, method=self.initialization, max_evals=500)
        x_prime = init.get_x_prime(problem)

        state = State("DR1", self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        weights = Weights(self.mu, self.lambda_, self.n)

        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        
        try:
            while self.budget > problem.n_evals:
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

                if uch.should_update():
                    z_prime = np.sum(Z[:, mu_best] * weights.w, axis=1, keepdims=True) * np.sqrt(weights.mueff)
                    zeta_w = np.sum(zeta_i[:, mu_best] * weights.w)
                    zeta_sel = np.exp(np.abs(z_prime) - root_pi)
                    sigma *= (
                        np.power(zeta_w, beta) * np.power(zeta_sel, beta_scale)
                    ).reshape(-1, 1)
                    sigma = sigma.clip(0, SIGMA_MAX)

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
        c3 = 1 / (5 * self.n)

        weights = Weights(self.mu, self.lambda_, self.n)
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        x_prime = init.get_x_prime(problem)

        state = State("DR2", self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > problem.n_evals:
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
                    sigma = min(sigma * np.power(
                        np.exp((np.linalg.norm(zeta) / c2) - 1 + c3), beta
                    ), SIGMA_MAX)
                    sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)
                    sigma_local = sigma_local.clip(0, SIGMA_MAX)

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
        self.mu = self.mu or self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")


    def __call__(self, problem: Objective):
        weights = Weights(self.mu, self.lambda_, self.n)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))

        init = Initializer(self.n, method=self.initialization, max_evals=500)
        x_prime = init.get_x_prime(problem)

        sigma = self.sigma0
        s = np.ones((self.n, 1))

        state = State("CSA", self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > problem.n_evals:
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
                s = ((1 - weights.c_s) * s) + (weights.sqrt_s * z_prime)

                if uch.should_update():
                    sigma = sigma * np.exp(weights.c_s / weights.d_s * (np.linalg.norm(s) / echi - 1))

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
    scale_by_std: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.mu or self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")

    def __call__(self, problem: Objective):
        weights = Weights(self.mu, self.lambda_, self.n)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))
        c_s = (weights.mueff + 2) / (self.n + weights.mueff + 5)
        c_1 = 2 / (pow(self.n + 1.3, 2) + weights.mueff)
        c_mu = min(1 - c_1, 2 * (weights.mueff - 2 + (1 / weights.mueff)) / (pow(self.n + 2, 2) + weights.mueff))
        d_s = 1 + c_s + 2 * max(0, np.sqrt((weights.mueff - 1) / (self.n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * weights.mueff)

        init = Initializer(self.n, method=self.initialization, max_evals=500)
        x_prime = init.get_x_prime(problem)
        sigma = self.sigma0
        M = np.eye(self.n)
        s = np.ones((self.n, 1))

        state = State("MA-ES", self.data_folder, self.test_gen, self.lambda_)
        uch = UncertaintyHandling(self.uncertainty_handling)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while self.budget > problem.n_evals:
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
                
                if self.scale_by_std:
                    d_prime = (1 / f.std()) * d_prime

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
class ARSSetting:
    alpha: float
    sigma: float
    lambda0: int
    mu: int = None 

    def __post_init__(self):
        self.mu = self.mu or self.lambda0

ARS_OPTIMAL_PARAMETERS = {
    "Swimmer-v4": ARSSetting(0.02, 0.01, 1),
    # "Hopper-v4": ARSSetting(0.02, 0.02, 4),
    "Hopper-v4": ARSSetting(0.01, 0.025, 8, 4),
    # "HalfCheetah-v4": ARSSetting(0.02, 0.03, 8),
    "HalfCheetah-v4": ARSSetting(0.02, 0.03, 32, 4),
    # "Walker2d-v4": ARSSetting(0.025, 0.01, 60),
    "Walker2d-v4": ARSSetting(0.03, 0.025, 40, 30),
    # "Ant-v4": ARSSetting(0.01, 0.025, 40),
    "Ant-v4": ARSSetting(0.015, 0.025, 60, 20),
    "Humanoid-v4": ARSSetting(0.02, 0.0075, 230),
}

@dataclass
class ARS:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    alpha: float = 0.02       # learning rate alpha
    lambda_: int = 16         # n offspring for each direction
    mu: int = 16              # best offspring
    sigma0: float = 0.03      # noise parameter
    initialization: str = "zero"

    def __post_init__(self):
        self.lambda_ = self.lambda_ or int(init_lambda(self.n) / 2)
        self.mu = self.mu or self.lambda_
        self.sigma0 = self.sigma0 or 0.03

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        state = State("ARS", self.data_folder, self.test_gen, self.lambda_ * 2)
        try:
            while self.budget > problem.n_evals:
                delta = np.random.normal(size=(self.n, self.lambda_))

                neg = m - (self.sigma0 * delta)
                pos = m + (self.sigma0 * delta)

                neg_reward = -problem(neg)
                pos_reward = -problem(pos)
                
                best_rewards = np.maximum(neg_reward, pos_reward)
                idx = np.argsort(best_rewards)[::-1]
                mu_best = idx[: self.mu]

                rewards = np.c_[pos_reward[mu_best], neg_reward[mu_best]]
                sigma_rewards = rewards.std() + 1e-12
                weight = self.alpha / (self.mu * sigma_rewards)
                delta_rewards =  rewards[:, 0] -  rewards[:, 1]
                m += (weight * (delta_rewards * delta[:, mu_best]).sum(axis=1, keepdims=True))

                best_idx = mu_best[0]
                if neg_reward[best_idx] > pos_reward[best_idx]:
                    best = neg[:, best_idx]
                else:
                    best = pos[:, best_idx]

                state.update(
                    problem,
                    Solution(-best_rewards[best_idx],  best.copy()),
                    Solution(-np.mean(best_rewards), m.copy()),
                    self.sigma0,
                    np.r_[pos_reward, neg_reward]
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class EGS:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = None             
    kappa: float = 2.0    
    initialization: str = "zero"

    def __post_init__(self):
        self.lambda_ = self.lambda_ or int(init_lambda(self.n) / 2)
        self.mu = 1

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        state = State("EGS", self.data_folder, self.test_gen, self.lambda_)
        sigma = self.sigma0

        try:
            while self.budget > problem.n_evals:
                Z = np.random.normal(size=(self.n, self.lambda_))
                y_pos = m + sigma * Z
                y_neg = m - sigma * Z
                f_pos = problem(y_pos)
                f_neg = problem(y_neg)

                z_avg  = np.sum((f_neg - f_pos) * Z, axis=1, keepdims=True)
                z_prog = (np.sqrt(self.n) / self.kappa) * (z_avg / np.linalg.norm(z_avg))

                m = m + sigma * z_prog
                
                f = np.r_[f_pos, f_neg]
                X = np.hstack([y_pos, y_neg])
                best_idx = np.argmin(f)

                state.update(
                    problem,
                    Solution(f[best_idx],  X[:, best_idx].copy()),
                    Solution(np.mean(f), m.copy()),
                    sigma,
                    f
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean

@dataclass
class CMA_EGS:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = None             
    kappa: float = 2.0    
    initialization: str = "zero"
    sep: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or int(init_lambda(self.n) / 2)
        self.mu = 1

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        pc = np.zeros((self.n, 1))
        ps = np.zeros((self.n, 1))
        B = np.eye(self.n)
        C = np.eye(self.n)
        D = np.ones((self.n, 1))

        alpha = beta = 4 / (self.n + 4)
        gamma = 2 / pow(self.n + np.sqrt(2), 2)
        chi = 2 * self.n * (1 + (1 / beta))

        state = State("CMA-EGS", self.data_folder, self.test_gen, self.lambda_)
        sigma = self.sigma0
        try:
            while self.budget > problem.n_evals:
                Z = np.random.normal(size=(self.n, self.lambda_))
                Y = np.dot(B, D * Z)
                y_pos = m + (sigma * Y)
                y_neg = m - (sigma * Y)
                f_pos = problem(y_pos)
                f_neg = problem(y_neg)

                z_avg  = np.sum((f_neg - f_pos) * Z, axis=1, keepdims=True)
                z_prog = (np.sqrt(self.n) / self.kappa) * (z_avg / np.linalg.norm(z_avg))

                bd_z_prog = np.dot(B, D * z_prog)
                b_z_prog = np.dot(B, z_prog)
                m = m + sigma * bd_z_prog

                pc = ((1 - alpha) * pc) + (self.kappa * np.sqrt(alpha * (2 - alpha)) * bd_z_prog)
                ps = ((1 - beta) * ps) + (self.kappa * np.sqrt(beta * (2 - beta)) * b_z_prog)
                C = ((1 - gamma) * C) + (gamma * (pc * pc.T))

                sigma *= np.exp((pow(np.linalg.norm(ps), 2) - self.n) / chi)
                
                if np.isinf(C).any() or np.isnan(C).any() or (not 1e-16 < sigma < 1e6):
                    sigma = self.sigma0
                    pc = np.zeros((self.n, 1))
                    ps = np.zeros((self.n, 1))
                    C = np.eye(self.n)
                    B = np.eye(self.n)
                    D = np.ones((self.n, 1))
                elif self.n < 100 or state.counter % (self.n // 10) == 0:
                    C = np.triu(C) + np.triu(C, 1).T
                    if not self.sep:
                        D, B = np.linalg.eigh(C)
                    else:
                        D = np.diag(C)
                    D = np.sqrt(D).reshape(-1, 1)

                f = np.r_[f_pos, f_neg]
                X = np.hstack([y_pos, y_neg])
                best_idx = np.argmin(f)

                state.update(
                    problem,
                    Solution(f[best_idx],  X[:, best_idx].copy()),
                    Solution(np.mean(f), m.copy()),
                    sigma,
                    f
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class CSA_EGS:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = None             
    kappa: float = 2.0    
    initialization: str = "zero"

    def __post_init__(self):
        self.lambda_ = self.lambda_ or int(init_lambda(self.n) / 2)
        self.mu = 1

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        ps = np.zeros((self.n, 1))

        beta = 4 / (self.n + 4)
        chi = 2 * self.n * (1 + (1 / beta))

        state = State("CSA-EGS", self.data_folder, self.test_gen, self.lambda_)
        sigma = self.sigma0
        try:
            while self.budget > problem.n_evals:
                Z = np.random.normal(size=(self.n, self.lambda_))
                y_pos = m + (sigma * Z)
                y_neg = m - (sigma * Z)
                f_pos = problem(y_pos)
                f_neg = problem(y_neg)

                z_avg  = np.sum((f_neg - f_pos) * Z, axis=1, keepdims=True)
                z_norm = z_avg / np.linalg.norm(z_avg)
                z_prog = (np.sqrt(self.n) / self.kappa) * z_norm
                m = m + (sigma * z_prog)

                ps = ((1 - beta) * ps) + (self.kappa * np.sqrt(beta * (2 - beta)) * z_prog)
                sigma *= np.exp((pow(np.linalg.norm(ps), 2) - self.n) / chi)

                f = np.r_[f_pos, f_neg]
                X = np.hstack([y_pos, y_neg])
                best_idx = np.argmin(f)

                state.update(
                    problem,
                    Solution(f[best_idx],  X[:, best_idx].copy()),
                    Solution(np.mean(f), m.copy()),
                    sigma,
                    f
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class CMAES:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = None             
    initialization: str = "zero"
    sep: bool = False
    # tpa better with ineffective axis
    tpa: bool = False
    active: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.sep:
            # self.tpa = True
            self.lambda_ = min(4, self.lambda_)

        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2            
        
        print(self.n, self.lambda_, self.mu, self.sigma0)
        

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        w = np.log((self.lambda_ + 1) / 2) - np.log(np.arange(1, self.lambda_ + 1))
        w = w[: self.mu]
        mueff = w.sum() ** 2 / (w**2).sum()
        w = w / w.sum()
        w_active = np.r_[w, -1 * w[::-1]]

        # Learning rates
        n = self.n
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
            
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / 2)
        if self.sep:
            cmu *= ((n + 2) / 3)
        cc = (4 + (mueff / n)) / (n + 4 + (2 * mueff / n))

        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1.0 + (2.0 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs)
        chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # dynamic parameters
        dm = np.zeros((n, 1))
        pc = np.zeros((n, 1))
        ps = np.zeros((n, 1))
        B = np.eye(n)
        C = np.eye(n)
        D = np.ones((n, 1))
        invC = np.eye(n)

        state = State(f"{'sep-' if self.sep else ''}{'a' if self.active else ''}CMA-ES", self.data_folder, self.test_gen, self.lambda_)
        sigma = self.sigma0

        if self.tpa:
            cs = 0.3
        s = 0
        hs = True
        z_exponent = 0.5
        damp = n ** 0.5 
        
        try:
            while self.budget > problem.n_evals:
                active_tpa = self.tpa and state.counter != 0
                n_offspring = self.lambda_ - (2 * active_tpa)
                Z = np.random.normal(0, 1, (n, n_offspring))
                Y = np.dot(B, D * Z)
                if active_tpa:
                    Y = np.c_[-dm, dm, Y]

                X = m + (sigma * Y)
                f = np.array(problem(X))

                # select
                fidx = np.argsort(f)
                mu_best = fidx[: self.mu]

                # recombine
                m_old = m.copy()
                m = m_old + (1 * ((X[:, mu_best] - m_old) @ w).reshape(-1, 1))

                # adapt
                dm = (m - m_old) / sigma
                ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff) * invC @ dm)
                hs = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - np.power(1 - cs, 2 * (problem.n_evals / self.lambda_)))
                ) < (1.4 + (2 / (n + 1))) * chiN

                if not self.tpa:
                    sigma *= np.exp((cs / damps) * ((np.linalg.norm(ps) / chiN) - 1))
                elif state.counter != 0:
                    z = (fidx[0] - fidx[1]) / (self.lambda_ - 1)
                    s = (1 - cs) * s + cs * np.sign(z) * pow(np.abs(z), z_exponent)
                    sigma *= np.exp(s / damp)

                dhs = (1 - hs) * cc * (2 - cc)
                pc = (1 - cc) * pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm


                old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * C
                rank_one = c1 * pc * pc.T
                if self.active:
                    rank_mu = cmu * (w_active * Y[:, fidx] @ Y[:, fidx].T)
                else:
                    rank_mu = cmu * (w * Y[:, mu_best] @ Y[:, mu_best].T)
                
                C = old_C + rank_one + rank_mu

                if np.isinf(C).any() or np.isnan(C).any() or (not 1e-16 < sigma < 1e6):
                    sigma = self.sigma0
                    pc = np.zeros((n, 1))
                    ps = np.zeros((n, 1))
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones((n, 1))
                    invC = np.eye(n)
                else:
                    C = np.triu(C) + np.triu(C, 1).T
                    if not self.sep:
                        D, B = np.linalg.eigh(C)
                    else:
                        # D = np.diag(C)
                        pass


                D = np.sqrt(D).reshape(-1, 1)
                invC = np.dot(B, D ** -1 * B.T)
                
                best_idx = mu_best[0]
                state.update(
                    problem,
                    Solution(f[best_idx],  X[:, best_idx].copy()),
                    Solution(np.mean(f), m.copy()),
                    sigma,
                    f
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean




@dataclass
class SPSA:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    mu: int = 1
    lambda_: int = 1
    sigma0: float = 0.02     
    initialization: str = "zero"

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        state = State("SPSA", self.data_folder, self.test_gen, 1)
        import spsa
        
        iterator = spsa.iterator.minimize(problem, m)
        try:
            while self.budget > problem.n_evals:
                data = next(iterator)
                x_best = data['x_best']
                y_best, *_ = data['y_best']
                x = data['x']
                y, *_ = data['y']

                state.update(
                    problem,
                    Solution(y_best,  x_best.copy()),
                    Solution(y,  x.copy()),
                    self.sigma0,
                    np.array([y])
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean




import math

from typing import Any
from typing import cast
from typing import Optional

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class _SepCMA:
    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = np.array(
            [math.log(mu + 1) - math.log(i + 1) for i in range(mu)]
        )
        weights = weights_prime / sum(weights_prime)
        mu_eff = 1 / sum(weights**2)

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu_full = 2 / mu_eff / ((n_dim + np.sqrt(2)) ** 2) + (1 - 1 / mu_eff) * min(
            1, (2 * mu_eff - 1) / ((n_dim + 2) ** 2 + mu_eff)
        )
        cmu = (n_dim + 2) / 3 * cmu_full

        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 3)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update
        cc = 4 / (n_dim + 4)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim**2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean
        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._C: np.ndarray = np.ones(n_dim)

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def __getstate__(self) -> dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _eigen_decomposition(self) -> np.ndarray:
        if self._D is not None:
            return self._D
        self._D = np.sqrt(np.where(self._C < 0, _EPS, self._C))
        return self._D

    def _sample_solution(self) -> np.ndarray:
        D = self._eigen_decomposition()
        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        y = D * z  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points, for k=1, ..., popsize
        D = self._eigen_decomposition()
        self._D = None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (y_w / D)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = self._pc**2
        rank_mu = np.sum(
            np.array([w * (y**2) for w, y in zip(self._weights, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )
        
    def should_stop(self) -> bool:
        D = self._eigen_decomposition()

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * self._C < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(self._C))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(
            self._mean == self._mean + (0.1 * self._sigma * D[i] * np.ones(self._n_dim))
        ):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False


@dataclass
class SepCMA:
    n: int
    budget: int = 25_000
    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = None             
    initialization: str = "zero"

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2            
        
        print(self.n, self.lambda_, self.mu, self.sigma0)

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        state = State("sep-CMA", self.data_folder, self.test_gen, self.lambda_)

        cma = _SepCMA(m.ravel(), self.sigma0, population_size = self.lambda_, n_max_resampling=0)

        try:
            while self.budget > problem.n_evals:
                X = np.vstack([cma.ask() for _ in range(cma.population_size)])
                f = problem(X.T)
                solutions = list(zip(X, f))
                cma.tell(solutions)

                best_idx = np.argmin(f)

                state.update(
                    problem,
                    Solution(f[best_idx],  X[:, best_idx].copy()),
                    Solution(np.mean(f), cma._mean.copy()),
                    cma._sigma,
                    f
                )
         
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean
