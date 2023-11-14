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
            self.best.y = problem.eval_sequential(self.best.x)

        toc = time.perf_counter()
        dt = toc - self.tic
        self.tic = toc

        if best_offspring.y < self.best.y:
            self.best = best_offspring
            self.time_since_best_update = 0 
        self.mean = mean

        print(
            f"counter: {self.counter}, dt: {dt:.3f} n_evals {problem.n_evals}, "
            f"best (train): {-self.best.y}, mean (train): {-mean.y}, sigma: {sigma} "
            f"best (test): {self.best_test}, mean (test): {self.mean_test}"
        )

        if self.counter % self.test_gen == 0:
            self.best_test, self.best_median, self.best_std = problem.test(
                self.best.x,
                "rgb_array_list",
                False,
                name=f"t-{self.counter}-best",
            )
            print("Test with best x (max):", self.best_test)
            self.mean_test, self.mean_median, self.mean_std = problem.test(
                self.mean.x,
                "rgb_array_list",
                False,
                name=f"t-{self.counter}-mean",
            )
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
        return max(32, np.floor(n / 2).astype(int))
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
    max_evals: int = 500
    max_observed: float = -np.inf
    min_observed: float =  np.inf

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

    def get_x_prime(self, problem, samples_per_trial: int = 10) -> np.ndarray:
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
            self.min_observed = max(self.min_observed, f.max())
            
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

        w = np.log(len(sample_values) + 0.5) - np.log(np.arange(1, len(sample_values) + 1))
        w = w / w.sum()
        idx = np.argsort(sample_values)
        x_prime = np.sum(w * samples[:, idx], axis=1, keepdims=True)
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

        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        x_prime = init.get_x_prime(problem)

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
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
        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        x_prime = init.get_x_prime(problem)

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
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

        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        x_prime = init.get_x_prime(problem)

        sigma = self.sigma0
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
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

        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        x_prime = init.get_x_prime(problem)
        sigma = self.sigma0
        M = np.eye(self.n)
        s = np.ones((self.n, 1))

        state = State(self.data_folder, self.test_gen, self.lambda_, self.revaluate_best_after)
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
    alpha: float = 0.02       # learning rate alpha
    lambda_: int = 16         # n offspring for each direction
    mu: int = 16              # best offspring
    sigma0: float = 0.03      # noise parameter
    initialization: str = "zero"

    def __post_init__(self):
        self.lambda_ = self.lambda_ or 16
        self.mu = self.mu or 16

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        m = init.get_x_prime(problem)


        state = State(self.data_folder, self.test_gen, self.lambda_ * 2)
        try:
            while self.budget > problem.n_evals:
                delta = np.random.normal(size=(self.n, self.lambda_))

                neg = m - (self.sigma0 * delta)
                pos = m + (self.sigma0 * delta)

                neg_reward = -problem(neg)
                pos_reward = -problem(pos)
                
                best_rewards = np.maximum(neg_reward, pos_reward)
                idx = np.argsort(best_rewards)[::-1]

                f = np.r_[neg_reward, pos_reward]
                sigma_rewards = f.std() + 1e-12
                weight = self.alpha / (self.lambda_ * sigma_rewards)

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
        self.lambda_ = self.lambda_ or 16
        self.mu = 1

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        m = init.get_x_prime(problem)

        state = State(self.data_folder, self.test_gen, self.lambda_)
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

    def __post_init__(self):
        self.lambda_ = self.lambda_ or 16
        self.mu = 1

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=self.budget // 20)
        m = init.get_x_prime(problem)

        pc = np.zeros((self.n, 1))
        ps = np.zeros((self.n, 1))
        B = np.eye(self.n)
        C = np.eye(self.n)
        D = np.ones((self.n, 1))

        alpha = beta = 4 / (self.n + 4)
        gamma = 2 / pow(self.n + np.sqrt(2), 2)
        chi = 2 * self.n * (1 + (1 / beta))

        state = State(self.data_folder, self.test_gen, self.lambda_)
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
                    D, B = np.linalg.eigh(C)
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

