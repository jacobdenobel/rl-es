import os
import io
from dataclasses import dataclass
from contextlib import redirect_stdout

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video

from network import Network, identity, MinMaxNormalizer, argmax
from skimage.measure import block_reduce


@dataclass
class EnvSetting:
    name: str
    budget: int
    reward_shaping: callable = identity
    action_size: int = None
    state_size: int = None
    max_episode_steps: int = None
    reward_threshold: float = None
    last_activation: callable = identity
    is_discrete: bool = True

    def __post_init__(self):
        env = gym.make(self.name)
        self.state_size = np.prod(env.observation_space.shape)
        self.is_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.max_episode_steps = env.spec.max_episode_steps
        self.reward_threshold = env.spec.reward_threshold
        if self.is_discrete:
            self.action_size = env.action_space.n
            self.last_activation = argmax
        else:
            self.action_size = env.action_space.shape[0]


ENVIRONMENTS = {
    "CartPole-v1": EnvSetting("CartPole-v1", 1000),
    "Acrobot-v1": EnvSetting("Acrobot-v1", 1_000),
    "MountainCar-v0": EnvSetting("MountainCar-v0", 5_000),
    "LunarLander-v2": EnvSetting("LunarLander-v2", 5_000),
    "BipedalWalker-v3": EnvSetting(
        "BipedalWalker-v3", 20_000, lambda x: np.clip(x, -1, 1)
    ),
    # "Swimmer-v4": EnvSetting("Swimmer-v4", 10_000),
    "Hopper-v4": EnvSetting("Hopper-v4", 20_000, lambda x: x - 1),
    "HalfCheetah-v4": EnvSetting("HalfCheetah-v4", 20_000),
    "Walker2d-v4": EnvSetting("Walker2d-v4", 100_000, lambda x: x - 1),
    "Ant-v4": EnvSetting("Ant-v4", 100_000, lambda x: x - 1),
    "Humanoid-v4": EnvSetting("Humanoid-v4", 500_000, lambda x: x - 5),
}



def rgb_to_gray_flat(observations):
    gray = np.dot(observations[..., :3], [0.2989, 0.5870, 0.1140])
    reduced = block_reduce(gray, (1, 6, 6), np.max)
    return reduced.reshape(len(reduced), -1)


class Normalizer:
    def __init__(self, nb_inputs):
        self.mean = np.zeros(nb_inputs)
        self.std = np.ones(nb_inputs)

    def observe(self, _):
        pass

    def __call__(self, x):
        return (x - self.mean) / self.std


class Standardizer(Normalizer):
    def __init__(self, nb_inputs):
        super().__init__(nb_inputs)
        self.k = 0
        self.s = np.zeros(nb_inputs)

    def observe(self, X):
        for x in X:
            self.k += 1
            delta = x - self.mean
            self.mean += delta / self.k
            self.s += delta * (x - self.mean)

        if self.k > 1:
            self.std = np.sqrt(self.s / (self.k - 1)).clip(1e-8)


@dataclass
class Objective:
    setting: EnvSetting
    n_episodes: int = 5
    n_timesteps: int = 100
    n_hidden: int = 8
    n_layers: int = 3
    net: Network = None
    n: int = None
    parallel: bool = True
    n_test_episodes: int = 10
    normalized: bool = True
    no_bias: bool = False
    eval_total_timesteps: bool = True
    store_video: bool = True
    aggregator: callable = np.mean
    n_train_timesteps: int = 0
    n_train_episodes: int = 0
    n_evals: int = 0
    data_folder: str = None

    def __post_init__(self):
        if self.normalized:
            self.normalizer = Standardizer(self.setting.state_size)
        else:
            self.normalizer = Normalizer(self.setting.state_size)

        self.net = Network(
            self.setting.state_size,
            self.setting.action_size,
            self.n_hidden,
            self.n_layers,
            self.setting.last_activation,
            not self.no_bias,
        )
        self.n = self.net.n_weights
        self.nets = []
        self.train_writer = open(
            os.path.join(self.data_folder, "train_evals.csv"), "a+"
        )
        self.test_writer = open(os.path.join(self.data_folder, "test_evals.csv"), "a+")
        header = ", ".join([f"w{i}" for i in range(self.n)])
        header = f"evals, fitness, {header}\n"
        self.train_writer.write(header)
        self.test_writer.write(header)
        self.n_test_evals = 0

    def __call__(self, x):
        if self.parallel:
            f = self.eval_parallel(x)
        else:
            f = np.array([self.eval_sequential(xi) for xi in x.T])

        for y, xi in zip(f, x.T):
            self.n_evals += 1
            self.train_writer.write(f"{self.n_evals}, {y}, {', '.join(map(str, xi))}\n")
        return f

    def eval_sequential(self, x):
        envs = gym.make_vec(self.setting.name, num_envs=self.n_episodes)
        observations, _ = envs.reset()
        self.net.set_weights(x)

        data_over_time = np.empty((self.n_timesteps, 2, self.n_episodes))
        for t in range(self.n_timesteps):
            actions = self.net(self.normalizer(observations))
            self.normalizer.observe(observations)
            observations, rewards, dones, trunc, *_ = envs.step(actions)
            rewards = self.setting.reward_shaping(rewards)
            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])

        returns = []
        for i in range(self.n_episodes):
            returns.extend(self.calculate_returns(data_over_time[:, :, i]))

        self.n_train_timesteps += self.n_timesteps * self.n_episodes
        self.n_train_episodes += self.n_episodes
        return -self.aggregator(returns)

    def eval_parallel(self, x):
        n = x.shape[1]
        if n > len(self.nets):
            self.nets = [
                Network(
                    self.setting.state_size,
                    self.setting.action_size,
                    self.n_hidden,
                    self.n_layers,
                    self.setting.last_activation,
                    not self.no_bias,
                )
                for _ in range(n)
            ]
            self.envs = gym.make_vec(self.setting.name, num_envs=self.n_episodes * n)

        for net, w in zip(self.nets, x.T):
            net.set_weights(w)

        observations, *_ = self.envs.reset()

        n_total_episodes = action_shape = self.n_episodes * n
        
        if not self.setting.is_discrete:
            action_shape = (action_shape, self.setting.action_size)

        actions = np.ones(action_shape, dtype=int)
        data_over_time = np.empty((self.n_timesteps, 2, n_total_episodes))
        for t in range(self.n_timesteps):
            for i, net in enumerate(self.nets):
                idx = i * self.n_episodes
                obs = observations[idx : idx + self.n_episodes, :]
                actions[idx : idx + self.n_episodes] = net(self.normalizer(obs))
                self.normalizer.observe(obs)

            observations, rewards, dones, trunc, *_ = self.envs.step(actions)
            rewards = self.setting.reward_shaping(rewards)
            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])

        aggregated_returns = np.empty(n)
        for k, j in enumerate(range(0, n_total_episodes, self.n_episodes)):
            returns = []
            for i in range(self.n_episodes):
                returns.extend(self.calculate_returns(data_over_time[:, :, j + i]))
            aggregated_returns[k] = self.aggregator(returns)

        if self.eval_total_timesteps:
            self.n_train_timesteps += self.n_timesteps * self.n_episodes * n
            self.n_train_episodes += self.n_episodes * n
        else:
            raise NotImplementedError()

        return -aggregated_returns

    def calculate_returns(self, Y):
        _, idx = np.unique(np.cumsum(Y[:, 1]) - Y[:, 1], return_index=True)
        episodes = np.split(Y[:, 0], idx)[1:]
        if len(episodes) > 1:
            episodes = episodes[:-1]

        returns_ = [x.sum() for x in episodes]
        if not self.eval_total_timesteps:
            returns_ = returns_[:1]

        # TODO: we can remove incomplete episodes from the last optionally
        return returns_

    def test(self, x, render_mode=None, plot=False, name=None):
        self.net.set_weights(x)

        returns = []
        try:
            for episode_index in range(self.n_test_episodes):
                env = gym.make(self.setting.name, render_mode=render_mode)

                observation, *_ = env.reset()
                if render_mode == "human":
                    env.render()
                done = False
                ret = 0
                step_index = 0
                while not done:
                    action, *_ = self.net(self.normalizer(observation.reshape(1, -1)))
                    observation, reward, terminated, truncated, *_ = env.step(action)
                    done = terminated or truncated
                    ret += reward
                    if render_mode == "human":
                        print(
                            f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r"
                        )
                    step_index += 1
                if render_mode == "human":
                    print()
                if render_mode == "rgb_array_list" and episode_index == 0:
                    if self.store_video:
                        os.makedirs(f"{self.data_folder}/videos", exist_ok=True)
                        with redirect_stdout(io.StringIO()):
                            save_video(
                                env.render(),
                                f"{self.data_folder}/videos",
                                fps=env.metadata["render_fps"],
                                step_starting_index=0,
                                episode_index=0,
                                name_prefix=name,
                            )
                    os.makedirs(f"{self.data_folder}/policies", exist_ok=True)
                    np.save(f"{self.data_folder}/policies/{name}", x)
                    np.save(f"{self.data_folder}/policies/{name}-norm-std", self.normalizer.std)
                    np.save(f"{self.data_folder}/policies/{name}-norm-mean", self.normalizer.mean)
                    render_mode = None
                returns.append(ret)
                self.n_test_evals += 1
                self.test_writer.write(
                    f"{self.n_test_evals}, {ret}, {', '.join(map(str, x.ravel()))}\n"
                )

        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        if plot:
            plt.figure()
            plt.hist(returns)
            plt.grid()
            plt.xlabel("returns")
            plt.ylabel("freq")
            plt.savefig(f"{self.data_folder}/returns_{name}.png")
        return np.mean(returns), np.median(returns), np.std(returns)

    def play(self, x, name, plot=True):
        return self.test(x, "human", plot, name)
