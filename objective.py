import os
import io
from dataclasses import dataclass
from contextlib import redirect_stdout

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video

from network import Network, identity, MinMaxNormalizer, argmax


@dataclass
class Objective:
    n_episodes: int = 5
    n_timesteps: int = 100
    n_hidden: int = 8
    n_layers: int = 3
    net: Network = None
    n: int = None
    lb: np.ndarray = None
    ub: np.ndarray = None
    parallel: bool = True
    env_name: str = "LunarLander-v2"
    n_test_episodes: int = 10
    normalized: bool = True
    no_bias: bool = False
    eval_total_timesteps: bool = True
    store_video: bool = True
    aggregator: callable = np.mean

    def __post_init__(self):
        self.envs = gym.make_vec(self.env_name, num_envs=self.n_episodes)
        if self.env_name == "BipedalWalker-v3":
            self.action_size = 4
            self.n_actions = 4
            self.last_activation = identity
        else:
            self.action_size = self.envs.action_space[0].n
            self.last_activation = argmax
            self.n_actions = 1
        self.state_size = self.envs.observation_space.shape[1]

        if self.normalized:
            self.normalizer = MinMaxNormalizer(
                self.envs.observation_space.low[0], self.envs.observation_space.high[0]
            )
        else:
            self.normalizer = identity

        self.net = Network(
            self.state_size,
            self.action_size,
            self.n_hidden,
            self.n_layers,
            self.last_activation,
            not self.no_bias,
        )
        self.n = self.net.n_weights
        self.lb = -1 * np.ones(self.n)
        self.ub = 1 * np.ones(self.n)
        self.nets = []
        self.n_train_timesteps = 0
        self.n_train_episodes = 0

    def __call__(self, x):
        if self.parallel:
            return self.eval_parallel(x)
        return np.array([self.eval_sequential(xi) for xi in x.T])

    def eval_sequential(self, x):
        envs = gym.make_vec(self.env_name, num_envs=self.n_episodes)
        observations, _ = envs.reset()
        self.net.set_weights(x)

        data_over_time = np.empty((self.n_timesteps, 2, self.n_episodes))
        for t in range(self.n_timesteps):
            actions = self.net(self.normalizer(observations))
            observations, rewards, dones, trunc, *_ = envs.step(actions)
            data_over_time[t] = np.vstack(
                [self.fix_reward(rewards, dones), np.logical_or(dones, trunc)]
            )

        returns = []
        for i in range(self.n_episodes):
            returns.extend(self.calculate_returns(data_over_time[:, :, i]))
        return -self.aggregator(returns)

    def eval_parallel(self, x):
        n = x.shape[1]
        if n > len(self.nets):
            self.nets = [
                Network(
                    self.state_size,
                    self.action_size,
                    self.n_hidden,
                    self.n_layers,
                    self.last_activation,
                    not self.no_bias,
                )
                for _ in range(n)
            ]
            self.envs = gym.make_vec(self.env_name, num_envs=self.n_episodes * n)

        for net, w in zip(self.nets, x.T):
            net.set_weights(w)

        observations, *_ = self.envs.reset()
        n_total_episodes = action_shape = self.n_episodes * n
        if self.n_actions > 1:
            action_shape = (action_shape, self.n_actions)
        actions = np.ones(action_shape, dtype=int)
        data_over_time = np.empty((self.n_timesteps, 2, n_total_episodes))
        for t in range(self.n_timesteps):
            for i, net in enumerate(self.nets):
                idx = i * self.n_episodes
                actions[idx : idx + self.n_episodes] = net(
                    self.normalizer(observations[idx : idx + self.n_episodes, :])
                )
            observations, rewards, dones, trunc, *_ = self.envs.step(actions)
            data_over_time[t] = np.vstack(
                [self.fix_reward(rewards, dones), np.logical_or(dones, trunc)]
            )

        aggregated_returns = np.empty(n)
        for k, j in enumerate(range(0, n_total_episodes, self.n_episodes)):
            returns = []
            for i in range(self.n_episodes):
                returns.extend(self.calculate_returns(data_over_time[:, :, j + i]))
            aggregated_returns[k] = self.aggregator(returns)
        return -aggregated_returns

    def calculate_returns(self, Y):
        _, idx = np.unique(np.cumsum(Y[:, 1]) - Y[:, 1], return_index=True)
        returns_ = [x.sum() for x in np.split(Y[:, 0], idx)[1:]]
        if not self.eval_total_timesteps:
            returns_ = returns_[:1]
        
        # TODO: we can remove incomplete episodes from the last optionally
        return returns_

    def fix_reward(self, rewards, dones):
        if self.env_name == "CartPole-v1":
            return rewards - dones
        elif self.env_name in ("Acrobot-v1",):
            return rewards + (dones * self.n_timesteps)
        return rewards

    def test(self, x, render_mode=None, plot=False, data_folder=None, name=None):
        self.net.set_weights(x)

        returns = []
        try:
            for episode_index in range(self.n_test_episodes):
                env = gym.make(self.env_name, render_mode=render_mode)

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
                    reward = self.fix_reward(reward, terminated)
                    ret += reward
                    if render_mode == "human":
                        print(
                            f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r"
                        )
                    step_index += 1
                if render_mode == "human":
                    print()
                if (
                    self.store_video
                    and render_mode == "rgb_array_list"
                    and episode_index == 0
                ):
                    os.makedirs(f"{data_folder}/videos", exist_ok=True)
                    with redirect_stdout(io.StringIO()):
                        save_video(
                            env.render(),
                            f"{data_folder}/videos",
                            fps=env.metadata["render_fps"],
                            step_starting_index=0,
                            episode_index=0,
                            name_prefix=name,
                        )
                    os.makedirs(f"{data_folder}/policies", exist_ok=True)
                    np.save(f"{data_folder}/policies/{name}.pkl", x)
                    render_mode = None
                returns.append(ret)
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
            plt.savefig(f"{data_folder}/returns_{name}.png")
        return np.median(returns)

    def play(self, x, data_folder, name, plot=True):
        return self.test(x, "human", plot, data_folder, name)
