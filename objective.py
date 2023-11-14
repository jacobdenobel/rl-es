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

def rgb_to_gray_flat(observations):
    gray = np.dot(observations[..., :3], [0.2989, 0.5870, 0.1140])
    reduced = block_reduce(gray, (1, 6, 6), np.max)
    return reduced.reshape(len(reduced), -1)


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
    n_train_timesteps: int = 0
    n_train_episodes: int = 0
    n_evals: int = 0
    data_folder: str =  None

    def __post_init__(self):
        self.envs = gym.make_vec(self.env_name, num_envs=self.n_episodes)
        self.obs_mapper = identity
        self.state_size = self.envs.observation_space.shape[1]

        if self.env_name == "CarRacing-v2":
            self.action_size = 3
            self.n_actions = 3
            self.last_activation = identity
            self.obs_mapper = rgb_to_gray_flat
            self.state_size = 256
        elif self.env_name == "BipedalWalker-v3":
            self.action_size = 4
            self.n_actions = 4
            self.last_activation = identity
        elif self.env_name == "Hopper-v4":
            self.action_size = 3
            self.n_actions = 3
            self.last_activation = identity
        else:
            self.action_size = self.envs.action_space[0].n
            self.last_activation = argmax
            self.n_actions = 1

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
        self.train_writer = open(os.path.join(self.data_folder, "train_evals.csv"), "a+")
        self.test_writer = open(os.path.join(self.data_folder, "test_evals.csv"), "a+")

        header = ', '.join([f"w{i}"for i in range(self.n)])
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
            self.train_writer.write(
                f"{self.n_evals}, {y}, {', '.join(map(str, xi))}\n" 
            )
        return f

    def eval_sequential(self, x):
        envs = gym.make_vec(self.env_name, num_envs=self.n_episodes)
        observations, _ = envs.reset()
        self.net.set_weights(x)

        data_over_time = np.empty((self.n_timesteps, 2, self.n_episodes))
        for t in range(self.n_timesteps):
            actions = self.net(self.normalizer(observations))
            observations, rewards, dones, trunc, *_ = envs.step(actions)
            data_over_time[t] = np.vstack(
                [rewards, np.logical_or(dones, trunc)]
            )

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
        observations = self.obs_mapper(observations)

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
            observations = self.obs_mapper(observations)
            data_over_time[t] = np.vstack(
                [rewards, np.logical_or(dones, trunc)]
            )

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
                    observation = self.obs_mapper(observation)
                    done = terminated or truncated
                    ret += reward
                    if render_mode == "human":
                        print(
                            f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r"
                        )
                    step_index += 1
                if render_mode == "human":
                    print()
                if (
                    render_mode == "rgb_array_list"
                    and episode_index == 0
                ):
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
                    np.save(f"{self.data_folder}/policies/{name}.pkl", x)
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
