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
    single_episode_per_eval: bool = True
    store_video: bool = True

    def __post_init__(self):
        self.envs = gym.make_vec(self.env_name, num_envs=self.n_episodes)
        if self.env_name == "BipedalWalker-v3":
            self.action_size = 4
            self.n_actions = 4
            self.last_activation = identity
            self.collect_only_single_episode_reward = True
        else:
            self.action_size = self.envs.action_space[0].n
            self.last_activation = argmax
            self.n_actions = 1
            self.collect_only_single_episode_reward = False
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

        collect_reward = np.ones(self.n_episodes, dtype=int)
        episodic_return = np.zeros(self.n_episodes)
        episodic_returns = []

        for _ in range(self.n_timesteps):
            actions = self.net(self.normalizer(observations))
            observations, rewards, dones, trunc, *_ = envs.step(actions)
            rewards *= collect_reward
            rewards = self.fix_reward(rewards, dones)
            episodic_return += rewards
            self.n_train_timesteps += 1

            finished_episodes = np.logical_or(dones, trunc)
            if any(finished_episodes):
                if self.single_episode_per_eval:
                    collect_reward = (collect_reward - finished_episodes).clip(0)

                episodic_returns.extend(episodic_return[finished_episodes])  
                self.n_train_episodes += 1

            if not any(collect_reward):
                break

        return -np.median(episodic_returns)
    
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

        action_shape = self.n_episodes * n
        if self.n_actions > 1:
            action_shape = (action_shape, self.n_actions)

        actions = np.ones(action_shape, dtype=int)
        collect_reward = np.ones(self.n_episodes * n, dtype=int)

        total_returns = np.zeros((n, self.n_episodes))
        episodic_return = np.zeros((n * self.n_episodes))
        episodic_returns = [[] for _ in range(self.n_episodes * n)]
        
        for t in range(self.n_timesteps):
            for i, net in enumerate(self.nets):
                idx = i * self.n_episodes
                actions[idx : idx + self.n_episodes] = net(
                    self.normalizer(observations[idx : idx + self.n_episodes, :])
                )
            observations, rewards, dones, trunc, *_ = self.envs.step(actions)

            rewards *= collect_reward
            self.n_train_timesteps += collect_reward.sum()

            rewards = self.fix_reward(rewards, dones)
            total_returns += rewards.reshape(n, self.n_episodes)
            episodic_return += rewards
            
            finished_episodes = np.logical_or(dones, trunc)
            if any(finished_episodes):
                if self.single_episode_per_eval:
                    collect_reward = (collect_reward - finished_episodes).clip(0)

                idx, *_ = np.where(finished_episodes)
                for i in idx:
                    episodic_returns[i].append(episodic_return[i])
                    episodic_return[i] = 0
                    self.n_train_episodes += 1

            if not any(collect_reward):
                break

        returns = []
        if self.n_episodes != 1:
            for i, j in np.arange(self.n_episodes * n).reshape(n, self.n_episodes):
                returns.append(np.median(np.hstack(episodic_returns[i:j+1])))
        else:
            returns = [np.median(e) for e in episodic_returns]
            
        returns = np.array(returns)
        return -returns


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
                        print(f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r")
                    step_index += 1
                if render_mode == "human":
                    print()
                if self.store_video and render_mode == "rgb_array_list" and episode_index == 0:
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