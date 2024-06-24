import os
import io
from dataclasses import dataclass
from contextlib import redirect_stdout

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video

from skimage.measure import block_reduce

from .network import Network, identity, MinMaxNormalizer, argmax, softmax

def uint8tofloat(obs):
    return ((obs.astype(float) / 255) * 2) - 1



class GaussianProjection:
    def __init__(self, n_components, n_features, mapper=identity, orthogonal=False):
        rng = np.random.default_rng(42)
        self.projection = np.sqrt(3) * rng.choice(
            [-1, 0, 1], p=[1 / 6, 2 / 3, 1 / 6], size=(n_components, n_features)
        )
        if orthogonal:
            u, s, vh = np.linalg.svd(self.projection, full_matrices=False)
            self.projection = u @ vh
        self.mapper = mapper

    def __call__(self, obs):
        obs = self.mapper(obs)
        return obs @ self.projection.T


class FeatureSelector:
    """Simple class that selects a subset of features"""

    def __init__(self, idx, mapper=identity):
        self.idx = np.asarray(idx).astype(int)
        self.n = len(idx)
        self.mapper = mapper

    def __call__(self, obs):
        obs = np.atleast_2d(obs)
        obs = self.mapper(obs)
        obs = obs[:, self.idx]
        return obs


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset and loss of life for environments that are fixed until firing.

    """

    def __init__(
        self,
        env: gym.Env,
        terminate_on_life_loss: bool = False,
        random_fire_action_prob: float = 0.01,
    ):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self.info = dict()
        self.terminate_on_life_loss = terminate_on_life_loss
        self.random_fire_action_prob = random_fire_action_prob

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, trunc, self.info = self.env.step(1)
        return obs, self.info

    def step(self, *args, **kwargs):
        obs, rew, done, trunc, info = self.env.step(*args, **kwargs)
        if info.get("lives") is not None and info.get("lives") != self.info.get(
            "lives"
        ):
            if self.terminate_on_life_loss:
                done = True

        # Randomly add game start actions in order to ensure that the game start
        if np.random.uniform(0, 1) < self.random_fire_action_prob:
            obs, r, done, trunc, info = self.env.step(1)
            rew += r

        self.info = info
        return obs, rew, done, trunc, info


class FrameStacker(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int = 2):
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(self.env.observation_space.shape[0] * n_frames,),
            dtype=self.env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames - 1):
            obs_t, *_ = self.env.step(0)
            obs = np.r_[obs, obs_t]
        return obs, info

    def step(self, *args, **kwargs):
        obs, rew, done, trunc, info = self.env.step(*args, **kwargs)
        for _ in range(self.n_frames - 1):
            obs_t, rew_t, done_t, trunc_t, info = self.env.step(0)
            obs = np.r_[obs, obs_t]
            rew += rew_t
            done = done or done_t
            trunc = trunc or trunc_t

        return obs, rew, done, trunc, info


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
    env_kwargs: dict = None
    obs_mapper: callable = identity
    wrapper: object = None

    def __post_init__(self):
        if self.env_kwargs is None:
            self.env_kwargs = dict()

        env = gym.make(self.name, **self.env_kwargs)
        self.state_size = self.state_size or np.prod(env.observation_space.shape)
        self.is_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.max_episode_steps = self.max_episode_steps or env.spec.max_episode_steps
        self.reward_threshold = env.spec.reward_threshold
        if self.is_discrete:
            self.action_size = env.action_space.n
            self.last_activation = argmax
        else:
            self.action_size = env.action_space.shape[0]

        if self.max_episode_steps is None:
            self.max_episode_steps = int(
                env.spec.kwargs["max_num_frames_per_episode"]
                / env.spec.kwargs["frameskip"]
            )

    def make(self, **kwargs):
        env = gym.make(self.name, **{**self.env_kwargs, **kwargs})
        if self.wrapper is not None:
            env = self.wrapper(env)
        return env

    @property
    def n(self):
        return self.state_size * self.action_size


ENVIRONMENTS = {
    "CartPole-v1": EnvSetting("CartPole-v1", 1000),
    "Acrobot-v1": EnvSetting("Acrobot-v1", 5_000),
    "Pendulum-v1": EnvSetting(
        "Pendulum-v1", 5_000, last_activation=lambda x: 2 * np.tanh(x)
    ),
    "MountainCar-v0": EnvSetting("MountainCar-v0", 5_000),
    "LunarLander-v2": EnvSetting("LunarLander-v2", 10_000),
    "BipedalWalker-v3": EnvSetting(
        "BipedalWalker-v3", 20_000, lambda x: np.clip(x, -1, 1)
    ),
    "Swimmer-v4": EnvSetting("Swimmer-v4", 2_000, last_activation=np.tanh),
    "Reacher-v4": EnvSetting("Reacher-v4", 20_000, last_activation=np.tanh),
    "InvertedPendulum-v4": EnvSetting(
        "InvertedPendulum-v4", 5_000, last_activation=lambda x: 3 * np.tanh(x)
    ),
    "Hopper-v4": EnvSetting(
        "Hopper-v4", 20_000, lambda x: x - 1, last_activation=np.tanh
    ),
    "HalfCheetah-v4": EnvSetting("HalfCheetah-v4", 10_000, last_activation=np.tanh),
    "Walker2d-v4": EnvSetting(
        "Walker2d-v4", 50_000, lambda x: x - 1, last_activation=np.tanh
    ),
    "Ant-v4": EnvSetting(
        "Ant-v4",
        50_000,
        lambda x: x - 1,
        last_activation=np.tanh,
        # obs_mapper=GaussianProjection(8, 27),
        # state_size=8
    ),
    "Humanoid-v4": EnvSetting(
        "Humanoid-v4",
        500_000,
        lambda x: x - 5,
        last_activation=lambda x: 0.4 * np.tanh(x),
    ),
    "HumanoidStandup-v4": EnvSetting(
        "HumanoidStandup-v4",
        500_000,
        last_activation=lambda x: 0.4 * np.tanh(x),
    ),
    "SpaceInvaders-v5": EnvSetting(
        "ALE/SpaceInvaders-v5",
        100_000,
        env_kwargs={
            "obs_type": "ram",
            "frameskip": 4,
            "repeat_action_probability": 0.0,
        },
        obs_mapper=uint8tofloat,  # GaussianProjection(16, 128, uint8tofloat),
        state_size=128,
        # wrapper=lambda env: FrameStacker(env, 3)
    ),
    "Breakout-v5": EnvSetting(
        "ALE/Breakout-v5",
        100_000,
        env_kwargs={
            "obs_type": "ram",
            "repeat_action_probability": 0.0,
            "frameskip": 4,
        },
        obs_mapper=FeatureSelector(
            [
                0,
                6,
                12,
                13,
                18,
                19,
                24,
                25,
                30,
                31,
                57,
                70,
                71,
                72,
                74,
                75,
                77,
                84,
                86,
                90,
                91,
                94,
                95,
                96,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                109,
                119,
                121,
                122,
            ],
            uint8tofloat,
        ),
        state_size=37,
        wrapper=FireResetEnv,
    ),
    "Boxing-v5": EnvSetting(
        "ALE/Boxing-v5",
        100_000,
        env_kwargs={
            "obs_type": "ram",
        },
        obs_mapper=uint8tofloat,
    ),
    "Pong-v5": EnvSetting(
        "ALE/Pong-v5",
        100_000,
        env_kwargs={
            "obs_type": "ram",
            "repeat_action_probability": 0.0,
            "frameskip": 4,
        },
        obs_mapper=FeatureSelector(
            [
                2,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                17,
                18,
                19,
                20,
                21,
                49,
                50,
                51,
                54,
                56,
                58,
                60,
                64,
                67,
                69,
                71,
                73,
                121,
                122,
            ],
            uint8tofloat,
        ),
        state_size=29,
    ),
}

CLASSIC_CONTROL = [
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCar-v0",
]

BOX2D = ["LunarLander-v2", "BipedalWalker-v3"]

MUJOCO = [
    "Reacher-v4",
    "InvertedPendulum-v4",
    "Hopper-v4",
    "HalfCheetah-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

ATARI = ["SpaceInvaders-v5", "Pong-v5", "Breakout-v5"]


def rgb_to_gray_flat(observations):
    gray = np.dot(observations[..., :3], [0.2989, 0.5870, 0.1140])
    reduced = block_reduce(gray, (1, 6, 6), np.max)
    return reduced.reshape(len(reduced), -1)


class Normalizer:
    def __init__(self, nb_inputs):
        self.mean = np.zeros(nb_inputs)
        self.var = np.ones(nb_inputs)
        self.std = np.ones(nb_inputs)

    def observe(self, _):
        pass

    def __call__(self, x):
        return (x - self.mean) / self.std


class Standardizer(Normalizer):
    def __init__(self, nb_inputs):
        super().__init__(nb_inputs)
        self.k = nb_inputs
        self.s = np.zeros(nb_inputs)

    def observe(self, X):
        for x in X:
            self.k += 1
            delta = x - self.mean.copy()
            self.mean += delta / self.k
            self.s += delta * (x - self.mean)

        self.var = self.s / (self.k - 1)
        self.std = np.sqrt(self.var)
        self.std[self.std < 1e-7] = np.inf


def regularize(x, y, alpha=0.1):
    reg = np.power(x, 2).sum(axis=0) * np.abs(y) * alpha
    return y + reg


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
    normalized: bool = False
    bias: bool = False
    eval_total_timesteps: bool = True
    store_video: bool = True
    aggregator: callable = np.mean
    n_train_timesteps: int = 0
    n_train_episodes: int = 0
    n_test_evals: int = 0
    n_evals: int = 0
    data_folder: str = None
    regularized: bool = False
    seed_train_envs: int = None

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
            self.bias,
        )
        self.n = self.net.n_weights
        self.nets = []

    def open(self):
        self.train_writer = open(
            os.path.join(self.data_folder, "train_evals.csv"), "a+"
        )
        self.test_writer = open(os.path.join(self.data_folder, "test_evals.csv"), "a+")
        # header = ", ".join([f"w{i}" for i in range(self.n)])
        header = f"evals, fitness\n"  # , {header}\n"
        self.train_writer.write(header)
        self.test_writer.write(header)

    def __call__(self, x):
        if self.parallel:
            f = self.eval_parallel(x)
        else:
            f = np.array([self.eval_sequential(xi) for xi in x.T])
        for y, xi in zip(f, x.T):
            self.n_evals += 1
            self.train_writer.write(f"{self.n_evals}, {y}\n")
        return f

    def reset_envs(self, envs):
        seeds = None
        if self.seed_train_envs is not None:
            seeds = [self.seed_train_envs * 7 * i for i in range(1, 1 + envs.num_envs)]
        observations, *_ = envs.reset(seed=seeds)
        return observations

    def eval_sequential(self, x, count: bool = True):
        envs = gym.vector.AsyncVectorEnv(
            [lambda: self.setting.make() for _ in range(self.n_episodes)]
        )
        observations = self.reset_envs(envs)

        self.net.set_weights(x)

        data_over_time = np.zeros((self.n_timesteps, 2, self.n_episodes))

        for t in range(self.n_timesteps):
            observations = self.setting.obs_mapper(observations)
            actions = self.net(self.normalizer(observations))
            self.normalizer.observe(observations)
            observations, rewards, dones, trunc, *_ = envs.step(actions)
            
            if count:
                rewards = self.setting.reward_shaping(rewards)

            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])
            if not self.eval_total_timesteps and np.logical_or(dones, trunc):
                break

        returns = []
        for i in range(self.n_episodes):
            ret, n_eps, n_timesteps = self.calculate_returns(data_over_time[:, :, i])
            if count:
                self.n_train_timesteps += n_timesteps
                self.n_train_episodes += n_eps
            returns.extend(ret)

        y = -self.aggregator(returns)
        if self.regularized:
            y = regularize(x, y)
        return y

    def eval_parallel(self, x, count: bool = True):
        n = x.shape[1]
        if n != len(self.nets):
            self.nets = [
                Network(
                    self.setting.state_size,
                    self.setting.action_size,
                    self.n_hidden,
                    self.n_layers,
                    self.setting.last_activation,
                    self.bias,
                )
                for _ in range(n)
            ]
            self.envs = gym.vector.AsyncVectorEnv(
                [lambda: self.setting.make() for _ in range(self.n_episodes * n)]
            )

        for net, w in zip(self.nets, x.T):
            net.set_weights(w)

        observations = self.reset_envs(self.envs)

        n_total_episodes = action_shape = self.n_episodes * n

        actions = np.ones(action_shape, dtype=int)
        if not self.setting.is_discrete:
            action_shape = (action_shape, self.setting.action_size)
            actions = np.ones(action_shape, dtype=float)

        data_over_time = np.zeros((self.n_timesteps, 2, n_total_episodes))
        for t in range(self.n_timesteps):
            observations = self.setting.obs_mapper(observations)
            for i, net in enumerate(self.nets):
                idx = i * self.n_episodes
                obs = observations[idx : idx + self.n_episodes, :]
                actions[idx : idx + self.n_episodes] = net(self.normalizer(obs))
                self.normalizer.observe(obs)

            observations, rewards, dones, trunc, info = self.envs.step(actions)
            if count:
                rewards = self.setting.reward_shaping(rewards)
            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])

            first_ep_done = data_over_time[:, 1, :].sum(axis=0) >= 1
            first_ep_all_done = first_ep_done.all()

            if not self.eval_total_timesteps and first_ep_all_done:
                break

        aggregated_returns = np.empty(n)
        for k, j in enumerate(range(0, n_total_episodes, self.n_episodes)):
            returns = []
            for i in range(self.n_episodes):
                ret, n_eps, n_timesteps = self.calculate_returns(
                    data_over_time[:, :, j + i]
                )
                if count:
                    self.n_train_timesteps += n_timesteps
                    self.n_train_episodes += n_eps
                returns.extend(ret)
            aggregated_returns[k] = self.aggregator(returns)

        y = -aggregated_returns
        if self.regularized:
            y = regularize(x, y)
        return y

    def calculate_returns(self, Y):
        _, idx = np.unique(np.cumsum(Y[:, 1]) - Y[:, 1], return_index=True)
        episodes = np.split(Y[:, 0], idx)[1:]
        if len(episodes) > 1:
            episodes = episodes[:-1]

        returns_ = [x.sum() for x in episodes]
        n_timesteps = len(Y)
        if not self.eval_total_timesteps:
            returns_ = returns_[:1]
            n_timesteps = len(episodes[0])

        # TODO: we can remove incomplete episodes from the last optionally
        return returns_, len(returns_), n_timesteps

    def test_policy(self, x, name: str = None, save: bool = True):
        x = x.reshape(-1, 1)
        X = np.tile(x, (1, self.n_test_episodes))
        returns = self.eval_parallel(X, False)
        if save:
            os.makedirs(f"{self.data_folder}/policies", exist_ok=True)
            loc = f"{self.data_folder}/policies/{name}"
            
            np.save(loc, x)
            np.save(f"{loc}-norm-std", self.normalizer.std)
            np.save(f"{loc}-norm-mean", self.normalizer.mean)
            
            for ret in returns:
                self.n_test_evals += 1
                self.test_writer.write(f"{self.n_test_evals}, {ret}\n")
            
            self.play_check(loc, 'rgb_array_list', name)

        return np.mean(returns), np.median(returns), np.std(returns)
    
    def load_network(self, loc: str):
        net = Network(
            self.setting.state_size,
            self.setting.action_size,
            self.n_hidden,
            self.n_layers,
            self.setting.last_activation,
            self.bias,
        )
        net.set_weights(np.load(f"{loc}.npy"))
        if self.normalized:
            normalizer = Standardizer(self.setting.state_size)
        else:
            normalizer = Normalizer(self.setting.state_size)
        normalizer.std = np.load(f"{loc}-norm-std.npy")
        normalizer.mean = np.load(f"{loc}-norm-mean.npy")
        return net, normalizer

    def play_check(self, location, render_mode=None, name=None, n_reps: int = 1):
        if not self.store_video and render_mode == "rgb_array_list":
            return 

        net, normalizer = self.load_network(location)
        returns = []
        try:
            for episode_index in range(n_reps):
                env = self.setting.make(render_mode=render_mode)
                if render_mode == "rgb_array_list":
                    env.metadata['render_fps'] = max(env.metadata.get('render_fps') or 60, 60)
                
                observation, *_ = env.reset()

                if render_mode == "human":
                    env.render()
                done = False
                ret = 0
                step_index = 0
                while not done:
                    observation = self.setting.obs_mapper(observation)
                    obs = normalizer(observation.reshape(1, -1))
                    action, *_ = net(obs)
                    observation, reward, terminated, truncated, *_ = env.step(action)
                    done = terminated or truncated

                    ret += reward
                    if render_mode == "human":
                        print(
                            f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r"
                        )
                    step_index += 1
                returns.append(ret)
                if render_mode == "human":
                    print()
                if render_mode == "rgb_array_list" and episode_index == 0:
                    if self.store_video:
                        os.makedirs(f"{self.data_folder}/videos", exist_ok=True)
                        with redirect_stdout(io.StringIO()):
                            save_video(
                                env.render(),
                                f"{self.data_folder}/videos",
                                fps=env.metadata.get("render_fps"),
                                step_starting_index=0,
                                episode_index=0,
                                name_prefix=name,
                            )
                    render_mode = None

        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return returns