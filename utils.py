import numpy as np 
import gymnasium as gym
from skimage.measure import block_reduce


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()


def argmax(x):
    return np.argmax(x, axis=1)


def identity(x):
    return x


def clip(lb, ub):
    def inner(x):
        return np.clip(x, lb, ub)

    return inner


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
        random_fire_action_prob: float = 0.0,
    ):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self.info = dict()
        self.terminate_on_life_loss = terminate_on_life_loss
        self.random_fire_action_prob = random_fire_action_prob
        self.t = 0

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.t = 0
        obs, _, done, trunc, self.info = self.env.step(1)
        if done:
            self.env.reset(**kwargs)

        obs, _, done, trunc, self.info = self.env.step(2)

        if done:
            self.env.reset(**kwargs)
        return obs, self.info

    def step(self, *args, **kwargs):
        self.t += 1
        obs, rew, done, trunc, info = self.env.step(*args, **kwargs)
        if info.get("lives") is not None and info.get("lives") != self.info.get(
            "lives"
        ):
            if self.terminate_on_life_loss:
                done = True

        # Randomly add game start actions in order to ensure that the game start
        # if np.random.uniform(0, 1) < self.random_fire_action_prob:
        if self.t % 100 == 0:
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




def rgb_to_gray_flat(observations):
    gray = np.dot(observations[..., :3], [0.2989, 0.5870, 0.1140])
    reduced = block_reduce(gray, (1, 6, 6), np.max)
    return reduced.reshape(len(reduced), -1)