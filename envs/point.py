import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from gymnasium import utils
import os


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Simple 2D point mass robot - much easier than Ant."""

    FILE = "point.xml"

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, file_path=None, frame_skip=5, expose_all_qpos=True, **kwargs):
        self.expose_all_qpos = expose_all_qpos

        # Resolve default model path if none provided
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), "assets", self.FILE)

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )

        # Gymnasium's MujocoEnv requires an observation_space argument; we
        # provide None to let the env infer it from _get_obs during setup.
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip, observation_space=None)
        utils.EzPickle.__init__(self, file_path, frame_skip, expose_all_qpos, **kwargs)


    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:2],  # x, y position
                self.data.qvel.flat[:2],  # x, y velocity
            ]
        )

    def get_xy(self):
        """Get the x, y position of the point mass."""
        return self.data.qpos.flat[:2].copy()

    def set_xy(self, xy):
        """Set the x, y position of the point mass."""
        qpos = self.data.qpos.copy()
        qpos[:2] = xy
        qvel = self.data.qvel.copy()
        self.set_state(qpos, qvel)

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.1, 0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        # Apply action and simulate
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = 0.0  # Reward handled by MazeEnv wrapper
        terminated = False
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
