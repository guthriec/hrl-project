import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from gymnasium import utils
import os
import xml.etree.ElementTree as ET


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

        # Parse XML to get joint limits
        tree = ET.parse(file_path)
        joints = tree.findall(".//joint")

        # Build observation bounds: [x, y, theta, vx, vy, vtheta]
        qpos_low = []
        qpos_high = []
        for joint in joints:
            if joint.get("limited") == "true" and "range" in joint.attrib:
                range_str = joint.get("range")
                assert range_str is not None
                low, high = map(float, range_str.split())
                qpos_low.append(low)
                qpos_high.append(high)
            else:
                # Unlimited joint
                qpos_low.append(-np.inf)
                qpos_high.append(np.inf)

        # Velocity limits (assume symmetric and large)
        qvel_low = [-np.inf] * 3
        qvel_high = [np.inf] * 3

        obs_low = np.array(qpos_low + qvel_low, dtype=np.float64)
        obs_high = np.array(qpos_high + qvel_high, dtype=np.float64)

        observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        mujoco_env.MujocoEnv.__init__(
            self,
            file_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=kwargs.get("render_mode", None),
        )
        utils.EzPickle.__init__(self, file_path, frame_skip, expose_all_qpos, **kwargs)

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flat[:3],  # x, y, theta position
                self.data.qvel.flat[:3],  # x, y, theta velocity
            ]
        )
        assert self.observation_space.contains(
            obs
        ), f"Observation {obs} not in space {self.observation_space}"
        return obs

    def get_xy(self):
        """Get the x, y position of the car."""
        return self.data.qpos.flat[:2].copy()

    def set_xy(self, xy):
        """Set the x, y position of the car."""
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
        # Convert car controls (steering, throttle) to x/y velocities
        # action[0] = steering rate (angular velocity)
        # action[1] = throttle (forward speed)

        theta = self.data.qpos[2]  # Current heading
        steering = action[0] if len(action) > 0 else 0.0
        throttle = action[1] if len(action) > 1 else 0.0

        # Convert to world frame velocities
        vx = throttle * np.cos(theta)
        vy = throttle * np.sin(theta)

        # Create control vector: [steering, vx, vy]
        ctrl = np.array([steering, vx, vy])

        # Apply action and simulate
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        reward = 0.0  # Reward handled by MazeEnv wrapper
        terminated = False
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
