from gymnasium import spaces
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WorkerGoalConfig(object):
    def __init__(self, observation_box: spaces.Box):
        self.shape = observation_box.shape
        self.low = np.where(np.isfinite(observation_box.low), observation_box.low, -1)
        self.high = np.where(np.isfinite(observation_box.high), observation_box.high, 1)

    def sample_goal(self):
        return (self.high - self.low) * np.random.sample(self.high.shape) + self.low

    def goal_dim(self):
        # This is NOT 2. This is the full obervation space.
        return self.shape[0]  # x, y position

    def goal_scale(self):
        return np.maximum(self.high, -self.low)
