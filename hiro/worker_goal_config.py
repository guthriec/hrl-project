import time
import warnings
from gymnasium import spaces
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("error", category=RuntimeWarning)


class WorkerGoalConfig(object):
    def __init__(self, observation_box: spaces.Box):
        self.obs_shape = observation_box.shape
        self.obs_low = np.where(
            np.isfinite(observation_box.low), observation_box.low, -1
        )
        self.obs_high = np.where(
            np.isfinite(observation_box.high), observation_box.high, 1
        )

    def random_obs(self):
        return (self.obs_high - self.obs_low) * np.random.sample(
            self.obs_high.shape
        ) + self.obs_low

    def sample_goal(self):
        raise NotImplementedError

    def goal_dim(self):
        raise NotImplementedError

    def goal_scale(self):
        raise NotImplementedError

    # Returns the ideal difference in state space represented by the given goal.
    # Used for logging purposes.
    def ideal_state_change(self, sg):
        return sg

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        raise NotImplementedError

    def subgoal_transition(self, s, sg, n_s):
        return s[: sg.shape[0]] + sg - n_s[: sg.shape[0]]


class PointGoalConfig(WorkerGoalConfig):
    def __init__(self, observation_box: spaces.Box):
        super(PointGoalConfig, self).__init__(observation_box)

    def sample_goal(self):
        return self.random_obs()

    def goal_dim(self):
        # This will be the full observation space.
        return self.obs_shape[0]

    def goal_scale(self):
        return np.maximum(self.obs_high, -self.obs_low)

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) - np.array(first_s))[
            :, np.newaxis, : self.goal_dim()
        ]

        scale = self.goal_scale()
        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(
            loc=diff_goal,
            scale=0.5 * scale[None, None, :],
            size=(batch_size, candidate_goals, original_goal.shape[-1]),
        )
        random_goals = random_goals.clip(-scale, scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.goal_dim())
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:, c]
            candidate = (subgoal + states[:, 0, : self.goal_dim()])[
                :, None
            ] - states[:, :, : self.goal_dim()]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = policy_actions - true_actions
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape(
            (ncands, batch_size, seq_len) + action_dim
        ).transpose(1, 0, 2, 3)

        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]
