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

    def point_representation(self, sg):
        return sg

    def worker_reward(self, s, sg, n_s):
        raise NotImplementedError

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
        return self.obs_shape[0]

    def goal_scale(self):
        return np.maximum(self.obs_high, -self.obs_low)

    # Use potential-based reward
    def worker_reward(self, s, sg, n_s):
        abs_sg = s[: sg.shape[0]] + sg
        prev_dist = np.sqrt(np.sum((abs_sg - s[: sg.shape[0]]) ** 2))
        new_dist = np.sqrt(np.sum((abs_sg - n_s[: sg.shape[0]]) ** 2))
        return prev_dist - new_dist

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        goal_scale = self.goal_scale()
        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) - np.array(first_s))[
            :, np.newaxis, : self.goal_dim()
        ]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(
            loc=diff_goal,
            scale=0.5 * goal_scale[None, None, :],
            size=(batch_size, candidate_goals, original_goal.shape[-1]),
        )
        random_goals = random_goals.clip(-goal_scale, goal_scale)

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


class EllipsoidGoalConfig(WorkerGoalConfig):
    def __init__(self, observation_box: spaces.Box):
        super(EllipsoidGoalConfig, self).__init__(observation_box)

    def sample_goal(self):
        max_radius = 1e5 * self.obs_shape[0]
        max_log = np.log(max_radius)
        random_radii = 2 * max_log * np.random.sample(
            self.obs_shape[0]
        ) - max_log
        return np.concatenate((self.random_obs(), random_radii))

    def goal_dim(self):
        res = 2 * self.obs_shape[0]  # all observation dims + ellipsoid radii
        return res

    def point_representation(self, sg):
        return sg[: self.obs_shape[0]]

    def goal_scale(self):
        obs_scale = np.maximum(self.obs_high, -self.obs_low)
        max_radius = 1e5 * self.obs_shape[0]
        res = np.concatenate((obs_scale, np.log(max_radius) * np.ones(self.obs_shape[0])))
        return res

    def worker_reward(self, s, sg, n_s):
        abs_sg = s[: sg.shape[0] // 2] + sg[: sg.shape[0] // 2]
        log_radii = sg[sg.shape[0] // 2 :]
        prev_scaled_diffs = self.scaled_difference(abs_sg, log_radii, s)
        scaled_diffs = self.scaled_difference(abs_sg, log_radii, n_s)
        prev_dist = np.sqrt(np.sum(prev_scaled_diffs ** 2))
        new_dist = np.sqrt(np.sum(scaled_diffs ** 2))
        return prev_dist - new_dist

    def scaled_difference(self, abs_point_sg, log_radii, s):
        return (s[: abs_point_sg.shape[0]] - abs_point_sg) / np.exp(log_radii)

    def subgoal_transition(self, s, sg, n_s):
        n_sg = sg.copy()
        adj_s = s[:-1]
        adj_n_s = n_s[:-1]
        n_sg[:adj_s.shape[0]] += adj_s - adj_n_s
        return n_sg

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=None
    ):
        start_time = time.perf_counter()
        res = np.array(self.corrected_sgoals(
            sgoals, states
        ))
        end_time = time.perf_counter()
        # print(f"Execution time: {end_time - start_time:.2f} seconds")
        return res

    def corrected_sgoals(
        self, sgoals, states
    ):
        return [self.corrected_sgoal(sg, state_seq)
                for sg, state_seq in zip(sgoals, states)]

    def corrected_sgoal(self, sg, state_seq):
        final_s = state_seq[-1]
        initial_s = state_seq[0]
        abs_point_sg = initial_s[: sg.shape[0] // 2] + sg[: sg.shape[0] // 2]
        log_radii = sg[sg.shape[0] // 2 :]
        while True:
            scaled_difference = self.scaled_difference(abs_point_sg, log_radii, final_s)
            if np.sum(scaled_difference ** 2) < 1:
                break
            # Pick dimensions that contribute the most to the norm violation.
            # Use absolute value and a sensible threshold of 1/sqrt(d) where d is dimension.
            dim = scaled_difference.size
            thresh = 1.0 / np.sqrt(dim)
            mask = np.abs(scaled_difference) > thresh
            if np.any(mask):
                # Bump all offending radii at once (vectorized)
                log_radii[mask] += 0.5
            else:
                # If nothing exceeds the per-dim threshold, bump the worst offender
                idx = int(np.argmax(np.abs(scaled_difference)))
                log_radii[idx] += 0.5
        return np.concatenate((abs_point_sg, log_radii))
