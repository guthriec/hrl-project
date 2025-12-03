import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size,
                 prioritized=True, alpha=0.6, beta=0.4, eps=1e-6):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.goal = np.zeros((buffer_size, goal_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = device
        # Prioritized Experience Replay (optional)
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        if self.prioritized:
            # priorities initialized small; use max on append for new transitions
            self.priorities = np.full((buffer_size,), 1.0, dtype=np.float32)

    def append(self, state, goal, action, n_state, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        # PER: set priority for new transition to current max to ensure sampling
        if self.prioritized:
            max_prio = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.ptr] = max(max_prio, self.eps)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        # If not prioritized, use uniform sampling as before
        if not self.prioritized:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.goal[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.n_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                # Return indices and uniform weights for compatibility
                torch.LongTensor(ind).to(self.device),
                torch.ones((self.batch_size, 1), dtype=torch.float32).to(self.device),
            )
        # Prioritized sampling
        valid_prios = self.priorities[: self.size]
        probs = np.power(valid_prios + self.eps, self.alpha)
        probs /= probs.sum()
        ind = np.random.choice(self.size, size=self.batch_size, replace=True, p=probs)
        # Importance-sampling weights
        N = self.size
        weights = ((N * probs[ind]) ** (-self.beta)).astype(np.float32)
        weights /= weights.max() + self.eps  # normalize to [0,1]
        weights = weights.reshape(-1, 1)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(ind).to(self.device),
            torch.FloatTensor(weights).to(self.device),
        )

    def update_priorities(self, indices, new_priorities):
        if not self.prioritized:
            return
        # new_priorities is a 1D array-like (e.g., TD errors)
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        # ensure strictly positive
        self.priorities[indices] = np.maximum(new_priorities, self.eps)


class LowReplayBuffer(ReplayBuffer):
    def __init__(
        self, state_dim, worker_goal_config, action_dim, buffer_size, batch_size,
        prioritized=False, alpha=0.6, beta=0.4, eps=1e-6
    ):
        super(LowReplayBuffer, self).__init__(
            state_dim,
            worker_goal_config.goal_dim(),
            action_dim,
            buffer_size,
            batch_size,
            prioritized=prioritized, alpha=alpha, beta=beta, eps=eps,
        )
        self.n_goal = np.zeros((buffer_size, worker_goal_config.goal_dim()))

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.n_goal[self.ptr] = n_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        # PER: set priority for new transition
        if self.prioritized:
            max_prio = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.ptr] = max(max_prio, self.eps)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        if not self.prioritized:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.goal[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.n_state[ind]).to(self.device),
                torch.FloatTensor(self.n_goal[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.LongTensor(ind).to(self.device),
                torch.ones((self.batch_size, 1), dtype=torch.float32).to(self.device),
            )
        valid_prios = self.priorities[: self.size]
        probs = np.power(valid_prios + self.eps, self.alpha)
        probs /= probs.sum()
        ind = np.random.choice(self.size, size=self.batch_size, replace=True, p=probs)
        N = self.size
        weights = ((N * probs[ind]) ** (-self.beta)).astype(np.float32)
        weights /= weights.max() + self.eps
        weights = weights.reshape(-1, 1)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.n_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(ind).to(self.device),
            torch.FloatTensor(weights).to(self.device),
        )


class HighReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        state_dim,
        goal_dim,
        worker_goal_config,
        action_dim,
        buffer_size,
        batch_size,
        freq,
        prioritized=True, alpha=0.6, beta=0.4, eps=1e-6
    ):
        super(HighReplayBuffer, self).__init__(
            state_dim, goal_dim, action_dim, buffer_size, batch_size,
            prioritized=prioritized, alpha=alpha, beta=beta, eps=eps
        )
        self.action = np.zeros((buffer_size, worker_goal_config.goal_dim()))
        self.state_arr = np.zeros((buffer_size, freq, state_dim))
        self.action_arr = np.zeros((buffer_size, freq, action_dim))

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.state_arr[self.ptr, :, :] = state_arr
        self.action_arr[self.ptr, :, :] = action_arr
        # PER: set priority for new transition
        if self.prioritized:
            max_prio = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.ptr] = max(max_prio, self.eps)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        if not self.prioritized:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.goal[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.n_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.state_arr[ind]).to(self.device),
                torch.FloatTensor(self.action_arr[ind]).to(self.device),
                torch.LongTensor(ind).to(self.device),
                torch.ones((self.batch_size, 1), dtype=torch.float32).to(self.device),
            )
        valid_prios = self.priorities[: self.size]
        probs = np.power(valid_prios + self.eps, self.alpha)
        probs /= probs.sum()
        ind = np.random.choice(self.size, size=self.batch_size, replace=True, p=probs)
        N = self.size
        weights = ((N * probs[ind]) ** (-self.beta)).astype(np.float32)
        weights /= weights.max() + self.eps
        weights = weights.reshape(-1, 1)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_arr[ind]).to(self.device),
            torch.FloatTensor(self.action_arr[ind]).to(self.device),
            torch.LongTensor(ind).to(self.device),
            torch.FloatTensor(weights).to(self.device),
        )
