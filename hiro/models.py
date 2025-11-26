##################################################
# @copyright Kandai Watanabe
# @email kandai.wata@gmail.com
# @institute University of Colorado Boulder
#
# NN Models for HIRO
# (Data-Efficient Hierarchical Reinforcement Learning)
# Parameters can be find in the original paper
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hiro.buffers import HighReplayBuffer
from .utils import get_tensor
from .invertible_net import InvertibleNet

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("Using device:", device)


class TD3Actor(nn.Module):
    """Non-invertible TD3 Actor - used by high-level controller"""
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(action_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))


class InvertibleActor(nn.Module):
    """Invertible Actor using normalizing flows - used by low-level controller"""
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super(InvertibleActor, self).__init__()
        if scale is None:
            scale = torch.ones(action_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        # Track dimensions for slicing
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.invertible_net = InvertibleNet(n_layers=4, goal_dim=goal_dim, state_dim=state_dim, action_dim=action_dim, hidden_dim=300)

    def forward(self, state, goal):
        return self.invertible_net.forward(torch.cat([state, goal], 1))


class TD3Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(TD3Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class TD3Controller(object):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_class=InvertibleActor,  # Default to InvertibleActor for backward compatibility
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
    ):
        self.name = "td3"
        self.scale = scale
        self.model_path = model_path
        self.actor_class = actor_class

        # Track dimensions for slicing
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        # Determine if actor outputs full vector (invertible) or just actions
        self.is_invertible = (actor_class == InvertibleActor)

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = actor_class(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = actor_class(state_dim, goal_dim, action_dim, scale=scale).to(
            device
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=critic_lr
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=critic_lr
        )
        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, episode):
        # create episode directory. (e.g. model/2000)
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save file (e.g. model/2000/high_actor.h)
        torch.save(
            self.actor.state_dict(), os.path.join(model_path, self.name + "_actor.h5")
        )
        torch.save(
            self.critic1.state_dict(),
            os.path.join(model_path, self.name + "_critic1.h5"),
        )
        torch.save(
            self.critic2.state_dict(),
            os.path.join(model_path, self.name + "_critic2.h5"),
        )

    def load(self, episode):
        # episode is -1, then read most updated
        if episode < 0:
            episode_list = map(int, os.listdir(self.model_path))
            episode = max(episode_list)

        model_path = os.path.join(self.model_path, str(episode))

        self.actor.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_actor.h5"))
        )
        self.critic1.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_critic1.h5"))
        )
        self.critic2.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_critic2.h5"))
        )

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done):
        self.total_it += 1
        with torch.no_grad():
            n_actions_full = self.actor_target(n_states, n_goals)

            if self.is_invertible:
                # For invertible actor: only add noise to action dimensions
                noise = (torch.randn_like(n_actions_full[:, :self.action_dim]) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                n_actions_full[:, :self.action_dim] = n_actions_full[:, :self.action_dim] + noise
                n_actions_full[:, :self.action_dim] = torch.min(n_actions_full[:, :self.action_dim], self.actor.scale)
                n_actions_full[:, :self.action_dim] = torch.max(n_actions_full[:, :self.action_dim], -self.actor.scale)
                # Slice to get just actions for critic
                n_actions = n_actions_full[:, :self.action_dim]
            else:
                # For non-invertible actor: add noise to all outputs
                noise = (torch.randn_like(n_actions_full) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                n_actions_full = n_actions_full + noise
                n_actions_full = torch.min(n_actions_full, self.actor.scale)
                n_actions_full = torch.max(n_actions_full, -self.actor.scale)
                n_actions = n_actions_full

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).detach()

        # Slice actions for critic if invertible
        if self.is_invertible:
            current_Q1 = self.critic1(states, goals, actions[:, :self.action_dim])
            current_Q2 = self.critic2(states, goals, actions[:, :self.action_dim])
        else:
            current_Q1 = self.critic1(states, goals, actions)
            current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = F.smooth_l1_loss(current_Q1, target_Q_detached)
        critic2_loss = F.smooth_l1_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        td_error = (target_Q_detached - current_Q1).mean().cpu().data.numpy()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a_full = self.actor(states, goals)
            # Slice to get just actions for critic if invertible
            if self.is_invertible:
                a = a_full[:, :self.action_dim]
            else:
                a = a_full
            Q1 = self.critic1(states, goals, a)
            actor_loss = -Q1.mean()  # multiply by neg becuz gradient ascent

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {
                "actor_loss_" + self.name: actor_loss,
                "critic_loss_" + self.name: critic_loss,
            }, {"td_error_" + self.name: td_error}

        return {"critic_loss_" + self.name: critic_loss}, {
            "td_error_" + self.name: td_error
        }

    def train(self, replay_buffer, iterations=1):
        states, goals, actions, n_states, rewards, not_done = replay_buffer.sample()
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action_full = self.actor(state, goal)

        if to_numpy:
            return action_full.cpu().data.numpy().squeeze()

        return action_full.squeeze()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action_full = self.actor(state, goal)

        if self.is_invertible:
            # Add noise and clamp only the action dimensions
            action_full[:, :self.action_dim] = action_full[:, :self.action_dim] + self._sample_exploration_noise(action_full[:, :self.action_dim])
            # TODO: this should use the worker goal box instead of the scale
            action_full[:, :self.action_dim] = torch.min(action_full[:, :self.action_dim], self.actor.scale)
            action_full[:, :self.action_dim] = torch.max(action_full[:, :self.action_dim], -self.actor.scale)
        else:
            # Add noise and clamp all outputs
            action_full = action_full + self._sample_exploration_noise(action_full)
            # TODO: this should use the worker goal box instead of the scale
            action_full = torch.min(action_full, self.actor.scale)
            action_full = torch.max(action_full, -self.actor.scale)

        if to_numpy:
            return action_full.cpu().data.numpy().squeeze()

        return action_full.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        # expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise * var)


class HigherController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        worker_goal_config,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
    ):
        super(HigherController, self).__init__(
            state_dim,
            goal_dim,
            worker_goal_config.goal_dim(),
            worker_goal_config.goal_scale(),
            model_path,
            actor_class=TD3Actor,  # Use non-invertible actor for high-level
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            expl_noise=expl_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            gamma=gamma,
            policy_freq=policy_freq,
            tau=tau,
        )
        self.name = "high"
        self.worker_goal_config = worker_goal_config

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) - np.array(first_s))[
            :, np.newaxis, : self.worker_goal_config.goal_dim()
        ]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(
            loc=diff_goal,
            scale=0.5 * self.scale[None, None, :],
            size=(batch_size, candidate_goals, original_goal.shape[-1]),
        )
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim_full = actions[0][0].shape  # Full action vector shape
        action_dim_env = (low_con.action_dim,)  # Environment action shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim_full)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.worker_goal_config.goal_dim())
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        # Only store environment action dimensions for comparison
        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim_env)

        for c in range(ncands):
            subgoal = candidates[:, c]
            candidate = (subgoal + states[:, 0, : self.worker_goal_config.goal_dim()])[
                :, None
            ] - states[:, :, : self.worker_goal_config.goal_dim()]
            candidate = candidate.reshape(*goal_shape)
            # Policy returns full vector, slice to get just actions for comparison
            policy_actions[c] = low_con.policy(observations, candidate)[:, :low_con.action_dim]

        # Slice true_actions to get just the action dimensions for comparison
        true_actions_sliced = true_actions[:, :low_con.action_dim]
        difference = policy_actions - true_actions_sliced
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape(
            (ncands, batch_size, seq_len) + action_dim_env
        ).transpose(1, 0, 2, 3)

        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer: HighReplayBuffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = (
            replay_buffer.sample()
        )

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy(),
        )

        actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)


class LowerController(TD3Controller):
    def __init__(
        self,
        state_dim,
        worker_goal_config,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
    ):
        super(LowerController, self).__init__(
            state_dim,
            worker_goal_config.goal_dim(),
            action_dim,
            scale,
            model_path,
            actor_class=InvertibleActor,  # Use invertible actor for low-level
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            expl_noise=expl_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            gamma=gamma,
            policy_freq=policy_freq,
            tau=tau,
        )
        self.name = "low"

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        states, sgoals, actions, n_states, n_sgoals, rewards, not_done = (
            replay_buffer.sample()
        )

        return self._train(
            states, sgoals, actions, rewards, n_states, n_sgoals, not_done
        )
