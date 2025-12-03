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
import copy

from hiro.buffers import HighReplayBuffer
from .utils import get_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device, flush=True)


def _orthogonal_init(module, gain_hidden=nn.init.calculate_gain("relu"), gain_out=1.0):
    if isinstance(module, nn.Linear):
        # Choose gain based on whether it's an output layer by size heuristic
        is_output = (
            module.out_features == 1
            or module.out_features == 300
            and module.in_features != 300
        )
        gain = gain_out if is_output else gain_hidden
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class TD3Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)
        # Orthogonal init: ReLU gain for hidden, smaller gain for tanh output
        self.apply(
            lambda m: _orthogonal_init(
                m, gain_hidden=nn.init.calculate_gain("relu"), gain_out=0.01
            )
        )

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))


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
        # Orthogonal init: ReLU gain for hidden, linear output uses gain 1.0
        self.apply(
            lambda m: _orthogonal_init(
                m, gain_hidden=nn.init.calculate_gain("relu"), gain_out=1.0
            )
        )

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
        actor_lr=0.0001,
        critic_lr=0.0005,
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
        expl_noise_decay=-1,
    ):
        print("Initializing with critic LR: ", critic_lr, " actor LR: ", actor_lr)
        self.name = "td3"
        self.scale = scale
        self.model_path = model_path

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(
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
        self.logger = None  # optional external logger
        self.critic_grad_clip = 1.0  # max L2 norm for critic gradients
        self.expl_noise_decay = expl_noise_decay

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

    def _train(
        self,
        states,
        goals,
        actions,
        rewards,
        n_states,
        n_goals,
        not_done,
        p_indices,
        buffer,
    ):
        self.total_it += 1

        # Log norms of states and goals
        if self.logger is not None:
            states_norm = states.norm(2, dim=1).mean().item()
            goals_norm = goals.norm(2, dim=1).mean().item()
            max_reward = rewards.max().item()
            self.logger.write(f"norm/goals_{self.name}", goals_norm, self.total_it)
            # Log max reward
            self.logger.write(f"norm/max_reward_{self.name}", max_reward, self.total_it)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            n_actions = self.actor_target(n_states, n_goals) + noise
            n_actions = torch.min(n_actions, self.actor.scale)
            n_actions = torch.max(n_actions, -self.actor.scale)

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(states, goals, actions)
        current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = F.smooth_l1_loss(current_Q1, target_Q_detached)
        critic2_loss = F.smooth_l1_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        all_td_errors = target_Q_detached - current_Q1
        td_error = all_td_errors.mean().cpu().data.numpy()

        buffer.update_priorities(
            p_indices, torch.abs(all_td_errors).cpu().data.numpy().flatten()
        )

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        # Log critic grad norm (L2 over grads that exist)
        if self.logger is not None:
            total_sq = 0.0
            for p in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                if p.grad is not None:
                    g = p.grad.detach()
                    total_sq += float(g.norm(2).item() ** 2)
            grad_norm = total_sq**0.5
            # Use controller's iteration counter as step
            self.logger.write(f"grad/critic_norm_{self.name}", grad_norm, self.total_it)
        # Clip critic gradients before stepping
        # torch.nn.utils.clip_grad_norm_(
        #     self.critic1.parameters(), max_norm=self.critic_grad_clip
        # )
        # torch.nn.utils.clip_grad_norm_(
        #     self.critic2.parameters(), max_norm=self.critic_grad_clip
        # )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a = self.actor(states, goals)
            Q1 = self.critic1(states, goals, a)
            actor_loss = -Q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # # Optional: Add gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
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
        # Assert no reward is larger than 100
        assert (
            rewards.max().item() <= 100
        ), f"Reward exceeds 100: {rewards.max().item()}"
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        action = action + self._sample_exploration_noise(action)
        # TODO: this should use the worker goal box instead of the scale
        action = torch.min(action, self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = self.actor.scale
        # var = torch.ones(actions.size()).to(device)
        expl_noise = self.expl_noise
        if self.expl_noise_decay > 0:
            expl_noise = max(
                1e-8,
                expl_noise
                - (self.expl_noise / (400 * self.expl_noise_decay))
                * (self.total_it // 1000),
            )
        res = torch.normal(mean, expl_noise * var)
        return res

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "logger":
                setattr(result, k, None)  # ignore logger in deep copy
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class HigherController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        worker_goal_config,
        model_path,
        actor_lr=0.01,
        critic_lr=0.2,
        expl_noise=2.0,
        policy_noise=1.0,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
        expl_noise_decay=5.0,
    ):
        super(HigherController, self).__init__(
            state_dim,
            goal_dim,
            worker_goal_config.goal_dim(),
            worker_goal_config.goal_scale(),
            model_path,
            actor_lr,
            critic_lr,
            expl_noise,
            policy_noise,
            noise_clip,
            gamma,
            policy_freq,
            tau,
        )
        self.name = "high"
        self.worker_goal_config = worker_goal_config

    def train(self, replay_buffer: HighReplayBuffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        (
            states,
            goals,
            actions,
            n_states,
            rewards,
            not_done,
            states_arr,
            actions_arr,
            p_indices,
            p_weights,
        ) = replay_buffer.sample()

        # if (rewards.max().item() >= 100):
        #     print("Revisiting success!")

        actions = self.worker_goal_config.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy(),
        )

        actions = get_tensor(actions)
        return self._train(
            states,
            goals,
            actions,
            rewards,
            n_states,
            goals,
            not_done,
            p_indices,
            replay_buffer,
        )


class LowerController(TD3Controller):
    def __init__(
        self,
        state_dim,
        worker_goal_config,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0005,
        critic_lr=0.0002,
        expl_noise=0.5,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
        expl_noise_decay=1.0,
    ):
        super(LowerController, self).__init__(
            state_dim,
            worker_goal_config.goal_dim(),
            action_dim,
            scale,
            model_path,
            actor_lr,
            critic_lr,
            expl_noise,
            policy_noise,
            noise_clip,
            gamma,
            policy_freq,
            tau,
            expl_noise_decay,
        )
        self.name = "low"

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        (
            states,
            sgoals,
            actions,
            n_states,
            n_sgoals,
            rewards,
            not_done,
            p_indices,
            p_weights,
        ) = replay_buffer.sample()

        return self._train(
            states,
            sgoals,
            actions,
            rewards,
            n_states,
            n_sgoals,
            not_done,
            p_indices,
            replay_buffer,
        )
