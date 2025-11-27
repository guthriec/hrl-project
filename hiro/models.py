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
import time

from hiro.buffers import HighReplayBuffer
from .utils import get_tensor
from . import utils
from .invertible_net import InvertibleNet


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

        # Track dimensions for slicing
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        # Pass scale directly to InvertibleNet (it will process it)
        self.invertible_net = InvertibleNet(
            n_layers=4,
            goal_dim=goal_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=300,
            scale=scale
        )

    def forward(self, state, goal):
        return self.invertible_net.forward(state, goal)


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

        # Create actor
        self.actor = actor_class(state_dim, goal_dim, action_dim, scale=scale).to(utils.device)
        self.actor_target = actor_class(state_dim, goal_dim, action_dim, scale=scale).to(utils.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(utils.device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(utils.device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(utils.device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(utils.device)

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
                scale_tensor = get_tensor(self.scale)
                n_actions_full[:, :self.action_dim] = torch.min(n_actions_full[:, :self.action_dim], scale_tensor)
                n_actions_full[:, :self.action_dim] = torch.max(n_actions_full[:, :self.action_dim], -scale_tensor)
                # Slice to get just actions for critic
                n_actions = n_actions_full[:, :self.action_dim]
            else:
                # For non-invertible actor: add noise to all outputs
                noise = (torch.randn_like(n_actions_full) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                n_actions_full = n_actions_full + noise
                scale_tensor = get_tensor(self.scale)
                n_actions_full = torch.min(n_actions_full, scale_tensor)
                n_actions_full = torch.max(n_actions_full, -scale_tensor)
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

            # Add reconstruction and latent losses for invertible actor
            if self.is_invertible:
                # Output structure: [action (2) | state (8) | latent (6)]
                # a_full shape: (batch, state_dim + goal_dim) = (batch, 16)

                # State reconstruction loss: output state should match input state
                state_start = self.action_dim  # 2
                state_end = self.action_dim + self.state_dim  # 2 + 8 = 10
                reconstructed_state = a_full[:, state_start:state_end]
                state_reconstruction_loss = F.mse_loss(reconstructed_state, states)
                # if self.total_it % 10 == 0:
                #     print(self.total_it, " ", states[0, :], " ", reconstructed_state[0, :])

                # Latent regularization loss: encourage N(0, 1) distribution
                latent_start = state_end  # 10
                latent_vars = a_full[:, latent_start:]  # dims 10:16 (6 dimensions)

                # Compute batch statistics
                latent_mean = latent_vars.mean(dim=0)  # mean across batch for each latent dim
                latent_std = latent_vars.std(dim=0)   # std across batch for each latent dim

                # Regularize to N(0, 1)
                latent_mean_loss = (latent_mean ** 2).mean()  # encourage mean ≈ 0
                latent_std_loss = ((latent_std - 1.0) ** 2).mean()  # encourage std ≈ 1
                latent_regularization_loss = latent_mean_loss + latent_std_loss

                # Debug: Check inverse quality every 1000 iterations
                if self.total_it % 1000 == 0:
                    with torch.no_grad():
                        # Test inverse with latent=0
                        # Use first sample from batch
                        test_state = states[0:1]  # (1, state_dim)
                        test_goal = goals[0:1]    # (1, goal_dim)
                        test_output = a_full[0:1]  # (1, state_dim + goal_dim)

                        # Replace latent with zeros
                        test_output_zero_latent = test_output.clone()
                        test_output_zero_latent[:, latent_start:] = 0

                        # Invert to recover state and goal
                        state_inv, goal_inv = self.actor.invertible_net.inverse(test_output_zero_latent)

                        # Compute errors
                        state_error = (state_inv - test_state).abs().mean().item()
                        goal_error = (goal_inv - test_goal).abs().mean().item()

                        print(f"\n[Inverse Quality Check @ iter {self.total_it}]")
                        print(f"  Original goal: {test_goal[0].cpu().numpy()}")
                        print(f"  Inverted goal: {goal_inv[0].cpu().numpy()}")
                        print(f"  Goal error (MAE): {goal_error:.6f}")
                        print(f"  State error (MAE): {state_error:.6f}\n")

                # Inverse reconstruction loss: enforce cycle consistency
                # Sample latent from N(0,1) to ensure robustness across the distribution
                latent_sampled = torch.randn_like(latent_vars)
                # Construct synthetic output: [action, state, latent_sampled]
                synthetic_output = torch.cat([
                    a_full[:, :self.action_dim],  # actions from forward pass
                    states,  # input state (not reconstructed)
                    latent_sampled  # sampled latent
                ], dim=1)

                # Invert to get (state, goal)
                state_inv, goal_inv = self.actor.invertible_net.inverse(synthetic_output)

                # Only goal should reconstruct (state already handled by forward reconstruction)
                inverse_reconstruction_loss = F.mse_loss(goal_inv, goals)

                # Combine losses (using weights from LowerController)
                actor_loss = (actor_loss +
                             self.recon_weight * state_reconstruction_loss +
                             self.latent_weight * latent_regularization_loss +
                             self.inverse_recon_weight * inverse_reconstruction_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            losses_dict = {
                "actor_loss_" + self.name: actor_loss,
                "critic_loss_" + self.name: critic_loss,
            }

            # Add invertible-specific losses to logging
            if self.is_invertible:
                losses_dict["state_recon_loss_" + self.name] = state_reconstruction_loss
                losses_dict["latent_reg_loss_" + self.name] = latent_regularization_loss
                losses_dict["inverse_recon_loss_" + self.name] = inverse_reconstruction_loss

            return losses_dict, {"td_error_" + self.name: td_error}

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

        scale_tensor = get_tensor(self.scale)
        if self.is_invertible:
            # Add noise and clamp only the action dimensions
            action_full[:, :self.action_dim] = action_full[:, :self.action_dim] + self._sample_exploration_noise(action_full[:, :self.action_dim])
            action_full[:, :self.action_dim] = torch.min(action_full[:, :self.action_dim], scale_tensor)
            action_full[:, :self.action_dim] = torch.max(action_full[:, :self.action_dim], -scale_tensor)
        else:
            # Add noise and clamp all outputs
            action_full = action_full + self._sample_exploration_noise(action_full)
            action_full = torch.min(action_full, scale_tensor)
            action_full = torch.max(action_full, -scale_tensor)

        if to_numpy:
            return action_full.cpu().data.numpy().squeeze()

        return action_full.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(utils.device)
        var = torch.ones(actions.size()).to(utils.device)
        # expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise * var)


class HigherController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        worker_goal_config,
        model_path,
        opc_method="invertible",  # "original", "invertible", or "none"
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
        self.opc_method = opc_method

    def off_policy_corrections_original(
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
    
    def no_off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        """
        Use invertible net inversion to generate candidate subgoals.
        Vectorized implementation.
        """
        original_goal = np.array(sgoals)
        return original_goal
    

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        """
        Use invertible net inversion to generate candidate subgoals.
        Generates candidates using: original goal, latent=0, and sampled latents.
        """
        t_start = time.time()

        states_np = np.array(states)  # (batch_size, seq_len, state_dim)
        actions_np = np.array(actions)  # (batch_size, seq_len, full_action_dim)
        original_goals = np.array(sgoals)  # (batch_size, subgoal_dim)
        seq_len = states_np.shape[1]
        action_dim = low_con.action_dim
        subgoal_dim = self.worker_goal_config.goal_dim()

        # Latent dimension: output is [action | state | latent], total = state_dim + goal_dim
        # So latent_dim = (state_dim + goal_dim) - action_dim - state_dim = goal_dim - action_dim
        latent_dim = low_con.goal_dim - action_dim

        # === Generate candidates via inversion ===
        # Use first timestep for inversion
        s_0 = states_np[:, 0, :]  # (batch_size, state_dim)
        a_0 = actions_np[:, 0, :action_dim]  # (batch_size, action_dim)

        # Generate candidates with different latents:
        # 1 with latent=0 + (candidate_goals-2) with sampled latents + 1 original goal = candidate_goals total
        n_sampled = candidate_goals - 2

        # Create latent vectors: [zero] + [sampled_1, ..., sampled_n]
        latents_list = [np.zeros((batch_size, latent_dim))]
        for _ in range(n_sampled):
            latents_list.append(np.random.randn(batch_size, latent_dim))

        # Stack all latents: (batch_size, n_latent_candidates, latent_dim)
        latents_stacked = np.stack(latents_list, axis=1)  # (batch, candidate_goals-1, latent_dim)
        n_latent_candidates = latents_stacked.shape[1]

        # Construct output vectors for all latent candidates: [action, state, latent]
        # Tile action and state for each latent candidate
        actions_tiled = np.tile(a_0[:, None, :], (1, n_latent_candidates, 1))  # (batch, n_latent_candidates, action_dim)
        states_tiled = np.tile(s_0[:, None, :], (1, n_latent_candidates, 1))  # (batch, n_latent_candidates, state_dim)

        output_vectors = np.concatenate([actions_tiled, states_tiled, latents_stacked], axis=-1)
        # (batch, n_latent_candidates, action_dim + state_dim + latent_dim)

        # Flatten for batch inversion
        output_flat = output_vectors.reshape(-1, low_con.state_dim + low_con.goal_dim)

        t_prep = time.time() - t_start
        t0 = time.time()

        output_tensor = get_tensor(output_flat)

        # Invert all at once
        with torch.no_grad():
            _, goals_inv = low_con.actor.invertible_net.inverse(output_tensor)

        goals_inv_np = goals_inv.cpu().numpy()  # (batch*n_latent_candidates, goal_dim)

        t_invert = time.time() - t0
        t0 = time.time()

        goals_inv_np = goals_inv_np.reshape(batch_size, n_latent_candidates, -1)  # (batch, n_latent_candidates, goal_dim)
        goals_inv_np = goals_inv_np[:, :, :subgoal_dim]  # Only use subgoal dimensions

        # Check for nan/inf and filter
        if np.any(np.isnan(goals_inv_np)) or np.any(np.isinf(goals_inv_np)):
            print("Warning: NaN or Inf detected in inverted goals, replacing with zeros...")
            goals_inv_np = np.nan_to_num(goals_inv_np, nan=0.0, posinf=0.0, neginf=0.0)

        # At t=0, inverted relative_goal equals the subgoal (no conversion needed)
        # relative_goal_0 = s_0 + subgoal - s_0 = subgoal
        inverted_candidates = goals_inv_np  # (batch, n_latent_candidates, subgoal_dim)

        # Filter unrealistic goals: clip to reasonable bounds (3x goal scale)
        goal_scale = self.worker_goal_config.goal_scale()[:subgoal_dim]
        #inverted_candidates = np.clip(inverted_candidates, -3 * goal_scale, 3 * goal_scale)

        # Add original goal as first candidate
        # candidates shape: (batch, candidate_goals, subgoal_dim)
        candidates = np.concatenate([
            original_goals[:, None, :],  # (batch, 1, subgoal_dim)
            inverted_candidates  # (batch, n_latent_candidates, subgoal_dim)
        ], axis=1)

        # === Evaluate candidates (vectorized) ===
        ncands = candidate_goals
        s_t = states_np[:, :, :subgoal_dim]  # (batch, seq_len, subgoal_dim)
        s_0_for_scoring = states_np[:, 0:1, :subgoal_dim]  # (batch, 1, subgoal_dim)

        # For each candidate, compute relative goals for all timesteps
        # Shape expansions: candidates (batch, ncands, subgoal_dim), s_0 (batch, 1, subgoal_dim), s_t (batch, seq_len, subgoal_dim)
        candidates_expanded = candidates[:, :, None, :]  # (batch, ncands, 1, subgoal_dim)
        s_0_expanded = s_0_for_scoring[:, None, :, :]  # (batch, 1, 1, subgoal_dim)
        s_t_expanded = s_t[:, None, :, :]  # (batch, 1, seq_len, subgoal_dim)

        # relative_goal = s_0 + subgoal - s_t
        relative_goals = s_0_expanded + candidates_expanded - s_t_expanded  # (batch, ncands, seq_len, subgoal_dim)

        # Flatten for policy evaluation
        relative_goals_flat = relative_goals.reshape(batch_size * ncands * seq_len, -1)
        states_flat = np.tile(states_np, (ncands, 1, 1)).reshape(batch_size * ncands * seq_len, -1)

        t_candidate_prep = time.time() - t0
        t0 = time.time()

        # Get policy actions
        policy_out = low_con.policy(states_flat, relative_goals_flat)[:, :action_dim]
        policy_actions = policy_out.reshape(batch_size, ncands, seq_len, action_dim)

        t_policy_eval = time.time() - t0
        t0 = time.time()

        # Compare with true actions
        actions_env = actions_np[:, :, :action_dim]  # (batch, seq_len, action_dim)
        true_actions_expanded = actions_env[:, None, :, :]  # (batch, 1, seq_len, action_dim)

        difference = policy_actions - true_actions_expanded  # (batch, ncands, seq_len, action_dim)
        difference = np.where(difference != -np.inf, difference, 0)

        # Compute log probabilities
        # Sum over timesteps and action dimensions
        logprobs = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)  # (batch, ncands)

        # Pick best candidate per batch
        max_indices = np.argmax(logprobs, axis=-1)  # (batch,)

        t_scoring = time.time() - t0

        if not hasattr(self, '_opc_timer_count'):
            self._opc_timer_count = 0
            self._opc_prep = 0.0
            self._opc_invert = 0.0
            self._opc_cand_prep = 0.0
            self._opc_policy = 0.0
            self._opc_score = 0.0

        self._opc_timer_count += 1
        self._opc_prep += t_prep
        self._opc_invert += t_invert
        self._opc_cand_prep += t_candidate_prep
        self._opc_policy += t_policy_eval
        self._opc_score += t_scoring

        if self._opc_timer_count % 100 == 0:
            n = 100
            print(f"  [OPC breakdown] prep={self._opc_prep/n:.4f}s, invert={self._opc_invert/n:.4f}s, cand_prep={self._opc_cand_prep/n:.4f}s, policy={self._opc_policy/n:.4f}s, score={self._opc_score/n:.4f}s")
            print(f"  [OPC candidates] batch[0] original goal: {original_goals[0]}")
            print(f"  [OPC candidates] batch[0] all candidates:\n{candidates[0]}")
            print(f"  [OPC candidates] batch[0] selected index: {max_indices[0]}, selected: {candidates[0, max_indices[0]]}")
            self._opc_prep = 0.0
            self._opc_invert = 0.0
            self._opc_cand_prep = 0.0
            self._opc_policy = 0.0
            self._opc_score = 0.0

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer: HighReplayBuffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = (
            replay_buffer.sample()
        )

        t0 = time.time()
        # Select off-policy correction method
        if self.opc_method == "original":
            actions = self.off_policy_corrections_original(
                low_con,
                replay_buffer.batch_size,
                actions.cpu().data.numpy(),
                states_arr.cpu().data.numpy(),
                actions_arr.cpu().data.numpy(),
            )
        elif self.opc_method == "invertible":
            actions = self.off_policy_corrections(
                low_con,
                replay_buffer.batch_size,
                actions.cpu().data.numpy(),
                states_arr.cpu().data.numpy(),
                actions_arr.cpu().data.numpy(),
            )
        elif self.opc_method == "none":
            actions = self.no_off_policy_corrections(
                low_con,
                replay_buffer.batch_size,
                actions.cpu().data.numpy(),
                states_arr.cpu().data.numpy(),
                actions_arr.cpu().data.numpy(),
            )
        else:
            raise ValueError(f"Unknown opc_method: {self.opc_method}. Must be 'original', 'invertible', or 'none'")
        t_opc = time.time() - t0

        actions = get_tensor(actions)

        t0 = time.time()
        result = self._train(states, goals, actions, rewards, n_states, goals, not_done)
        t_train = time.time() - t0

        if not hasattr(self, '_timer_count'):
            self._timer_count = 0
            self._timer_opc_total = 0.0
            self._timer_train_total = 0.0

        self._timer_count += 1
        self._timer_opc_total += t_opc
        self._timer_train_total += t_train

        # Print every 100 high-level training steps
        if self._timer_count % 100 == 0:
            print(f"[High Controller] Avg over last 100: off_policy_corrections={self._timer_opc_total/100:.4f}s, _train={self._timer_train_total/100:.4f}s")
            self._timer_opc_total = 0.0
            self._timer_train_total = 0.0

        return result


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
        recon_weight=1.0,
        latent_weight=1.0,
        inverse_recon_weight=1.0,
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
        self.recon_weight = recon_weight
        self.latent_weight = latent_weight
        self.inverse_recon_weight = inverse_recon_weight

    def pretrain_reconstruction(self, replay_buffer):
        """Pre-train only the reconstruction losses of the invertible network."""
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, n_goals, rewards, not_done = (
            replay_buffer.sample()
        )

        # Forward pass through invertible actor
        a_full = self.actor(states, goals)

        # Compute only reconstruction losses (no RL loss)
        if self.is_invertible:
            # State reconstruction loss
            state_start = self.action_dim
            state_end = self.action_dim + self.state_dim
            reconstructed_state = a_full[:, state_start:state_end]
            state_reconstruction_loss = F.mse_loss(reconstructed_state, states)

            # Latent regularization loss: encourage N(0, 1) distribution
            latent_start = state_end
            latent_vars = a_full[:, latent_start:]

            # Compute batch statistics
            latent_mean = latent_vars.mean(dim=0)
            latent_std = latent_vars.std(dim=0)

            # Regularize to N(0, 1)
            latent_mean_loss = (latent_mean ** 2).mean()
            latent_std_loss = ((latent_std - 1.0) ** 2).mean()
            latent_regularization_loss = latent_mean_loss + latent_std_loss

            # Inverse reconstruction loss (cycle consistency)
            # Sample latent from N(0,1) to ensure robustness across the distribution
            latent_sampled = torch.randn_like(latent_vars)
            synthetic_output = torch.cat([
                a_full[:, :self.action_dim],
                states,
                latent_sampled
            ], dim=1)
            state_inv, goal_inv = self.actor.invertible_net.inverse(synthetic_output)
            # Only goal should reconstruct (state already handled by forward reconstruction)
            inverse_reconstruction_loss = F.mse_loss(goal_inv, goals)

            # Total reconstruction loss

            total_loss = (self.recon_weight * state_reconstruction_loss +
                         self.latent_weight * latent_regularization_loss +
                         self.inverse_recon_weight * inverse_reconstruction_loss)

            # Backprop and update
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()

            return {
                "pretrain_state_recon": state_reconstruction_loss.item(),
                "pretrain_latent_reg": latent_regularization_loss.item(),
                "pretrain_inverse_recon": inverse_reconstruction_loss.item(),
                "pretrain_total": total_loss.item()
            }
        else:
            return {}

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        states, sgoals, actions, n_states, n_sgoals, rewards, not_done = (
            replay_buffer.sample()
        )

        return self._train(
            states, sgoals, actions, rewards, n_states, n_sgoals, not_done
        )
