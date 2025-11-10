from hiro.utils import _is_update
import numpy as np
import time
from .models import TD3Controller, HigherController, LowerController
from hiro.buffers import ReplayBuffer, LowReplayBuffer, HighReplayBuffer
from hiro.worker_goal_config import WorkerGoalConfig


class Agent:
    def __init__(self):
        self.worker_goal = tuple()

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        raise NotImplementedError

    def append(self, step, s, a, n_s, r, d):
        """
        Adds the given transition to the agent's replay buffer.
        """
        raise NotImplementedError

    def train(self, global_step):
        raise NotImplementedError

    def end_step(self):
        raise NotImplementedError

    def end_episode(self, episode, logger=None):
        raise NotImplementedError

    def evaluate_policy(
        self, env, eval_episodes=10, render=False, save_video=False, sleep=-1
    ):
        if save_video:
            env = RecordVideo(
                env,
                video_folder="video",
                episode_trigger=lambda x: True,
                disable_logger=True,
            )
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs["desired_goal"]
            s = obs["observation"]
            done = False
            reward_episode_sum = 0
            step = 0

            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep > 0:
                    time.sleep(sleep)

                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r

                s = n_s
                step += 1
                self.end_step()
            else:
                error = np.sqrt(np.sum(np.square(fg - s[:2])))
                subgoal_error = np.sqrt(np.sum(np.square(self.worker_goal[:2] - s[:2])))
                subgoal_total_error = np.sqrt(np.sum(np.square(self.worker_goal - s[:-1])))
                print(
                    "Goal, Curr, SG: (%5.2f, %5.2f), (%5.2f, %5.2f), (%5.2f, %5.2f)   Error:%5.2f  SG Err:%5.2f  SG Total Err:%5.2f"
                    % (
                        fg[0],
                        fg[1],
                        s[0],
                        s[1],
                        self.worker_goal[0],
                        self.worker_goal[1],
                        error,
                        subgoal_error,
                        subgoal_total_error
                    )
                )
                rewards.append(reward_episode_sum)
                success += 1 if error <= 5 else 0
                self.end_episode(e)

        env.evaluate = False
        return np.array(rewards), success / eval_episodes


class TD3Agent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        scale,
        model_path,
        model_save_freq,
        buffer_size,
        batch_size,
        start_training_steps,
    ):
        self.con = TD3Controller(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            scale=scale,
            model_path=model_path,
        )

        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        self.model_save_freq = model_save_freq
        self.start_training_steps = start_training_steps

    def step(self, s, env, step, global_step=0, explore=False):
        if explore:
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s)
        else:
            a = self._choose_action(s)

        obs, r, done, _ = env.step(a)
        n_s = obs["observation"]

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        self.replay_buffer.append(s, self.fg, a, n_s, r, d)

    def train(self, global_step):
        return self.con.train(self.replay_buffer)

    def _choose_action(self, s):
        return self.con.policy(s, self.fg)

    def _choose_action_with_noise(self, s):
        return self.con.policy_with_noise(s, self.fg)

    def end_step(self):
        pass

    def end_episode(self, episode, logger=None):
        if logger:
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

    def save(self, episode):
        self.con.save(episode)

    def load(self, episode):
        self.con.load(episode)


class HiroAgent(Agent):
    def __init__(
        self,
        observation_box,
        state_dim,
        action_dim,
        goal_dim,
        scale_low,
        start_training_steps,
        model_save_freq,
        model_path,
        buffer_size,
        batch_size,
        buffer_freq,
        train_freq,
        reward_scaling,
        policy_freq_high,
        policy_freq_low,
    ):
        self.worker_goal_config = WorkerGoalConfig(observation_box)

        self.model_save_freq = model_save_freq

        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            worker_goal_config=self.worker_goal_config,
            model_path=model_path,
            policy_freq=policy_freq_high,
        )

        self.low_con = LowerController(
            state_dim=state_dim,
            worker_goal_config=self.worker_goal_config,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low,
        )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            worker_goal_config=self.worker_goal_config,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            worker_goal_config=self.worker_goal_config,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq,
        )

        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0

        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.array([0, 0])
        self.sg = self.worker_goal_config.sample_goal()

        self.start_training_steps = start_training_steps

    def step(self, s, env, step, global_step=0, explore=False):
        ## Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        # Take action
        obs, r, done, _ = env.step(a)
        n_s = obs["observation"]

        ## Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_training_steps:
                n_sg = self.worker_goal_config.sample_goal()
                self.worker_goal = s[:-1] + n_sg
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
                self.worker_goal = s[:-1] + n_sg
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)

        self.n_sg = n_sg

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.replay_buffer_low.append(s, self.sg, a, n_s, self.n_sg, self.sr, float(d))

        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7]),
                )
            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]

        self.buf[3] += self.reward_scaling * r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def train(self, global_step):
        losses = {}
        td_errors = {}

        if global_step >= self.start_training_steps:
            loss, td_error = self.low_con.train(self.replay_buffer_low)
            losses.update(loss)
            td_errors.update(td_error)

            if global_step % self.train_freq == 0:
                loss, td_error = self.high_con.train(
                    self.replay_buffer_high, self.low_con
                )
                losses.update(loss)
                td_errors.update(td_error)

        return losses, td_errors

    def _choose_action_with_noise(self, s, sg):
        return self.low_con.policy_with_noise(s, sg)

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:  # Should be zero
            sg = self.high_con.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg)

    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
            self.worker_goal = s[:-1] + sg
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        return s[: sg.shape[0]] + sg - n_s[: sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        abs_s = s[: sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[: sg.shape[0]]) ** 2))

    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        if logger:
            # log
            logger.write("reward/Intrinsic Reward", self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def save(self, episode):
        self.low_con.save(episode)
        self.high_con.save(episode)

    def load(self, episode):
        self.low_con.load(episode)
        self.high_con.load(episode)
