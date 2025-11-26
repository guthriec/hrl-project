import os
import argparse
import numpy as np
import datetime
import copy
import csv
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env, create_eval_maze_env
from hiro.utils import Logger, _is_update, record_experience_to_csv, listdirs, set_device
from hiro.agents import HiroAgent, TD3Agent
import time


def run_evaluation(args, env, agent):
    agent.load(args.load_episode)

    rewards, success_rate = agent.evaluate_policy(
        env, args.eval_episodes, args.render, args.save_video, args.sleep
    )

    print(
        "mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}".format(
            mean=np.mean(rewards),
            std=np.std(rewards),
            median=np.median(rewards),
            success=success_rate,
        )
    )


class Trainer:
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path)

        # Initialize CSV logging for eval metrics
        self.eval_csv_path = os.path.join(log_path, 'eval_metrics.csv')
        with open(self.eval_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'success_rate', 'reward_mean', 'reward_std', 'reward_median'])
            writer.writeheader()

    def train(self):
        global_step = 0
        episode_times = []  # Track episode durations
        episode_steps = []  # Track steps per episode
        k = 100  # Window size for averaging

        for e in np.arange(self.args.num_episode) + 1:
            episode_start = time.time()

            obs = self.env.reset()
            fg = obs["desired_goal"]
            s = obs["observation"]
            done = False

            step = 0
            episode_reward = 0

            self.agent.set_final_goal(fg)

            # Track step-level timing
            if not hasattr(self, '_step_timers'):
                self._step_timers = {'step': 0.0, 'append': 0.0, 'train': 0.0, 'log': 0.0, 'other': 0.0}
                self._step_count = 0

            while not done:
                t0 = time.time()
                # Take action
                a, r, n_s, done, info = self.agent.step(
                    s, self.env, step, global_step, explore=True
                )
                self._step_timers['step'] += time.time() - t0

                t0 = time.time()
                # Append
                self.agent.append(step, s, a, n_s, r, done)
                self._step_timers['append'] += time.time() - t0

                # Pre-train reconstruction if we just hit start_training_steps
                if global_step == self.args.start_training_steps and self.args.pretrain_steps > 0:
                    if hasattr(self.agent, 'pretrain'):  # Only for HiroAgent
                        self.agent.pretrain(self.args.pretrain_steps)

                t0 = time.time()
                # Train
                losses, td_errors = self.agent.train(global_step)
                self._step_timers['train'] += time.time() - t0

                t0 = time.time()
                # Log
                self.log(global_step, [losses, td_errors])
                self._step_timers['log'] += time.time() - t0

                t0 = time.time()
                # Updates
                s = n_s
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()
                self._step_timers['other'] += time.time() - t0

                self._step_count += 1

            episode_duration = time.time() - episode_start
            episode_times.append(episode_duration)
            episode_steps.append(step)
            if len(episode_times) > k:
                episode_times.pop(0)
            if len(episode_steps) > k:
                episode_steps.pop(0)

            # Print timing info every 10 episodes
            if e % 10 == 0:
                avg_time = np.mean(episode_times)
                avg_steps = np.mean(episode_steps)
                print(f"Episode {e:05d}: Avg time/episode (last {len(episode_times)}): {avg_time:.3f}s, Avg steps/episode: {avg_steps:.1f}")
                if self._step_count > 0:
                    print(f"  [Step breakdown] step={self._step_timers['step']/self._step_count:.4f}s, append={self._step_timers['append']/self._step_count:.4f}s, train={self._step_timers['train']/self._step_count:.4f}s, log={self._step_timers['log']/self._step_count:.4f}s, other={self._step_timers['other']/self._step_count:.4f}s")
                    # Reset counters
                    self._step_timers = {'step': 0.0, 'append': 0.0, 'train': 0.0, 'log': 0.0, 'other': 0.0}
                    self._step_count = 0

            self.agent.end_episode(e, self.logger)
            self.logger.write("reward/Reward", episode_reward, e)
            self.evaluate(e)
            

    def log(self, global_step, data):
        losses, td_errors = data[0], data[1]

        # Logs
        if global_step >= self.args.start_training_steps and _is_update(
            global_step, args.writer_freq
        ):
            for k, v in losses.items():
                self.logger.write("loss/%s" % (k), v, global_step)

            for k, v in td_errors.items():
                self.logger.write("td_error/%s" % (k), v, global_step)

    def evaluate(self, e):
        # Print
        if _is_update(e, args.print_freq):
            agent = copy.deepcopy(self.agent)
            rewards, success_rate = agent.evaluate_policy(
                create_eval_maze_env(self.args.env, render_mode=("human" if self.args.render else None)),
                render=self.args.render,
                save_video=self.args.save_video,
                sleep=self.args.sleep,
            )
            # Log evaluation metrics to TensorBoard
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            reward_median = np.median(rewards)

            self.logger.write("eval/Success_Rate", success_rate, e)
            self.logger.write("eval/Reward_Mean", reward_mean, e)
            self.logger.write("eval/Reward_Std", reward_std, e)
            self.logger.write("eval/Reward_Median", reward_median, e)

            # Log evaluation metrics to CSV
            with open(self.eval_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode', 'success_rate', 'reward_mean', 'reward_std', 'reward_median'])
                writer.writerow({
                    'episode': e,
                    'success_rate': success_rate,
                    'reward_mean': reward_mean,
                    'reward_std': reward_std,
                    'reward_median': reward_median
                })

            print(
                "episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}".format(
                    episode=e,
                    mean=reward_mean,
                    std=reward_std,
                    median=reward_median,
                    success=success_rate,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--sleep", type=float, default=-1)
    parser.add_argument("--eval_episodes", type=float, default=5, help="Unit = Episode")
    parser.add_argument("--env", default="PointMazeEasy", type=str)
    parser.add_argument("--td3", action="store_true")
    parser.add_argument("--device", default="cpu", type=str, help="Device to use: 'cpu', 'cuda', or 'cuda:0', etc.")

    # Training
    parser.add_argument("--num_episode", default=4000, type=int)
    parser.add_argument(
        "--start_training_steps", default=2500, type=int, help="Unit = Global Step"
    )
    parser.add_argument(
        "--pretrain_steps", default=0, type=int, help="Number of steps to pre-train reconstruction losses (invertible net only)"
    )
    parser.add_argument(
        "--writer_freq", default=25, type=int, help="Unit = Global Step"
    )
    # Training (Model Saving)
    parser.add_argument("--load_episode", default=-1, type=int)
    parser.add_argument(
        "--model_save_freq", default=2000, type=int, help="Unit = Episodes"
    )
    parser.add_argument("--print_freq", default=100, type=int, help="Unit = Episode")
    parser.add_argument("--exp_name", default=None, type=str)
    # Model
    parser.add_argument("--model_path", default="model", type=str)
    parser.add_argument("--log_path", default="log", type=str)
    parser.add_argument("--policy_freq_low", default=2, type=int)
    parser.add_argument("--policy_freq_high", default=2, type=int)
    parser.add_argument("--opc_method", default="invertible", type=str,
                        choices=["original", "invertible", "none"],
                        help="Off-policy correction method: 'original' (HIRO), 'invertible' (new), or 'none' (no relabeling)")
    # Reconstruction loss weights (for invertible net)
    parser.add_argument("--recon_weight", default=1.0, type=float, help="Weight for state reconstruction loss")
    parser.add_argument("--latent_weight", default=1.0, type=float, help="Weight for latent regularization loss")
    parser.add_argument("--inverse_recon_weight", default=1.0, type=float, help="Weight for inverse reconstruction loss")
    # Replay Buffer
    parser.add_argument("--buffer_size", default=200000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--buffer_freq", default=10, type=int)
    parser.add_argument("--train_freq", default=10, type=int)
    parser.add_argument("--reward_scaling", default=0.1, type=float)
    args = parser.parse_args()

    # Set device for training
    set_device(args.device)

    # Select or Generate a name for this experiment
    if args.exp_name:
        experiment_name = args.exp_name
    else:
        if args.eval:
            # choose most updated experiment for evaluation
            dirs_str = listdirs(args.model_path)
            dirs = np.array(list(map(int, dirs_str)))
            experiment_name = dirs_str[np.argmax(dirs)]
        else:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(experiment_name)

    # Environment and its attributes
    env, obs_box = None, None
    if args.env != "SimpleCarMaze":
        env = EnvWithGoal(create_maze_env(args.env, render_mode=("human" if args.render else None)), args.env)
        obs_box = env.observation_space
    else:
        env = create_maze_env(args.env, render_mode=("human" if args.render else None))
        obs_box = env.observation_space["observation"]
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    # Spawn an agent
    if args.td3:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale=scale,
            model_save_freq=args.model_save_freq,
            model_path=os.path.join(args.model_path, experiment_name),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            start_training_steps=args.start_training_steps,
        )
    else:
        agent = HiroAgent(
            observation_box=obs_box,
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale_low=scale,
            start_training_steps=args.start_training_steps,
            model_path=os.path.join(args.model_path, experiment_name),
            model_save_freq=args.model_save_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            buffer_freq=args.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=args.reward_scaling,
            policy_freq_high=args.policy_freq_high,
            policy_freq_low=args.policy_freq_low,
            opc_method=args.opc_method,
            recon_weight=args.recon_weight,
            latent_weight=args.latent_weight,
            inverse_recon_weight=args.inverse_recon_weight,
        )

    # Run training or evaluation
    if args.train:
        # Record this experiment with arguments to a CSV file
        record_experience_to_csv(args, experiment_name)
        # Start training
        trainer = Trainer(args, env, agent, experiment_name)
        trainer.train()
    if args.eval:
        run_evaluation(args, env, agent)
