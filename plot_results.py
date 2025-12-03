import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define experiment groups
experiments = {
    'Ellipsoid with max radius 16': 'gc-16.0-12',
    'Ellipsoid with max radius 8': 'gc-8.0-12',
    'Ellipsoid with max radius 4': 'gc-4.0-12',
    'Ellipsoid with max radius 2': 'gc-2.0-12',
    'Ellipsoid with max radius 1': 'gc-1.0-12',
    'No OPC': 'no-correction-12',
}

log_dir = Path('log')
num_runs = 2

def load_experiment_data(exp_prefix, num_runs):
    """Load and aggregate data from multiple runs of the same experiment."""
    all_runs = []

    for i in range(1, num_runs + 1):
        exp_name = f"{exp_prefix}-{i}"
        csv_path = log_dir / exp_name / 'eval_metrics.csv'

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_runs.append(df)
        else:
            print(f"Warning: {csv_path} not found")

    if not all_runs:
        return None

    # Align all runs to the same episodes (use intersection of all episodes)
    episodes = set(all_runs[0]['episode'])
    for df in all_runs[1:]:
        episodes = episodes.intersection(set(df['episode']))
    episodes = sorted(list(episodes))

    # Extract data for common episodes
    success_rates = []
    reward_means = []

    for df in all_runs:
        df_filtered = df[df['episode'].isin(episodes)].sort_values('episode')
        success_rates.append(df_filtered['success_rate'].values)
        reward_means.append(df_filtered['reward_mean'].values)

    # Convert to numpy arrays
    success_rates = np.array(success_rates)  # (num_runs, num_episodes)
    reward_means = np.array(reward_means)

    # Compute mean and std across runs
    success_mean = np.mean(success_rates, axis=0)
    success_std = np.std(success_rates, axis=0)
    reward_mean = np.mean(reward_means, axis=0)
    reward_std = np.std(reward_means, axis=0)

    return {
        'episodes': np.array(episodes),
        'success_mean': success_mean,
        'success_std': success_std,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'num_runs': len(all_runs)
    }

# Load all experiments
results = {}
for name, prefix in experiments.items():
    print(f"Loading {name}...")
    data = load_experiment_data(prefix, num_runs)
    if data:
        results[name] = data
        print(f"  Loaded {data['num_runs']} runs")

# Create plots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# Plot 1: Success Rate
# for name, data in results.items():
#     episodes = data['episodes']
#     mean = data['success_mean']
#     std = data['success_std']
#
#     ax1.plot(episodes, mean, label=name, linewidth=2)
#     ax1.fill_between(episodes, mean - std, mean + std, alpha=0.2)
#
# ax1.set_xlabel('Episode', fontsize=12)
# ax1.set_ylabel('Success Rate', fontsize=12)
# ax1.set_title('Evaluation Success Rate', fontsize=14, fontweight='bold')
# ax1.legend(fontsize=10)
# ax1.grid(True, alpha=0.3)
# ax1.set_ylim([0, 1])

# Plot 2: Reward
for name, data in results.items():
    episodes = data['episodes']
    mean = data['reward_mean']
    std = data['reward_std']

    ax2.plot(episodes, mean, label=name, linewidth=2)
    ax2.fill_between(episodes, mean - std, mean + std, alpha=0.2)

ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Mean Reward', fontsize=12)
ax2.set_title('Evaluation Mean Reward', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'comparison.png'")
plt.show()

# Print final performance summary
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY (last episode)")
print("="*60)
for name, data in results.items():
    final_success = data['success_mean'][-1]
    final_success_std = data['success_std'][-1]
    final_reward = data['reward_mean'][-1]
    final_reward_std = data['reward_std'][-1]

    print(f"\n{name}:")
    print(f"  Success Rate: {final_success:.3f} ± {final_success_std:.3f}")
    print(f"  Mean Reward:  {final_reward:.3f} ± {final_reward_std:.3f}")
