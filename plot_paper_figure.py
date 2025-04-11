"""
Generate paper-quality figures for comparing DQN and Double DQN,
similar to those in the Double DQN paper by van Hasselt et al.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import os
import pickle
import argparse

def smooth_data(data, window_size=5):
    """Apply moving average smoothing"""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def generate_paper_figure(dqn_metrics_list, ddqn_metrics_list, env_name, save_path=None):
    """
    Generate a figure similar to those in the Double DQN paper
    
    Args:
        dqn_metrics_list: List of MetricTracker objects for DQN
        ddqn_metrics_list: List of MetricTracker objects for Double DQN
        env_name: Environment name
        save_path: Path to save the figure
    """
    # Set the style for academic paper
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 12),
        'figure.dpi': 300
    })
    
    # Create a figure with grid layout: 3 rows (top, middle, bottom)
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # Top row: Value estimates with actual return horizontal lines
    ax_top = fig.add_subplot(gs[0])
    
    # Extract data
    dqn_estimated = [np.array(m.estimated_values) for m in dqn_metrics_list]
    dqn_actual = [np.array(m.actual_values) for m in dqn_metrics_list]
    ddqn_estimated = [np.array(m.estimated_values) for m in ddqn_metrics_list]
    ddqn_actual = [np.array(m.actual_values) for m in ddqn_metrics_list]
    
    # Calculate median and quantiles for DQN
    min_points = min([len(d) for d in dqn_estimated])
    dqn_data = np.zeros((len(dqn_estimated), min_points))
    for i, d in enumerate(dqn_estimated):
        dqn_data[i, :] = d[:min_points]
    
    dqn_median = np.median(dqn_data, axis=0)
    dqn_low = np.percentile(dqn_data, 10, axis=0)
    dqn_high = np.percentile(dqn_data, 90, axis=0)
    dqn_actual_mean = np.mean([np.mean(a) for a in dqn_actual])
    
    # Calculate median and quantiles for Double DQN
    min_points = min([len(d) for d in ddqn_estimated])
    ddqn_data = np.zeros((len(ddqn_estimated), min_points))
    for i, d in enumerate(ddqn_estimated):
        ddqn_data[i, :] = d[:min_points]
    
    ddqn_median = np.median(ddqn_data, axis=0)
    ddqn_low = np.percentile(ddqn_data, 10, axis=0)
    ddqn_high = np.percentile(ddqn_data, 90, axis=0)
    ddqn_actual_mean = np.mean([np.mean(a) for a in ddqn_actual])
    
    # Plot data
    eval_points = np.arange(min_points)
    ax_top.plot(eval_points, dqn_median, color='#FF7F0E', linewidth=2, label='DQN')
    ax_top.fill_between(eval_points, dqn_low, dqn_high, color='#FF7F0E', alpha=0.2)
    ax_top.axhline(y=dqn_actual_mean, color='#FF7F0E', linestyle='--', linewidth=1.5)
    
    ax_top.plot(eval_points, ddqn_median, color='#1F77B4', linewidth=2, label='Double DQN')
    ax_top.fill_between(eval_points, ddqn_low, ddqn_high, color='#1F77B4', alpha=0.2)
    ax_top.axhline(y=ddqn_actual_mean, color='#1F77B4', linestyle='--', linewidth=1.5)
    
    ax_top.set_title(f'Value Estimates on {env_name}', fontweight='bold')
    ax_top.set_xlabel('Evaluation Episode', fontweight='bold')
    ax_top.set_ylabel('Value Estimate', fontweight='bold')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, linestyle='--', alpha=0.7)
    
    # Middle row: Log scale value estimates
    ax_middle = fig.add_subplot(gs[1])
    
    # Use log scale for middle plot (if values are positive)
    if np.all(dqn_median > 0) and np.all(ddqn_median > 0):
        ax_middle.plot(eval_points, dqn_median, color='#FF7F0E', linewidth=2, label='DQN')
        ax_middle.fill_between(eval_points, dqn_low, dqn_high, color='#FF7F0E', alpha=0.2)
        
        ax_middle.plot(eval_points, ddqn_median, color='#1F77B4', linewidth=2, label='Double DQN')
        ax_middle.fill_between(eval_points, ddqn_low, ddqn_high, color='#1F77B4', alpha=0.2)
        
        ax_middle.set_yscale('log')
        ax_middle.yaxis.set_major_formatter(ScalarFormatter())
    else:
        # If there are non-positive values, show overoptimism directly
        dqn_overopt = np.array([np.array(m.overoptimism_values)[:min_points] for m in dqn_metrics_list])
        ddqn_overopt = np.array([np.array(m.overoptimism_values)[:min_points] for m in ddqn_metrics_list])
        
        dqn_overopt_median = np.median(dqn_overopt, axis=0)
        dqn_overopt_low = np.percentile(dqn_overopt, 10, axis=0)
        dqn_overopt_high = np.percentile(dqn_overopt, 90, axis=0)
        
        ddqn_overopt_median = np.median(ddqn_overopt, axis=0)
        ddqn_overopt_low = np.percentile(ddqn_overopt, 10, axis=0)
        ddqn_overopt_high = np.percentile(ddqn_overopt, 90, axis=0)
        
        ax_middle.plot(eval_points, dqn_overopt_median, color='#FF7F0E', linewidth=2, label='DQN')
        ax_middle.fill_between(eval_points, dqn_overopt_low, dqn_overopt_high, color='#FF7F0E', alpha=0.2)
        
        ax_middle.plot(eval_points, ddqn_overopt_median, color='#1F77B4', linewidth=2, label='Double DQN')
        ax_middle.fill_between(eval_points, ddqn_overopt_low, ddqn_overopt_high, color='#1F77B4', alpha=0.2)
        
        ax_middle.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    ax_middle.set_title(f'{"Value Estimates (Log Scale)" if np.all(dqn_median > 0) and np.all(ddqn_median > 0) else "Overestimation"} on {env_name}', 
                      fontweight='bold')
    ax_middle.set_xlabel('Evaluation Episode', fontweight='bold')
    ax_middle.set_ylabel('Value Estimate (log)' if np.all(dqn_median > 0) and np.all(ddqn_median > 0) else 'Overestimation (Est. - Act.)', 
                       fontweight='bold')
    ax_middle.legend(loc='upper left')
    ax_middle.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom row: Score achieved during training
    ax_bottom = fig.add_subplot(gs[2])
    
    # Extract reward data
    dqn_rewards = [np.array(m.episode_rewards) for m in dqn_metrics_list]
    ddqn_rewards = [np.array(m.episode_rewards) for m in ddqn_metrics_list]
    
    # Find minimum length across all runs
    min_episodes = min([len(r) for r in dqn_rewards + ddqn_rewards])
    
    # Calculate median and quantiles for DQN rewards
    dqn_reward_data = np.zeros((len(dqn_rewards), min_episodes))
    for i, r in enumerate(dqn_rewards):
        dqn_reward_data[i, :] = r[:min_episodes]
    
    dqn_reward_median = np.median(dqn_reward_data, axis=0)
    dqn_reward_low = np.percentile(dqn_reward_data, 10, axis=0)
    dqn_reward_high = np.percentile(dqn_reward_data, 90, axis=0)
    
    # Calculate median and quantiles for Double DQN rewards
    ddqn_reward_data = np.zeros((len(ddqn_rewards), min_episodes))
    for i, r in enumerate(ddqn_rewards):
        ddqn_reward_data[i, :] = r[:min_episodes]
    
    ddqn_reward_median = np.median(ddqn_reward_data, axis=0)
    ddqn_reward_low = np.percentile(ddqn_reward_data, 10, axis=0)
    ddqn_reward_high = np.percentile(ddqn_reward_data, 90, axis=0)
    
    # Apply smoothing
    window_size = min(10, min_episodes // 10)  # Adaptive window size
    if min_episodes > window_size * 2:
        dqn_reward_median_smooth = smooth_data(dqn_reward_median, window_size)
        dqn_reward_low_smooth = smooth_data(dqn_reward_low, window_size)
        dqn_reward_high_smooth = smooth_data(dqn_reward_high, window_size)
        ddqn_reward_median_smooth = smooth_data(ddqn_reward_median, window_size)
        ddqn_reward_low_smooth = smooth_data(ddqn_reward_low, window_size)
        ddqn_reward_high_smooth = smooth_data(ddqn_reward_high, window_size)
        
        smoothed_episodes = np.arange(len(dqn_reward_median_smooth))
        
        ax_bottom.plot(smoothed_episodes, dqn_reward_median_smooth, color='#FF7F0E', linewidth=2, label='DQN')
        ax_bottom.fill_between(smoothed_episodes, dqn_reward_low_smooth, dqn_reward_high_smooth, color='#FF7F0E', alpha=0.2)
        
        ax_bottom.plot(smoothed_episodes, ddqn_reward_median_smooth, color='#1F77B4', linewidth=2, label='Double DQN')
        ax_bottom.fill_between(smoothed_episodes, ddqn_reward_low_smooth, ddqn_reward_high_smooth, color='#1F77B4', alpha=0.2)
    else:
        # If there's not enough data for smoothing, just plot the raw data
        episodes = np.arange(min_episodes)
        ax_bottom.plot(episodes, dqn_reward_median, color='#FF7F0E', linewidth=2, label='DQN')
        ax_bottom.fill_between(episodes, dqn_reward_low, dqn_reward_high, color='#FF7F0E', alpha=0.2)
        
        ax_bottom.plot(episodes, ddqn_reward_median, color='#1F77B4', linewidth=2, label='Double DQN')
        ax_bottom.fill_between(episodes, ddqn_reward_low, ddqn_reward_high, color='#1F77B4', alpha=0.2)
    
    ax_bottom.set_title(f'Score During Training on {env_name}', fontweight='bold')
    ax_bottom.set_xlabel('Episode', fontweight='bold')
    ax_bottom.set_ylabel('Score', fontweight='bold')
    ax_bottom.legend(loc='upper left')
    ax_bottom.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("\nValue Estimation Statistics:")
    print(f"  DQN mean estimated: {np.mean(dqn_median):.2f}")
    print(f"  DQN mean actual: {dqn_actual_mean:.2f}")
    print(f"  DQN mean overestimation: {np.mean(dqn_median) - dqn_actual_mean:.2f}")
    print(f"  Double DQN mean estimated: {np.mean(ddqn_median):.2f}")
    print(f"  Double DQN mean actual: {ddqn_actual_mean:.2f}")
    print(f"  Double DQN mean overestimation: {np.mean(ddqn_median) - ddqn_actual_mean:.2f}")
    print(f"  Overestimation reduction: {((np.mean(dqn_median) - dqn_actual_mean) - (np.mean(ddqn_median) - ddqn_actual_mean)):.2f}")

def load_metrics_files(env_name, metrics_dir='results/metrics'):
    """Load all available metrics files for a given environment"""
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory not found: {metrics_dir}")
        return [], []
    
    # Find DQN and DDQN metrics files
    dqn_files = [f for f in os.listdir(metrics_dir) if f.startswith(f'dqn_{env_name}') and f.endswith('.pkl')]
    ddqn_files = [f for f in os.listdir(metrics_dir) if f.startswith(f'ddqn_{env_name}') and f.endswith('.pkl')]
    
    # Load metrics
    dqn_metrics = []
    for f in dqn_files:
        try:
            with open(os.path.join(metrics_dir, f), 'rb') as file:
                dqn_metrics.append(pickle.load(file))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    ddqn_metrics = []
    for f in ddqn_files:
        try:
            with open(os.path.join(metrics_dir, f), 'rb') as file:
                ddqn_metrics.append(pickle.load(file))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return dqn_metrics, ddqn_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate paper-quality figures for DQN vs Double DQN')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', 
                        help='Environment name')
    parser.add_argument('--metrics_dir', type=str, default='results/metrics', 
                        help='Directory containing saved metrics files')
    parser.add_argument('--save_dir', type=str, default='results/paper_figures', 
                        help='Directory to save generated figures')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load metrics files
    dqn_metrics_list, ddqn_metrics_list = load_metrics_files(args.env_name, args.metrics_dir)
    
    if not dqn_metrics_list or not ddqn_metrics_list:
        print(f"Error: No metrics files found for {args.env_name}")
        exit(1)
    
    print(f"Found {len(dqn_metrics_list)} DQN runs and {len(ddqn_metrics_list)} Double DQN runs")
    
    # Generate figure
    save_path = os.path.join(args.save_dir, f'{args.env_name}_paper_figure.png')
    generate_paper_figure(dqn_metrics_list, ddqn_metrics_list, args.env_name, save_path)
    
    print(f"Figure saved to {save_path}")
