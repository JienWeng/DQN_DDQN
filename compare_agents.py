import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pickle
from train import train

def compare_agents(args):
    """
    Train and compare DQN and Double DQN on the same environment
    """
    # Check for GPU availability
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nComparing agents on device: {device}")
    
    # Initialize lists to store metrics for multiple seeds
    all_dqn_metrics = []
    all_ddqn_metrics = []
    
    # If using multiple seeds, show warning for long computation time
    if args.seeds > 1 and device.type != 'cuda':
        print("\n⚠️ Warning: Running multiple seeds without GPU will be slow.")
        print("Consider using Google Colab or Kaggle for faster training with GPU.")
        # Ask for confirmation
        if not args.force:
            response = input("Continue with multiple seeds? [y/N]: ")
            if response.lower() != 'y':
                print("Reducing to single seed.")
                args.seeds = 1
    
    # Track total time for comparison
    import time
    start_time = time.time()
    
    for seed_idx in range(args.seeds):
        # Set a different seed for each run
        current_seed = args.seed + seed_idx
        print(f"\n=== Running with seed {current_seed} ({seed_idx+1}/{args.seeds}) ===")
        
        # Make a copy of args to modify
        import copy
        seed_args = copy.deepcopy(args)
        seed_args.seed = current_seed
        
        # Set numpy and torch seeds
        import numpy as np
        import torch
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        # Train DQN
        print("\n" + "="*50)
        print(f"Training DQN on {args.env_name}")
        print("="*50)
        seed_args.agent_type = 'dqn'
        dqn_metrics = train(seed_args)
        all_dqn_metrics.append(dqn_metrics)
        
        # Train Double DQN
        print("\n" + "="*50)
        print(f"Training Double DQN on {args.env_name}")
        print("="*50)
        seed_args.agent_type = 'ddqn'
        ddqn_metrics = train(seed_args)
        all_ddqn_metrics.append(ddqn_metrics)
        
        # Save metrics for this seed
        os.makedirs('results/metrics', exist_ok=True)
        with open(f'results/metrics/dqn_{args.env_name}_{current_seed}.pkl', 'wb') as f:
            pickle.dump(dqn_metrics, f)
        with open(f'results/metrics/ddqn_{args.env_name}_{current_seed}.pkl', 'wb') as f:
            pickle.dump(ddqn_metrics, f)
    
    # Show total computation time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal computation time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    
    # For single-seed runs or to show results of the last seed
    dqn_metrics = all_dqn_metrics[-1]
    ddqn_metrics = all_ddqn_metrics[-1]
    
    # Compare results
    print("\n" + "="*50)
    print(f"Comparison results for {args.env_name}")
    print("="*50)
    
    # Configure plot styling for academic paper look
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 300
    })
    
    # Plot comparative metrics
    if args.plot_paper_style and args.seeds > 1:
        # Generate paper-style plots using multiple seeds
        from plot_paper_figure import generate_paper_figure
        generate_paper_figure(all_dqn_metrics, all_ddqn_metrics, args.env_name, 
                            f'results/paper_style_{args.env_name}.png')
    else:
        # Generate basic comparison plots from the last run
        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(dqn_metrics.episode_rewards, color='#FF7F0E', label='DQN', alpha=0.8)
        plt.plot(ddqn_metrics.episode_rewards, color='#1F77B4', label='Double DQN', alpha=0.8)
        plt.title(f'Episode Rewards: DQN vs Double DQN on {args.env_name}', fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.ylabel('Reward', fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/rewards_{args.env_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot value estimates and overestimation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Value estimates plot
        eval_points = range(len(dqn_metrics.estimated_values))
        
        ax1.plot(eval_points, dqn_metrics.estimated_values, color='#FF7F0E', 
                label='DQN Estimated', linewidth=2)
        ax1.axhline(y=np.mean(dqn_metrics.actual_values), color='#FF7F0E', linestyle='--', 
                   linewidth=1.5, label='DQN Actual Return')
        
        ax1.plot(eval_points, ddqn_metrics.estimated_values, color='#1F77B4', 
                label='Double DQN Estimated', linewidth=2)
        ax1.axhline(y=np.mean(ddqn_metrics.actual_values), color='#1F77B4', linestyle='--', 
                   linewidth=1.5, label='Double DQN Actual Return')
        
        ax1.set_title(f'Value Estimates on {args.env_name}', fontweight='bold')
        ax1.set_ylabel('Value Estimate', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Overestimation plot
        ax2.plot(eval_points, dqn_metrics.overoptimism_values, color='#FF7F0E', 
                label='DQN', linewidth=2)
        ax2.plot(eval_points, ddqn_metrics.overoptimism_values, color='#1F77B4', 
                label='Double DQN', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        ax2.set_title(f'Value Overestimation on {args.env_name}', fontweight='bold')
        ax2.set_xlabel('Evaluation Point (Episode)', fontweight='bold')
        ax2.set_ylabel('Estimated - Actual', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/value_comparison_{args.env_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Compute summary statistics
    dqn_final_reward = np.mean(dqn_metrics.episode_rewards[-100:])
    ddqn_final_reward = np.mean(ddqn_metrics.episode_rewards[-100:])
    
    dqn_avg_overopt = np.mean(dqn_metrics.overoptimism_values)
    ddqn_avg_overopt = np.mean(ddqn_metrics.overoptimism_values)
    
    print("\nFinal performance (last 100 episodes):")
    print(f"  DQN average reward: {dqn_final_reward:.2f}")
    print(f"  Double DQN average reward: {ddqn_final_reward:.2f}")
    print(f"  Improvement: {(ddqn_final_reward - dqn_final_reward):.2f} ({(ddqn_final_reward / max(1, dqn_final_reward) - 1) * 100:.2f}%)")
    
    print("\nOveroptimism analysis:")
    print(f"  DQN average overoptimism: {dqn_avg_overopt:.4f}")
    print(f"  Double DQN average overoptimism: {ddqn_avg_overopt:.4f}")
    if dqn_avg_overopt != 0:
        reduction_pct = (1 - ddqn_avg_overopt / dqn_avg_overopt) * 100
        print(f"  Reduction: {(dqn_avg_overopt - ddqn_avg_overopt):.4f} ({reduction_pct:.2f}%)")
    
    return all_dqn_metrics, all_ddqn_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare DQN and Double DQN on Atari games')
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', 
                        help='Atari environment name')
    parser.add_argument('--total_frames', type=int, default=1000000, 
                        help='Total number of frames for training (each agent)')
    parser.add_argument('--buffer_size', type=int, default=100000, 
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for optimization')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0, 
                        help='Starting epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.1, 
                        help='Final epsilon for exploration')
    parser.add_argument('--eps_decay', type=int, default=100000, 
                        help='Epsilon decay rate')
    parser.add_argument('--target_update', type=int, default=1000, 
                        help='Target network update frequency')
    parser.add_argument('--lr', type=float, default=0.00025, 
                        help='Learning rate')
    parser.add_argument('--eval_frequency', type=int, default=50000, 
                        help='Evaluation frequency in frames')
    parser.add_argument('--eval_episodes', type=int, default=10, 
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment during final evaluation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--seeds', type=int, default=1, 
                        help='Number of different seeds to run')
    parser.add_argument('--plot_paper_style', action='store_true', 
                        help='Generate paper-style plots using multiple seeds')
    parser.add_argument('--force', action='store_true', 
                        help='Force continuation without confirmation')
    
    args = parser.parse_args()
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    compare_agents(args)
