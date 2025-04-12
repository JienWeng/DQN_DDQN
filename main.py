import argparse
import os
from train import train
from compare_agents import compare_agents

def main():
    parser = argparse.ArgumentParser(description='Train and compare DQN and Double DQN on Atari games')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train a single agent')
    train_parser.add_argument('--env_name', type=str, default='CartPole-v1', 
                              help='Environment name (Atari or classic control)')
    train_parser.add_argument('--agent_type', type=str, default='dqn', choices=['dqn', 'ddqn'],
                              help='Agent type: DQN or Double DQN')
    train_parser.add_argument('--total_episodes', type=int, default=500, 
                              help='Total number of episodes for training')
    train_parser.add_argument('--max_frames_per_episode', type=int, default=10000, 
                              help='Maximum number of frames per episode')
    train_parser.add_argument('--buffer_size', type=int, default=100000, 
                              help='Replay buffer size')
    train_parser.add_argument('--batch_size', type=int, default=32, 
                              help='Batch size for optimization')
    train_parser.add_argument('--gamma', type=float, default=0.99, 
                              help='Discount factor')
    train_parser.add_argument('--eps_start', type=float, default=1.0, 
                              help='Starting epsilon for exploration')
    train_parser.add_argument('--eps_end', type=float, default=0.1, 
                              help='Final epsilon for exploration')
    train_parser.add_argument('--eps_decay', type=int, default=500, 
                              help='Epsilon decay rate (in episodes)')
    train_parser.add_argument('--target_update', type=int, default=10, 
                              help='Target network update frequency (in episodes)')
    train_parser.add_argument('--lr', type=float, default=0.00025, 
                              help='Learning rate')
    train_parser.add_argument('--eval_frequency', type=int, default=50, 
                              help='Evaluation frequency in episodes')
    train_parser.add_argument('--eval_episodes', type=int, default=10, 
                              help='Number of episodes for evaluation')
    train_parser.add_argument('--render', action='store_true', 
                              help='Render the environment during final evaluation')
    train_parser.add_argument('--seed', type=int, default=42, 
                              help='Random seed')
    train_parser.add_argument('--force', action='store_true',
                              help='Force execution without confirmation prompts')
    train_parser.add_argument('--no_cuda', action='store_true',
                              help='Disable CUDA even if available')
    train_parser.add_argument('--verbose', action='store_true',
                              help='Enable verbose output')
    train_parser.add_argument('--quiet', action='store_true',
                              help='Reduce output to minimum')
    
    # Compare parser
    compare_parser = subparsers.add_parser('compare', help='Compare DQN and Double DQN')
    compare_parser.add_argument('--env_name', type=str, default='CartPole-v1', 
                                help='Environment name (Atari or classic control)')
    compare_parser.add_argument('--total_episodes', type=int, default=500, 
                                help='Total number of episodes for training')
    compare_parser.add_argument('--max_frames_per_episode', type=int, default=10000, 
                                help='Maximum number of frames per episode')
    compare_parser.add_argument('--buffer_size', type=int, default=100000, 
                                help='Replay buffer size')
    compare_parser.add_argument('--batch_size', type=int, default=32, 
                                help='Batch size for optimization')
    compare_parser.add_argument('--gamma', type=float, default=0.99, 
                                help='Discount factor')
    compare_parser.add_argument('--eps_start', type=float, default=1.0, 
                                help='Starting epsilon for exploration')
    compare_parser.add_argument('--eps_end', type=float, default=0.1, 
                                help='Final epsilon for exploration')
    compare_parser.add_argument('--eps_decay', type=int, default=500, 
                                help='Epsilon decay rate (in episodes)')
    compare_parser.add_argument('--target_update', type=int, default=10, 
                                help='Target network update frequency (in episodes)')
    compare_parser.add_argument('--lr', type=float, default=0.00025, 
                                help='Learning rate')
    compare_parser.add_argument('--eval_frequency', type=int, default=50, 
                                help='Evaluation frequency in episodes')
    compare_parser.add_argument('--eval_episodes', type=int, default=10, 
                                help='Number of episodes for evaluation')
    compare_parser.add_argument('--render', action='store_true', 
                                help='Render the environment during final evaluation')
    compare_parser.add_argument('--seeds', type=int, default=1, 
                                help='Number of random seeds to use for each algorithm')
    compare_parser.add_argument('--plot_paper_style', action='store_true',
                                help='Generate paper-quality plots similar to those in the Double DQN paper')
    compare_parser.add_argument('--seed', type=int, default=42, 
                                help='Base random seed')
    compare_parser.add_argument('--force', action='store_true',
                                help='Force execution without confirmation prompts')
    compare_parser.add_argument('--no_cuda', action='store_true',
                                help='Disable CUDA even if available')
    compare_parser.add_argument('--verbose', action='store_true',
                                help='Enable verbose output')
    compare_parser.add_argument('--quiet', action='store_true',
                                help='Reduce output to minimum')
    
    # Parse arguments first
    args = parser.parse_args()
    
    # Set CUDA settings before anything else
    if args.command and hasattr(args, 'no_cuda') and args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA entirely
        print("CUDA has been disabled by command line argument.")
    
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'compare':
        compare_agents(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
