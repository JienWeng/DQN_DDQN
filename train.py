import gymnasium as gym
import numpy as np
import torch
import time
import os
import argparse
from collections import deque
from tqdm import tqdm, trange

from models.dqn import DQNAgent
from models.ddqn import DoubleDQNAgent
from utils.atari_wrappers import make_atari_env
from utils.metrics import MetricTracker
from utils.evaluation import evaluate_agent, measure_overoptimism

def train(args):
    """
    Train DQN or Double DQN agent based on episodes instead of frames
    """
    # First check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # For Colab/Kaggle, display more detailed GPU info if available
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1024**2:.0f} MB")
        print(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Create environment
    env = None
    eval_env = None
    
    # Check if this is an Atari environment or a classic control environment
    is_atari = any(env_type in args.env_name.lower() for env_type in ['breakout', 'pong', 'ale'])
    
    # Detect if we're using Gymnasium or classic Gym
    try:
        import gymnasium
        USING_GYMNASIUM = True
        print("Using Gymnasium API")
    except ImportError:
        USING_GYMNASIUM = False
        print("Using classic Gym API")
    
    if is_atari:
        # Create Atari environment
        from utils.atari_wrappers import make_atari_env
        env = make_atari_env(args.env_name, frame_stack=4, scale=True)
        eval_env = make_atari_env(args.env_name, frame_stack=4, scale=True)
    else:
        # Create a simple environment
        if USING_GYMNASIUM:
            import gymnasium as gym
            env = gym.make(args.env_name)
            eval_env = gym.make(args.env_name)
        else:
            import gym
            env = gym.make(args.env_name)
            eval_env = gym.make(args.env_name)
        
        # For simple environments, we need to handle the observation space differently
        input_shape = env.observation_space.shape
        if len(input_shape) == 1:
            # For 1D observation spaces (like CartPole), we need to reshape for CNNs
            # or use a different network architecture
            print("Non-image environment detected. Using MLP architecture instead of CNN.")
            # Here you would need to adjust the network architecture
            
    # Get input shape and number of actions
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Determine if it's an image-based environment or a classic control environment
    is_atari = len(input_shape) > 1 or (len(input_shape) == 3 and input_shape[2] > 1)
    
    print(f"Environment: {args.env_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Using {'image-based' if is_atari else 'vector-based'} network architecture")
    
    # Create agent
    if args.agent_type == 'dqn':
        agent = DQNAgent(
            input_shape=input_shape,
            n_actions=n_actions,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            target_update=args.target_update,
            learning_rate=args.lr
        )
    else:
        agent = DoubleDQNAgent(
            input_shape=input_shape,
            n_actions=n_actions,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            target_update=args.target_update,
            learning_rate=args.lr
        )
    
    # Initialize metrics tracker
    metrics = MetricTracker()
    
    # Start training based on episodes
    if USING_GYMNASIUM:
        state, _ = env.reset()
    else:
        state = env.reset()
        
    total_frames = 0
    episode_reward = 0
    episode_length = 0
    episode_loss = []
    episode_q_values = []
    
    print(f"Starting training {args.agent_type} on {args.env_name}...")
    print(f"Training for {args.total_episodes} episodes")
    
    # Create progress bar for episodes
    episode_pbar = trange(1, args.total_episodes + 1, desc=f"Training {args.agent_type}", 
                         unit="episode", ncols=100, colour="green")
    
    for episode in episode_pbar:
        # Reset for new episode
        if USING_GYMNASIUM:
            state, _ = env.reset()
        else:
            state = env.reset()
            
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        episode_q_values = []
        done = False
        
        # Episode loop
        frame_pbar = tqdm(range(1, args.max_frames_per_episode + 1), 
                          desc=f"Episode {episode}", leave=False, 
                          unit="frame", ncols=80, colour="blue")
        
        for frame in frame_pbar:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            if USING_GYMNASIUM:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state if not done else None, done)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            total_frames += 1
            
            # Optimize model
            loss = agent.optimize_model()
            if loss > 0:
                episode_loss.append(loss)
            
            # Get Q-values for current state
            q_values = agent.get_q_values(state)
            episode_q_values.append(np.max(q_values))
            
            # Update inner progress bar
            frame_pbar.set_postfix({
                'reward': f"{episode_reward:.1f}", 
                'q': f"{np.mean(episode_q_values):.2f}",
                'loss': f"{np.mean(episode_loss) if episode_loss else 0:.4f}"
            })
            
            # Move to next state
            state = next_state
            
            # If episode is done, break the loop
            if done:
                break
        
        frame_pbar.close()
                
        # Update target network every N episodes
        if episode % args.target_update == 0:
            agent.update_target_network()
            episode_pbar.write(f"Updated target network at episode {episode}")
        
        # Add episode metrics
        metrics.add_episode_reward(episode_reward)
        metrics.add_episode_length(episode_length)
        if episode_loss:
            metrics.add_loss(np.mean(episode_loss))
        if episode_q_values:
            metrics.add_q_value(np.mean(episode_q_values))
        
        # Update outer progress bar
        avg_reward = np.mean(metrics.episode_rewards[-100:]) if len(metrics.episode_rewards) >= 100 else np.mean(metrics.episode_rewards)
        episode_pbar.set_postfix({
            'reward': f"{episode_reward:.1f}", 
            'avg_reward': f"{avg_reward:.1f}",
            'frames': total_frames
        })
        
        # Evaluate agent and measure overoptimism periodically
        if episode % args.eval_frequency == 0:
            episode_pbar.write(f"\nEvaluating agent at episode {episode}...")
            eval_results = evaluate_agent(eval_env, agent, num_episodes=args.eval_episodes)
            overopt_results = measure_overoptimism(eval_env, agent, gamma=args.gamma, 
                                                num_episodes=args.eval_episodes)
            
            episode_pbar.write(f"Evaluation results:")
            episode_pbar.write(f"  Mean reward: {eval_results['mean_reward']:.2f}")
            episode_pbar.write(f"  Mean Q-value: {eval_results['mean_q_value']:.4f}")
            episode_pbar.write(f"  Estimated value: {overopt_results['estimated_value']:.4f}")
            episode_pbar.write(f"  Actual value: {overopt_results['actual_value']:.4f}")
            episode_pbar.write(f"  Overoptimism: {overopt_results['overoptimism']:.4f}")
            
            # Record overoptimism measurement
            metrics.add_value_comparison(
                overopt_results['estimated_value'],
                overopt_results['actual_value']
            )
            
            # Save model
            model_path = f"models/saved/{args.agent_type}_{args.env_name.split('/')[-1].split('-')[0]}_ep{episode}.pt"
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode,
                'total_frames': total_frames
            }, model_path)
            episode_pbar.write(f"Model saved to {model_path}")
    
    episode_pbar.close()
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = evaluate_agent(eval_env, agent, num_episodes=args.eval_episodes, render=args.render)
    overopt_results = measure_overoptimism(eval_env, agent, gamma=args.gamma, num_episodes=args.eval_episodes)
    
    print(f"Final evaluation results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"  Mean Q-value: {eval_results['mean_q_value']:.4f}")
    print(f"  Estimated value: {overopt_results['estimated_value']:.4f}")
    print(f"  Actual value: {overopt_results['actual_value']:.4f}")
    print(f"  Overoptimism: {overopt_results['overoptimism']:.4f}")
    
    # Save final model - fixing the AttributeError here
    final_model_path = f"models/saved/{args.agent_type}_{args.env_name.split('/')[-1]}_final.pt"
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': args.total_episodes,  # Use total_episodes instead of total_frames
        'total_frames': total_frames  # Still save the actual number of frames used
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Create directory for plots
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot metrics with academic styling
    metrics.plot_rewards(title=f"{args.agent_type} on {args.env_name} - Rewards", 
                         save_path=f"results/plots/{args.agent_type}_{args.env_name}_rewards.png")
    metrics.plot_q_values(title=f"{args.agent_type} on {args.env_name} - Q-Values",
                         save_path=f"results/plots/{args.agent_type}_{args.env_name}_qvalues.png")
    metrics.plot_losses(title=f"{args.agent_type} on {args.env_name} - Losses",
                       save_path=f"results/plots/{args.agent_type}_{args.env_name}_losses.png")
    metrics.plot_overoptimism(title=f"{args.agent_type} on {args.env_name} - Value Estimation",
                            save_path=f"results/plots/{args.agent_type}_{args.env_name}_overestimation.png")
    
    env.close()
    eval_env.close()
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN/DDQN on Atari games')
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', 
                        help='Atari environment name')
    parser.add_argument('--agent_type', type=str, default='dqn', choices=['dqn', 'ddqn'],
                        help='Agent type: DQN or Double DQN')
    parser.add_argument('--total_episodes', type=int, default=1000, 
                        help='Total number of episodes for training')
    parser.add_argument('--max_frames_per_episode', type=int, default=10000, 
                        help='Maximum number of frames per episode')
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
    parser.add_argument('--eps_decay', type=int, default=1000000, 
                        help='Epsilon decay rate')
    parser.add_argument('--target_update', type=int, default=10, 
                        help='Target network update frequency in episodes')
    parser.add_argument('--lr', type=float, default=0.00025, 
                        help='Learning rate')
    parser.add_argument('--eval_frequency', type=int, default=50, 
                        help='Evaluation frequency in episodes')
    parser.add_argument('--eval_episodes', type=int, default=10, 
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment during final evaluation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    train(args)
