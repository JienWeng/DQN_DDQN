import numpy as np
import torch
import time

# Check if we're using Gymnasium or classic Gym
try:
    import gymnasium
    USING_GYMNASIUM = True
except ImportError:
    USING_GYMNASIUM = False

def evaluate_agent(env, agent, num_episodes=10, render=False, quiet=False):
    """
    Evaluate agent's performance on the environment
    """
    episode_rewards = []
    episode_lengths = []
    q_values = []
    
    for i in range(num_episodes):
        if USING_GYMNASIUM:
            state, _ = env.reset()
        else:
            state = env.reset()
            
        episode_reward = 0
        episode_length = 0
        done = False
        episode_q_vals = []
        
        while not done:
            if render:
                env.render()
                time.sleep(0.02)
                
            # Get Q-values and select action with small epsilon
            q_vals = agent.get_q_values(state)
            action = agent.select_action(state, evaluate=True)
            episode_q_vals.append(np.max(q_vals))
            
            if USING_GYMNASIUM:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, _ = env.step(action)
                
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        q_values.append(np.mean(episode_q_vals))
    
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_q = np.mean(q_values)
    
    # Only print if not in quiet mode
    if not quiet:
        print(f"Evaluation over {num_episodes} episodes:")
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Mean length: {mean_length:.2f}")
        print(f"  Mean Q-value: {mean_q:.4f}")
    
    return {
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'mean_q_value': mean_q,
        'rewards': episode_rewards
    }

def compute_actual_values(env, agent, gamma=0.99, num_episodes=10):
    """
    Compute the actual discounted returns for the current policy
    """
    actual_values = []
    
    # Check if we're using Gymnasium or classic Gym
    try:
        import gymnasium
        USING_GYMNASIUM = True
    except ImportError:
        USING_GYMNASIUM = False
    
    for _ in range(num_episodes):
        try:
            # Reset the environment
            if USING_GYMNASIUM:
                state, _ = env.reset()
            else:
                state = env.reset()
                
            done = False
            rewards = []
            
            # Perform rollout
            while not done:
                try:
                    action = agent.select_action(state, evaluate=True)
                    
                    if USING_GYMNASIUM:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = env.step(action)
                        
                    rewards.append(reward)
                    state = next_state
                except Exception as e:
                    print(f"Error during rollout step: {e}")
                    done = True
            
            # Calculate discounted return (actual value)
            if rewards:
                discounted_return = 0
                for r in reversed(rewards):
                    discounted_return = r + gamma * discounted_return
                    
                actual_values.append(discounted_return)
        except Exception as e:
            print(f"Error in episode rollout: {e}")
    
    # Handle edge case: no valid actual values
    if not actual_values:
        print("Warning: No valid actual values collected. Using zero.")
        return 0.0
        
    return np.mean(actual_values)

def measure_overoptimism(env, agent, gamma=0.99, num_episodes=10):
    """
    Measure the overoptimism by comparing estimated and actual values
    """
    estimated_values = []
    
    # Check if we're using Gymnasium or classic Gym
    try:
        import gymnasium
        USING_GYMNASIUM = True
    except ImportError:
        USING_GYMNASIUM = False
    
    for _ in range(num_episodes):
        try:
            # Reset the environment
            if USING_GYMNASIUM:
                state, _ = env.reset()
            else:
                state = env.reset()
            
            # The estimated value is the maximum Q-value at the initial state
            try:
                q_vals = agent.get_q_values(state)
                if q_vals is not None and q_vals.size > 0:
                    estimated_values.append(np.max(q_vals))
                else:
                    print("Warning: get_q_values returned empty array")
            except Exception as e:
                print(f"Error in getting Q-values: {e}")
        except Exception as e:
            print(f"Error in environment reset: {e}")
            
    # Handle edge case: no valid estimated values
    if not estimated_values:
        print("Warning: No valid estimated values collected. Using zero.")
        estimated_value = 0.0
    else:
        estimated_value = np.mean(estimated_values)
    
    # Get actual value through rollouts
    try:
        actual_value = compute_actual_values(env, agent, gamma, num_episodes)
    except Exception as e:
        print(f"Error computing actual values: {e}")
        actual_value = 0.0
    
    return {
        'estimated_value': estimated_value,
        'actual_value': actual_value,
        'overoptimism': estimated_value - actual_value
    }
