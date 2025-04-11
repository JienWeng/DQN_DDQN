"""
Test script to verify Atari environment setup
"""
import os
import sys
import time
from utils.atari_wrappers import make_atari_env, USING_GYMNASIUM

def test_environments():
    """Test various environment formats to see which ones work"""
    print("\n=== Testing Atari Environment Setup ===")
    print(f"Using {'Gymnasium' if USING_GYMNASIUM else 'Classic Gym'}")
    
    # List of environment IDs to try
    env_ids = [
        "Breakout-v0",
        "BreakoutNoFrameskip-v4",
        "ALE/Breakout-v5",
        "Pong-v0",
        "ALE/Pong-v5"
    ]
    
    working_envs = []
    
    for env_id in env_ids:
        print(f"\nTrying to create environment: {env_id}")
        try:
            env = make_atari_env(env_id, frame_stack=4, scale=True)
            print(f"✓ SUCCESS: Created environment {env_id}")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
            
            # Test reset and step
            print("  Testing reset and step...")
            obs = env.reset()
            action = env.action_space.sample()
            
            if USING_GYMNASIUM:
                next_obs, reward, terminated, truncated, info = env.step(action)
                print(f"  Step result - reward: {reward}, terminated: {terminated}, truncated: {truncated}")
            else:
                next_obs, reward, done, info = env.step(action)
                print(f"  Step result - reward: {reward}, done: {done}")
            
            env.close()
            working_envs.append(env_id)
        except Exception as e:
            print(f"✗ FAILED: Could not create environment {env_id}")
            print(f"  Error: {e}")
    
    print("\n=== Environment Test Results ===")
    if working_envs:
        print("Working environments:")
        for env_id in working_envs:
            print(f"  - {env_id}")
        print(f"\nRecommended environment to use: {working_envs[0]}")
    else:
        print("No environments are working. Please check your installation.")
        print("Try running: pip install 'gymnasium[atari,accept-rom-license]'")
    
    return working_envs

if __name__ == "__main__":
    working_envs = test_environments()
    
    if not working_envs:
        sys.exit(1)
    
    # Demonstrate a working environment
    print("\n=== Demonstrating Working Environment ===")
    env_id = working_envs[0]
    print(f"Using environment: {env_id}")
    
    env = make_atari_env(env_id)
    obs = env.reset()
    
    print("Running 100 random actions...")
    for _ in range(100):
        action = env.action_space.sample()
        
        if USING_GYMNASIUM:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished!")
            obs = env.reset()
    
    env.close()
    print("Environment test completed successfully!")
    
    print("\nTo use this environment in your main code, use:")
    print(f"python main.py train --env_name {env_id} --agent_type dqn --total_frames 1000000")
