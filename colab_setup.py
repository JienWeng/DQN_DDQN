"""
Setup script for running the DQN project in Google Colab
"""
import os
import sys
import subprocess
import torch
import IPython

def setup_colab():
    """Set up the environment for Google Colab"""
    print("Setting up environment for Google Colab...")
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
        print("Not running in Colab. Setup aborted.")
        return False
    
    # Install required packages
    print("\nInstalling required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "torch", "torchvision", "gymnasium", 
                           "gymnasium[atari]", "gymnasium[accept-rom-license]",
                           "matplotlib", "numpy", "opencv-python", "tqdm"])
    
    # Clone the repository if not already present
    if not os.path.exists('DQN'):
        print("\nCloning the repository...")
        subprocess.check_call(["git", "clone", "https://github.com/YourGithubUsername/DQN.git"])
        os.chdir('DQN')
    
    # Check GPU availability
    print("\nChecking GPU availability...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"✓ GPU is available: {gpu_props.name}")
        print(f"  Total memory: {gpu_props.total_memory / 1024**2:.0f} MB")
        print(f"  CUDA capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        device = torch.device("cpu")
        print("✗ GPU is not available, using CPU instead")
    
    # Print Colab-specific instructions
    print("\nColab Setup Complete!")
    print("\nUsage examples:")
    print("1. Train a DQN agent on CartPole:")
    print("   !python main.py train --env_name CartPole-v1 --agent_type dqn --total_episodes 500")
    print("\n2. Compare DQN and Double DQN on CartPole:")
    print("   !python main.py compare --env_name CartPole-v1 --total_episodes 500")
    print("\n3. Generate paper-style figures:")
    print("   !python plot_paper_figure.py --env_name CartPole-v1")
    
    return True

# If this script is run directly
if __name__ == "__main__":
    setup_colab()
