# DQN and Double DQN Implementation and Evaluation

This project implements and evaluates Deep Q-Networks (DQN) and Double DQN algorithms on Atari games, with a focus on analyzing the overestimation bias of DQN and how Double DQN mitigates this issue.

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for zsh shell on macOS
pip install torch torchvision matplotlib numpy opencv-python
pip install "gymnasium[atari,accept-rom-license]"

# If you get errors related to the Atari environment, try installing the classic gym package as a fallback:
pip install gym "gym[atari]"
```

## Usage

### Check CUDA Setup
```bash
python check_cuda.py
```

### Training a Single Agent
 
```bash
# Use CartPole (simpler environment, works out of the box):
python main.py train --env_name CartPole-v1 --agent_type dqn --total_episodes 500

# For Atari environments (if properly installed):
python main.py train --env_name Breakout-v0 --agent_type dqn --total_episodes 2000

# To see prettier progress bars and detailed training information:
python main.py train --env_name CartPole-v1 --agent_type dqn --total_episodes 500 --eval_frequency 50
```

### Comparing DQN and Double DQN

```bash
# Using CartPole (recommended for testing):
python main.py compare --env_name CartPole-v1 --total_episodes 500

# Using Atari (if properly installed):
python main.py compare --env_name Breakout-v0 --total_episodes 2000

# Using multiple seeds for paper-quality plots:
python main.py compare --env_name CartPole-v1 --total_episodes 500 --seeds 5 --plot_paper_style

# Run for CartPole (n_runs = 10)
python main.py compare --env_name CartPole-v1 --total_episodes 2000 --seeds 10 --plot_paper_style --force
```



### Using Google Colab or Kaggle (with GPU acceleration)

For faster training with GPU acceleration, you can use the provided Colab notebook:

1. Open the `colab_demo.ipynb` notebook in Google Colab
2. Make sure to select GPU as the runtime type (Runtime → Change runtime type → GPU)
3. Run the notebook cells to set up the environment and run experiments

Alternatively, you can use the setup script in your own Colab notebook:

```python
# Clone the repository and set up the environment
!git clone https://github.com/jienweng/DQN.git
%cd DQN
!python colab_setup.py

# Train with GPU acceleration
!python main.py compare --env_name CartPole-v1 --total_episodes 500 --seeds 3
```

### Generating Paper-Style Figures

After running multiple experiments with different seeds, you can generate paper-style figures:

```bash
python plot_paper_figure.py --env_name CartPole-v1
```

### Environment Diagnostics

To check which environments are working on your system:

```bash
python diagnose_env.py
```

This will help identify working environments that you can use for training.

## Environment Names

Depending on your version of Gym/Gymnasium, you may need to use different environment IDs:

-   For newer versions of Gymnasium: `ALE/Breakout-v5`, `ALE/Pong-v5`, etc.
-   For older versions of Gym: `Breakout-v0`, `Pong-v0`, etc.
-   For simple environments (always work): `CartPole-v1`, `MountainCar-v0`, etc.

The code handles these different formats automatically.


### Common Arguments

-   `--env_name`: Environment name (example: 'CartPole-v1' or 'ALE/Breakout-v5')
-   `--total_episodes`: Total number of episodes for training (default: 500 for single agent, 500 for comparison)
-   `--max_frames_per_episode`: Maximum number of frames per episode (default: 10000)
-   `--buffer_size`: Replay buffer size (default: 100000)
-   `--batch_size`: Batch size for optimization (default: 32)
-   `--gamma`: Discount factor (default: 0.99)
-   `--eps_start`: Starting epsilon for exploration (default: 1.0)
-   `--eps_end`: Final epsilon for exploration (default: 0.1)
-   `--eps_decay`: Epsilon decay rate in episodes (default: 500)
-   `--target_update`: Target network update frequency in episodes (default: 10)
-   `--lr`: Learning rate (default: 0.00025)
-   `--eval_frequency`: Evaluation frequency in episodes (default: 50)
-   `--eval_episodes`: Number of episodes for evaluation (default: 10)
-   `--render`: Render the environment during final evaluation (flag)
-   `--seed`: Random seed (default: 42)
-   `--force`: Force execution without confirmation prompts
-   `--no_cuda`: Disable CUDA even if available

### Compare-Specific Arguments

-   `--seeds`: Number of seeds to use for each algorithm (default: 1)
-   `--plot_paper_style`: Generate paper-quality plots similar to those in the Double DQN paper (flag)

### Train-Specific Arguments

-   `--agent_type`: Agent type: DQN or Double DQN (default: 'dqn')

## Algorithms Overview

### DQN (Deep Q-Network)

DQN combines Q-learning with deep neural networks to learn optimal policies directly from high-dimensional sensory inputs. Key features include:

-   Experience replay buffer to break correlations in observation sequences
-   Separate target network to reduce overestimation and improve stability
-   CNN architecture to process visual inputs from Atari games

### Double DQN

Double DQN addresses the overestimation bias in standard DQN by decoupling action selection and evaluation:

-   Uses the online network to select actions
-   Uses the target network to evaluate these actions
-   Significantly reduces value overestimation, leading to more reliable Q-value estimates and better policies

## Metrics Tracked

-   Episode rewards
-   Average Q-values
-   Training loss
-   Value overestimation (difference between estimated and actual values)

## Empirical Evaluation of Overoptimism

The project specifically focuses on measuring the overestimation bias that occurs in DQN and how Double DQN addresses this issue. For each evaluation, we compare:

1.  **Estimated Value**: The maximum Q-value at the initial state
2.  **Actual Value**: The true discounted return obtained by following the current policy

The overoptimism is measured as the difference between the estimated and actual values. Results are visualized to demonstrate how Double DQN reduces this overestimation compared to standard DQN.

## References

Based on the following papers:

-   "Playing Atari with Deep Reinforcement Learning" by Mnih et al. (2013)
-   "Deep Reinforcement Learning with Double Q-learning" by van Hasselt et al. (2015)
-   "Human-level control through deep reinforcement learning" by Mnih et al. (2015)