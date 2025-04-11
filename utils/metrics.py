import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

class MetricTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_q_values = []
        self.episode_losses = []
        self.estimated_values = []
        self.actual_values = []
        self.overoptimism_values = []
        
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)
        
    def add_episode_length(self, length):
        self.episode_lengths.append(length)
        
    def add_q_value(self, q_value):
        self.episode_q_values.append(q_value)
        
    def add_loss(self, loss):
        self.episode_losses.append(loss)
        
    def add_value_comparison(self, estimated_value, actual_value):
        """
        Add estimated and actual value comparison for measuring overoptimism
        """
        try:
            if not np.isnan(estimated_value) and not np.isnan(actual_value):
                self.estimated_values.append(float(estimated_value))
                self.actual_values.append(float(actual_value))
                self.overoptimism_values.append(float(estimated_value - actual_value))
        except Exception as e:
            print(f"Error adding value comparison: {e}")
        
    def get_average_reward(self, window=100):
        if len(self.episode_rewards) == 0:
            return 0
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards)
        return np.mean(self.episode_rewards[-window:])
    
    def get_average_q_value(self, window=100):
        if len(self.episode_q_values) == 0:
            return 0
        if len(self.episode_q_values) < window:
            return np.mean(self.episode_q_values)
        return np.mean(self.episode_q_values[-window:])
    
    def get_average_loss(self, window=100):
        if len(self.episode_losses) == 0:
            return 0
        if len(self.episode_losses) < window:
            return np.mean(self.episode_losses)
        return np.mean(self.episode_losses[-window:])
    
    def get_average_overoptimism(self, window=100):
        if len(self.overoptimism_values) == 0:
            return 0
        if len(self.overoptimism_values) < window:
            return np.mean(self.overoptimism_values)
        return np.mean(self.overoptimism_values[-window:])
    
    def plot_rewards(self, window=100, title="Episode Rewards", save_path=None):
        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_rewards)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot moving average
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(self.episode_rewards)), moving_avg, 'r-')
        
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_q_values(self, title="Average Q-Values", save_path=None):
        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_q_values)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_losses(self, title="Training Loss", save_path=None):
        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_losses)
        plt.title(title)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_overoptimism(self, title="Value Overestimation", save_path=None):
        """
        Plot value overestimation in an academic paper style
        """
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 11,
            'figure.figsize': (8, 6),
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.framealpha': 0.7
        })
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Plot estimated values
        episodes = range(len(self.estimated_values))
        ax.plot(episodes, self.estimated_values, 'C0', linewidth=2, label='Estimated Values')
        
        # Plot actual values as a horizontal line
        if self.actual_values:
            actual_mean = np.mean(self.actual_values)
            ax.axhline(y=actual_mean, color='C0', linestyle='--', linewidth=1.5, 
                      label='Actual Discounted Return')
        
        # Formatting
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Training Episode', fontweight='bold')
        ax.set_ylabel('Value Estimate', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', frameon=True)
        
        # Add some padding to the y-axis
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot overoptimism value (estimated - actual)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(episodes, self.overoptimism_values, 'C1', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        ax.set_title(f"Overoptimism (Estimated - Actual)", fontweight='bold')
        ax.set_xlabel('Training Episode', fontweight='bold')
        ax.set_ylabel('Value Difference', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_diff{ext}", dpi=300, bbox_inches='tight')
        
        plt.show()
