#!/usr/bin/env python3
"""
Script to analyze and compare results from domain shift experiments.
Run this after completing all four experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import to_rgba

def load_csv_files(directory):
    """Load all CSV files from a directory and return as a list of dataframes."""
    files = glob.glob(os.path.join(directory, '*.csv'))
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Add filename as a column
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return dataframes

def aggregate_rewards(dataframes):
    """Aggregate episode rewards across all trials."""
    all_rewards = {}
    for df in dataframes:
        # Group by episode and get mean reward
        episode_rewards = df.groupby('episode')['cumulative_reward'].last().reset_index()
        
        # Extract trial number from filename
        filename = df['source_file'].iloc[0]
        try:
            trial_num = int(''.join(filter(str.isdigit, filename.split('_trial_')[1].split('.')[0])))
            all_rewards[trial_num] = episode_rewards['cumulative_reward'].values
        except:
            print(f"Could not extract trial number from {filename}")
            
    return all_rewards

def get_learning_curve(all_rewards, window=100):
    """Calculate the learning curve with moving average."""
    # Find the minimum length to align all trials
    min_length = min(len(rewards) for rewards in all_rewards.values())
    
    # Create an array of all rewards, truncated to the minimum length
    reward_array = np.array([rewards[:min_length] for rewards in all_rewards.values()])
    
    # Calculate mean and std across trials
    mean_rewards = np.mean(reward_array, axis=0)
    std_rewards = np.std(reward_array, axis=0)
    
    # Apply smoothing with moving average
    smooth_mean = np.zeros_like(mean_rewards)
    for i in range(len(mean_rewards)):
        smooth_mean[i] = np.mean(mean_rewards[max(0, i-window+1):i+1])
    
    return {
        'episodes': np.arange(min_length),
        'mean': mean_rewards,
        'std': std_rewards,
        'smooth_mean': smooth_mean
    }

def plot_comparison(bipedal_dsp, bipedal_control, cartpole_dsp, cartpole_control):
    """Create comparison plots for all four experiment types."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # Colors with transparency for uncertainty bands
    colors = {
        'dsp': '#1f77b4',  # blue
        'control': '#ff7f0e',  # orange
    }
    
    # BipedalWalker plot
    ax = axes[0]
    x_bipedal = bipedal_dsp['episodes']
    
    # Plot DSP
    ax.plot(x_bipedal, bipedal_dsp['smooth_mean'], color=colors['dsp'], linewidth=2, label='With Domain Shift Predictor')
    ax.fill_between(
        x_bipedal, 
        bipedal_dsp['smooth_mean'] - bipedal_dsp['std'], 
        bipedal_dsp['smooth_mean'] + bipedal_dsp['std'],
        color=to_rgba(colors['dsp'], 0.3)
    )
    
    # Plot Control
    ax.plot(x_bipedal, bipedal_control['smooth_mean'], color=colors['control'], linewidth=2, label='Without Domain Shift Predictor')
    ax.fill_between(
        x_bipedal, 
        bipedal_control['smooth_mean'] - bipedal_control['std'], 
        bipedal_control['smooth_mean'] + bipedal_control['std'],
        color=to_rgba(colors['control'], 0.3)
    )
    
    ax.set_title('BipedalWalker with Domain Shift', fontsize=16)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Cumulative Reward', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # CartPole plot
    ax = axes[1]
    x_cartpole = cartpole_dsp['episodes']
    
    # Plot DSP
    ax.plot(x_cartpole, cartpole_dsp['smooth_mean'], color=colors['dsp'], linewidth=2, label='With Domain Shift Predictor')
    ax.fill_between(
        x_cartpole, 
        cartpole_dsp['smooth_mean'] - cartpole_dsp['std'], 
        cartpole_dsp['smooth_mean'] + cartpole_dsp['std'],
        color=to_rgba(colors['dsp'], 0.3)
    )
    
    # Plot Control
    ax.plot(x_cartpole, cartpole_control['smooth_mean'], color=colors['control'], linewidth=2, label='Without Domain Shift Predictor')
    ax.fill_between(
        x_cartpole, 
        cartpole_control['smooth_mean'] - cartpole_control['std'], 
        cartpole_control['smooth_mean'] + cartpole_control['std'],
        color=to_rgba(colors['control'], 0.3)
    )
    
    ax.set_title('CartPole with Domain Shift', fontsize=16)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Cumulative Reward', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('domain_shift_comparison.png', dpi=300)
    plt.show()

def calculate_statistics(all_curves):
    """Calculate convergence statistics."""
    stats = {}
    
    for name, curve in all_curves.items():
        # Calculate convergence speed (episode where mean reward first reaches 90% of max)
        max_reward = np.max(curve['smooth_mean'])
        convergence_threshold = 0.9 * max_reward
        convergence_point = np.argmax(curve['smooth_mean'] >= convergence_threshold)
        
        # Calculate stability (inverse of std in last 100 episodes)
        stability = 1.0 / np.mean(curve['std'][-100:]) if len(curve['std']) >= 100 else 0
        
        # Calculate final performance (mean of last 100 episodes)
        final_performance = np.mean(curve['smooth_mean'][-100:]) if len(curve['smooth_mean']) >= 100 else 0
        
        stats[name] = {
            'convergence_episode': convergence_point,
            'stability': stability,
            'final_performance': final_performance,
            'max_reward': max_reward
        }
    
    return stats

def print_statistics(stats):
    """Print the statistics in a formatted table."""
    print("\n=== PERFORMANCE STATISTICS ===\n")
    print(f"{'Metric':<25} {'BipedalWalker DSP':<20} {'BipedalWalker Control':<20} {'CartPole DSP':<20} {'CartPole Control':<20}")
    print("-" * 105)
    
    metrics = ['convergence_episode', 'stability', 'final_performance', 'max_reward']
    
    for metric in metrics:
        row = f"{metric.replace('_', ' ').title():<25}"
        for env in ['bipedal_dsp', 'bipedal_control', 'cartpole_dsp', 'cartpole_control']:
            value = stats[env][metric]
            if metric == 'convergence_episode':
                row += f"{int(value):<20}"
            else:
                row += f"{value:.4f}:<20}"
        print(row)
    
    print("\nInterpretation:")
    print("- Lower convergence episode is better (faster learning)")
    print("- Higher stability is better (less variance)")
    print("- Higher final performance is better (better policy)")
    print("- Higher max reward is better (peak performance)")

def main():
    # Output directories (adjust paths as needed)
    bipedal_dsp_dir = 'results'
    bipedal_control_dir = 'results'
    cartpole_dsp_dir = 'results_cartpole'
    cartpole_control_dir = 'results_cartpole'
    
    print("Loading data...", flush=True)
    
    # Load all CSV files
    bipedal_dsp_dfs = load_csv_files(bipedal_dsp_dir)
    bipedal_control_dfs = load_csv_files(bipedal_control_dir)
    cartpole_dsp_dfs = load_csv_files(cartpole_dsp_dir)
    cartpole_control_dfs = load_csv_files(cartpole_control_dir)
    
    print("Processing rewards...", flush=True)
    
    # Aggregate rewards by trial
    bipedal_dsp_rewards = aggregate_rewards(bipedal_dsp_dfs)
    bipedal_control_rewards = aggregate_rewards(bipedal_control_dfs)
    cartpole_dsp_rewards = aggregate_rewards(cartpole_dsp_dfs)
    cartpole_control_rewards = aggregate_rewards(cartpole_control_dfs)
    
    # Calculate learning curves
    bipedal_dsp_curve = get_learning_curve(bipedal_dsp_rewards)
    bipedal_control_curve = get_learning_curve(bipedal_control_rewards)
    cartpole_dsp_curve = get_learning_curve(cartpole_dsp_rewards)
    cartpole_control_curve = get_learning_curve(cartpole_control_rewards)
    
    print("Plotting comparison...", flush=True)
    
    # Plot comparison
    plot_comparison(bipedal_dsp_curve, bipedal_control_curve, cartpole_dsp_curve, cartpole_control_curve)
    
    # Calculate and print statistics
    all_curves = {
        'bipedal_dsp': bipedal_dsp_curve,
        'bipedal_control': bipedal_control_curve,
        'cartpole_dsp': cartpole_dsp_curve,
        'cartpole_control': cartpole_control_curve
    }
    
    stats = calculate_statistics(all_curves)
    print_statistics(stats)
    
    print("\nAnalysis complete! Results have been saved as domain_shift_comparison.png")

if __name__ == "__main__":
    main()