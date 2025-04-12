import numpy as np
import torch

def verify_model_outputs(model, sample_input, logger):
    """Verify model output shapes and values"""
    try:
        with torch.no_grad():
            output = model(sample_input)
            logger.logger.info(f"Model output shape: {output.shape}")
            logger.logger.info(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            if torch.isnan(output).any():
                logger.logger.error("WARNING: Model output contains NaN values!")
                return False
        return True
    except Exception as e:
        logger.logger.error(f"Error in model verification: {e}")
        return False

def verify_training_progress(metrics, episode, min_reward_threshold, logger):
    """Verify if training is making progress"""
    if len(metrics.episode_rewards) < 100:
        return True
        
    recent_rewards = metrics.episode_rewards[-100:]
    avg_reward = np.mean(recent_rewards)
    reward_std = np.std(recent_rewards)
    
    logger.logger.info(f"Episode {episode} verification:")
    logger.logger.info(f"  Average reward (last 100): {avg_reward:.2f} ± {reward_std:.2f}")
    
    # Check for basic progress
    if avg_reward < min_reward_threshold:
        logger.logger.warning(f"  Low average reward: {avg_reward:.2f} < {min_reward_threshold}")
        return False
        
    # Check for learning stagnation
    if episode > 200:
        old_rewards = metrics.episode_rewards[-200:-100]
        old_avg = np.mean(old_rewards)
        if avg_reward < old_avg:
            logger.logger.warning(f"  Performance degradation detected: {avg_reward:.2f} < {old_avg:.2f}")
            return False
            
    return True
