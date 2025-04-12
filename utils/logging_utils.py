import logging
import os
import json
import numpy as np
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_name, log_dir='logs'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler
        log_file = f"{log_dir}/{experiment_name}_{self.timestamp}.log"
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store metrics
        self.metrics = {}
        
    def log_hyperparameters(self, args):
        """Log hyperparameters to file"""
        self.logger.info("Hyperparameters:")
        params = vars(args)
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save hyperparameters to JSON
        params_file = f"{self.log_dir}/{self.experiment_name}_{self.timestamp}_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
            
    def log_metrics(self, metrics_dict, step):
        """Log metrics for a given step"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            self.logger.info(f"Step {step} - {key}: {value}")
            
    def log_summary(self, metrics_dict):
        """Log summary statistics"""
        self.logger.info("\nExperiment Summary:")
        for key, value in metrics_dict.items():
            if isinstance(value, (list, np.ndarray)):
                mean_val = np.mean(value)
                std_val = np.std(value)
                self.logger.info(f"  {key}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
                
    def save_metrics(self):
        """Save metrics to file with proper JSON serialization"""
        metrics_file = f"{self.log_dir}/{self.experiment_name}_{self.timestamp}_metrics.json"
        
        # Convert numpy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj
        
        serializable_metrics = convert_to_serializable(self.metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
