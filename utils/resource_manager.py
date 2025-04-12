import os
import psutil
import torch
import gc
import numpy as np
from typing import Optional

class ResourceManager:
    def __init__(self, logger, memory_fraction: float = 0.8):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_fraction = memory_fraction
        self.process = psutil.Process()
        
        if self.device.type == 'cuda':
            # Set memory fraction for GPU
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        
        self.initial_cpu_memory = self.process.memory_info().rss
        
    def get_memory_usage(self):
        """Get current memory usage for both CPU and GPU"""
        cpu_memory = self.process.memory_info().rss
        gpu_memory = torch.cuda.memory_allocated() if self.device.type == 'cuda' else 0
        
        return {
            'cpu_used_gb': cpu_memory / (1024**3),
            'gpu_used_gb': gpu_memory / (1024**3) if self.device.type == 'cuda' else 0
        }
    
    def log_resource_usage(self):
        """Log current resource usage"""
        memory = self.get_memory_usage()
        cpu_percent = self.process.cpu_percent()
        
        self.logger.logger.info(
            f"Resource Usage - CPU: {cpu_percent:.1f}%, "
            f"Memory: {memory['cpu_used_gb']:.2f}GB"
        )
        
        if self.device.type == 'cuda':
            gpu_percent = torch.cuda.utilization()
            self.logger.logger.info(
                f"GPU Memory: {memory['gpu_used_gb']:.2f}GB, "
                f"Utilization: {gpu_percent}%"
            )
    
    def clear_memory(self, force_cuda_empty: bool = False):
        """Clear unused memory"""
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if force_cuda_empty:
                # Force CUDA memory release (use carefully)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def check_resources(self) -> bool:
        """Check if resources are sufficient to continue"""
        try:
            memory = self.get_memory_usage()
            
            # Check CPU memory (warn if over 90% of system memory)
            system_memory = psutil.virtual_memory()
            if system_memory.percent > 90:
                self.logger.logger.warning("System memory usage is very high!")
                return False
            
            # Check GPU memory if using CUDA
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated()
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory / total_gpu_memory > 0.9:
                    self.logger.logger.warning("GPU memory usage is very high!")
                    return False
            
            return True
        except Exception as e:
            self.logger.logger.error(f"Error checking resources: {e}")
            return False
    
    def optimize_batch_size(self, initial_batch_size: int) -> int:
        """Dynamically adjust batch size based on available memory"""
        if self.device.type != 'cuda':
            return initial_batch_size
            
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_per_sample = 512  # Estimated memory per sample in bytes
            
            # Calculate maximum possible batch size
            max_batch_size = int((free_memory * 0.8) / memory_per_sample)  # Use 80% of free memory
            
            # Keep batch size within reasonable bounds
            optimal_batch_size = min(max(32, initial_batch_size), max_batch_size)
            
            if optimal_batch_size != initial_batch_size:
                self.logger.logger.info(f"Adjusted batch size from {initial_batch_size} to {optimal_batch_size}")
            
            return optimal_batch_size
        except Exception as e:
            self.logger.logger.warning(f"Error optimizing batch size: {e}")
            return initial_batch_size
