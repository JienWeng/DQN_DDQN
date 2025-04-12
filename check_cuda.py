import torch

def check_cuda():
    """Check CUDA availability and print device information"""
    print("\nCUDA Environment Check:")
    print("-" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(current_device)
        print(f"\nGPU Device: {device_props.name}")
        print(f"Memory Total: {device_props.total_memory / 1024**2:.0f} MB")
        print(f"CUDA Capability: {device_props.major}.{device_props.minor}")
        print(f"Max threads per block: {device_props.max_threads_per_block}")
        print(f"Max threads per multiprocessor: {device_props.max_threads_per_multi_processor}")
        
        # Test CUDA tensor creation
        try:
            # Try creating and operating on CUDA tensors
            x = torch.rand(1000, 1000).cuda()
            y = x @ x.t()
            print("\nCUDA tensor operations: ✓ Working")
        except Exception as e:
            print(f"\nCUDA tensor operations failed: {e}")
    else:
        print("\nNo CUDA device available. Will use CPU instead.")
        print("To use CUDA, make sure you have:")
        print("1. NVIDIA GPU")
        print("2. CUDA Toolkit installed")
        print("3. PyTorch with CUDA support installed")

if __name__ == "__main__":
    check_cuda()
