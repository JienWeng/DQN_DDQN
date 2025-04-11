"""
Diagnostic tool for Atari environments
"""
import sys
import os
import importlib.util

def check_module_installed(module_name):
    """Check if a Python module is installed"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def diagnose_environment():
    """Run diagnostics on the environment setup"""
    print_section("SYSTEM INFORMATION")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    print_section("PACKAGE CHECK")
    packages = [
        "gymnasium", "gym", "ale_py", "numpy", "torch", "cv2"
    ]
    
    for package in packages:
        installed = check_module_installed(package)
        status = "✓ INSTALLED" if installed else "✗ NOT FOUND"
        print(f"{package}: {status}")
        
        if installed:
            try:
                module = importlib.import_module(package)
                if hasattr(module, "__version__"):
                    print(f"    Version: {module.__version__}")
                elif hasattr(module, "version"):
                    print(f"    Version: {module.version}")
            except Exception as e:
                print(f"    Error importing: {e}")
    
    print_section("GYMNASIUM DETAILS")
    if check_module_installed("gymnasium"):
        try:
            import gymnasium as gym
            print(f"Gymnasium version: {gym.__version__}")
            
            # Check for atari
            try:
                from gymnasium.envs.atari import AtariEnv
                print("✓ Gymnasium Atari environments found")
            except ImportError:
                print("✗ Gymnasium Atari environments not found")
            
            # Check environment registry
            print("\nRegistered Atari environments:")
            env_specs = gym.registry.values()
            atari_envs = [spec for spec in env_specs if "ALE" in spec.id or "atari" in spec.id.lower()]
            
            if atari_envs:
                for idx, spec in enumerate(atari_envs[:10]):  # Show first 10
                    print(f"  {spec.id}")
                if len(atari_envs) > 10:
                    print(f"  ... and {len(atari_envs) - 10} more")
            else:
                print("  No Atari environments found in registry")
        except Exception as e:
            print(f"Error inspecting gymnasium: {e}")
    
    print_section("ALE DETAILS")
    if check_module_installed("ale_py"):
        try:
            import ale_py
            print(f"ALE-Py version: {ale_py.__version__}")
            
            # Check for available ROMs
            try:
                import ale_py.roms as roms
                print("\nAvailable ROMs:")
                rom_modules = [name for name in dir(roms) if not name.startswith("_")]
                if rom_modules:
                    for rom in rom_modules[:10]:  # Show first 10
                        print(f"  {rom}")
                    if len(rom_modules) > 10:
                        print(f"  ... and {len(rom_modules) - 10} more")
                else:
                    print("  No ROMs found")
            except Exception as e:
                print(f"Error inspecting ROMs: {e}")
        except Exception as e:
            print(f"Error inspecting ale_py: {e}")
    
    print_section("ENVIRONMENT CREATION TEST")
    try:
        from utils.atari_wrappers import make_atari_env, USING_GYMNASIUM
        print("Attempting to create a simple Atari environment...")
        
        # Try simple classic control environments first
        print("\nTrying classic control environments:")
        classic_envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
        classic_success = False
        
        for env_name in classic_envs:
            try:
                if USING_GYMNASIUM:
                    import gymnasium as gym
                    env = gym.make(env_name)
                    print(f"✓ Successfully created classic environment: {env_name}")
                    # Test basic functionality
                    obs, info = env.reset()
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.close()
                    classic_success = True
                    break
                else:
                    import gym
                    env = gym.make(env_name)
                    print(f"✓ Successfully created classic environment: {env_name}")
                    # Test basic functionality
                    obs = env.reset()
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    env.close()
                    classic_success = True
                    break
            except Exception as e:
                print(f"✗ Failed to create classic environment {env_name}: {e}")
        
        if not classic_success:
            print("Warning: Could not create even simple environments. Check your installation.")
        
        # Now try Atari environments
        atari_envs = ["Pong-v0", "ALE/Pong-v5", "PongNoFrameskip-v4", 
                     "Breakout-v0", "ALE/Breakout-v5", "BreakoutNoFrameskip-v4"]
        
        for env_name in atari_envs:
            try:
                print(f"\nTrying Atari environment: {env_name}")
                env = make_atari_env(env_name)
                print(f"✓ Successfully created environment: {env_name}")
                print(f"  Observation space: {env.observation_space}")
                print(f"  Action space: {env.action_space}")
                
                # Test basic functionality with proper error handling
                try:
                    if USING_GYMNASIUM:
                        obs, info = env.reset()
                        print("  Reset successful")
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        print("  Step successful")
                    else:
                        obs = env.reset()
                        print("  Reset successful")
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        print("  Step successful")
                except Exception as e:
                    print(f"  Warning: Basic functionality test failed: {e}")
                
                env.close()
                print("  Environment closed successfully")
                print("\n✅ Found a working environment! Use this in your training code:")
                print(f"python main.py train --env_name {env_name} --agent_type dqn --total_frames 1000000")
                break
            except Exception as e:
                print(f"✗ Failed to create environment {env_name}: {e}")
        else:
            print("\n❌ Could not create any Atari test environments.")
            if classic_success:
                print("Consider using a classic control environment instead:")
                print(f"python main.py train --env_name {classic_envs[0]} --agent_type dqn --total_frames 100000")
    except Exception as e:
        print(f"Error running environment test: {e}")
    
    print_section("NEXT STEPS")
    print("If no environments could be created, here are some options:")
    print("1. Install ROMs manually with:")
    print("   python -m ale_py.roms --install-dir PATH_TO_ROMS")
    print("2. Try using a simpler environment like CartPole-v1:")
    print("   python main.py train --env_name CartPole-v1 --agent_type dqn --total_frames 100000")
    print("3. Make sure ale-py is properly installed:")
    print("   pip install ale-py gymnasium[accept-rom-license,atari]")

if __name__ == "__main__":
    diagnose_environment()
