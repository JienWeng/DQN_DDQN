import numpy as np
from collections import deque
import cv2

# Attempt to import gymnasium first, then fall back to gym
try:
    import gymnasium as gym
    USING_GYMNASIUM = True
except ImportError:
    import gym
    USING_GYMNASIUM = False

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset."""
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        if USING_GYMNASIUM:
            # Gymnasium version returns (obs, info)
            obs, info = self.env.reset(**kwargs)
        else:
            # Classic gym version returns just obs
            obs = self.env.reset(**kwargs)
            info = {}
            
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        
        for _ in range(noops):
            if USING_GYMNASIUM:
                obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
                done = terminated or truncated
                if done:
                    obs, info = self.env.reset(**kwargs)
            else:
                obs, _, done, info = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset(**kwargs)
        
        if USING_GYMNASIUM:
            return obs, info
        else:
            return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = truncated = done = False
        info = {}
        
        if USING_GYMNASIUM:
            # Gymnasium version
            for i in range(self._skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if i == self._skip - 2: self._obs_buffer[0] = obs
                if i == self._skip - 1: self._obs_buffer[1] = obs
                total_reward += reward
                if done:
                    break
                    
            max_frame = self._obs_buffer.max(axis=0)
            return max_frame, total_reward, terminated, truncated, info
        else:
            # Classic gym version
            for i in range(self._skip):
                obs, reward, done, info = self.env.step(action)
                if i == self._skip - 2: self._obs_buffer[0] = obs
                if i == self._skip - 1: self._obs_buffer[1] = obs
                total_reward += reward
                if done:
                    break
                    
            max_frame = self._obs_buffer.max(axis=0)
            return max_frame, total_reward, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """Warp frames to 84x84 as done in the Nature paper"""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        if USING_GYMNASIUM:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = {}
            
        for _ in range(self.k):
            self.frames.append(obs)
            
        if USING_GYMNASIUM:
            return self._get_ob(), info
        else:
            return self._get_ob()

    def step(self, action):
        if USING_GYMNASIUM:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_ob(), reward, terminated, truncated, info
        else:
            obs, reward, done, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # Careful! This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class TransposeFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Transpose observation space (required for CNNs in PyTorch)"""
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def make_atari(env_id):
    """
    Create an Atari environment handling both gym and gymnasium formats
    """
    env = None
    
    # Extract base environment name without version
    if '/' in env_id:
        # Handle ALE/Breakout-v5 format
        base_env_name = env_id.split('/')[1].split('-')[0]
    else:
        # Handle Breakout-v0 or BreakoutNoFrameskip-v4 format
        base_env_name = env_id.split('-')[0].replace('NoFrameskip', '')
    
    # For Gymnasium with ale-py already installed
    try:
        import ale_py
        print(f"Found ale_py version {ale_py.__version__}, attempting to create ALE environment")
        
        # Try to create the environment directly with Gymnasium
        try:
            import gymnasium as gym
            # Register ALE environments if needed
            try:
                from gymnasium.envs.registration import register
                register(
                    id=f"ALE/{base_env_name}-v5",
                    entry_point="gymnasium.envs.atari:AtariEnv",
                    kwargs={"game": base_env_name, "obs_type": "rgb", "frameskip": 1},
                    max_episode_steps=10000,
                )
            except Exception as e:
                print(f"Warning: couldn't register environment: {e}")
                
            # Try to create the environment
            env = gym.make(f"ALE/{base_env_name}-v5", render_mode=None)
            print(f"Successfully created Gymnasium ALE environment for {base_env_name}")
        except Exception as e:
            print(f"Gymnasium ALE creation failed: {e}")
            
            # Fallback: try direct ale-py
            try:
                from ale_py import ALEInterface
                from ale_py.roms import Breakout
                
                ale = ALEInterface()
                ale.loadROM(getattr(ale_py.roms, base_env_name))
                print(f"Successfully loaded ROM via ale-py directly")
                
                # Now try to wrap this with Gymnasium
                try:
                    from gymnasium.envs.atari import AtariEnv
                    env = AtariEnv(game=base_env_name, obs_type="rgb")
                    print(f"Successfully created environment using direct ale-py approach")
                except Exception as e:
                    print(f"Direct AtariEnv creation failed: {e}")
            except Exception as e:
                print(f"Direct ale-py ROM loading failed: {e}")
    except ImportError:
        print("ale-py not found, trying alternative approaches")
    
    # If the above didn't work, try other formats
    if env is None:
        # List of environment formats to try
        env_formats = []
        
        if USING_GYMNASIUM:
            env_formats.extend([
                f"ALE/{base_env_name}-v5",
                base_env_name,
                f"{base_env_name}-v5",
                f"{base_env_name}-v4",
                f"{base_env_name}-v0",
                env_id
            ])
        else:
            env_formats.extend([
                f"{base_env_name}NoFrameskip-v4",
                f"{base_env_name}-v0",
                f"{base_env_name}-v4",
                env_id
            ])
        
        # Try creating environment with each format
        for env_format in env_formats:
            try:
                if USING_GYMNASIUM:
                    env = gym.make(env_format, render_mode=None)
                else:
                    env = gym.make(env_format)
                print(f"Successfully created environment: {env_format}")
                break
            except Exception as e:
                print(f"Failed to create environment {env_format}: {e}")
    
    # If still no environment, suggest manual ROM installation
    if env is None:
        error_msg = (
            f"Failed to create any Atari environment for {env_id}.\n"
            f"You may need to manually install and accept the ROM license:\n"
            f"1. Run: python -m ale_py.roms --install-dir PATH_TO_ROMS\n"
            f"2. Download Atari ROMs from an appropriate source\n"
            f"3. Move the ROM files to the installation directory\n"
            f"Or try a different environment name."
        )
        print(error_msg)
        raise ValueError(error_msg)
    
    # Apply wrappers
    try:
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
    except Exception as e:
        print(f"Error applying wrappers: {e}")
        print("Returning unwrapped environment")
    
    return env

def wrap_deepmind(env, frame_stack=4, scale=False):
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    env = FrameStack(env, frame_stack)
    env = TransposeFrame(env)
    return env

def make_atari_env(env_id, frame_stack=4, scale=False):
    """
    Create an environment with appropriate wrappers
    """
    # Check if the environment is a classic control environment
    classic_envs = ["CartPole", "MountainCar", "Acrobot", "Pendulum"]
    is_classic = any(env_name in env_id for env_name in classic_envs)
    
    if is_classic:
        # For classic control environments, just return the basic environment
        print(f"Creating classic control environment: {env_id}")
        
        if USING_GYMNASIUM:
            import gymnasium as gym
            env = gym.make(env_id)
        else:
            import gym
            env = gym.make(env_id)
        
        return env
    else:
        # For Atari environments, apply the standard wrappers
        env = make_atari(env_id)
        return wrap_deepmind(env, frame_stack, scale)
