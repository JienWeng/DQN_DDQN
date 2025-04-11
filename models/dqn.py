import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()
        
        # Determine if input is image-based (Atari) or vector-based (CartPole)
        self.is_image_input = len(input_shape) > 1
        
        if self.is_image_input:
            # CNN architecture for image inputs (like Atari)
            self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            
            # Calculate output size of convolution layers
            conv_output_size = self._get_conv_output(input_shape)
            
            self.fc1 = nn.Linear(conv_output_size, 512)
            self.fc2 = nn.Linear(512, n_actions)
        else:
            # MLP architecture for vector inputs (like CartPole)
            self.fc1 = nn.Linear(input_shape[0], 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, n_actions)
        
    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(F.relu(o))
        o = self.conv3(F.relu(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if self.is_image_input:
            # Process image input through CNN
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        else:
            # Process vector input through MLP
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

class DQNAgent:
    def __init__(self, 
                input_shape,
                n_actions,
                buffer_size=100000, 
                batch_size=32,
                gamma=0.99,
                eps_start=1.0,
                eps_end=0.1,
                eps_decay=1000000,
                target_update=10000,
                learning_rate=0.00025,
                verbose=True):
        
        # Check if CUDA is available but only print if verbose is enabled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        if self.verbose:
            print(f"Using device: {self.device}")
            # Only print detailed GPU info once during initialization, not every call
            if self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.policy_net = DQNNetwork(input_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.n_actions = n_actions
        
        self.steps_done = 0
        self.is_image_input = len(input_shape) > 1
        
    def select_action(self, state, evaluate=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
            
        if evaluate:
            eps_threshold = 0.05  # Small epsilon for evaluation
            
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Handle state preprocessing based on input type
                if self.is_image_input:
                    # Image input (4D tensor: batch_size x channels x height x width)
                    state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                else:
                    # Vector input from environments like CartPole
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    dtype=torch.bool, device=self.device)
        
        # Process states based on input type
        if self.is_image_input:
            # Image states
            non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) 
                                            for s in batch.next_state if s is not None]).to(self.device)
            state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        else:
            # Vector states
            non_final_next_states = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(self.device)
            state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states (DQN)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.sum() > 0:  # Check if there are any non-final next states
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_q_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_q_values(self, state):
        """
        Get Q-values for a given state
        Handles different state formats (numpy arrays, lists, tensors)
        """
        with torch.no_grad():
            try:
                if self.is_image_input:
                    # Handle image input (4D tensor)
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    elif isinstance(state, list):
                        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                    else:
                        # Assume it's already a tensor
                        state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 3 else state.to(self.device)
                else:
                    # Handle vector input
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.FloatTensor(state).to(self.device)
                    elif isinstance(state, list):
                        state_tensor = torch.FloatTensor(state).to(self.device)
                    else:
                        # Assume it's already a tensor
                        state_tensor = state.to(self.device)
                    
                    # Ensure correct shape
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                
                return self.policy_net(state_tensor).cpu().numpy()
            except Exception as e:
                print(f"Error processing state in get_q_values: {e}")
                print(f"State type: {type(state)}")
                print(f"State shape or length: {state.shape if hasattr(state, 'shape') else len(state)}")
                # Return zeros as fallback
                return np.zeros((1, self.n_actions))
