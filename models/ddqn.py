import torch
import torch.nn.functional as F
import numpy as np
from models.dqn import DQNAgent, Transition  # Import Transition from dqn.py

class DoubleDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))  # Use the imported Transition instead of self.Transition
        
        # Process states based on input type
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    dtype=torch.bool, device=self.device)
        
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
        
        # Double DQN: Use policy net to select actions and target net to evaluate them
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        if non_final_mask.sum() > 0:  # Only process if there are non-final states
            # Get actions from policy network
            with torch.no_grad():
                next_action_values = self.policy_net(non_final_next_states)
                next_actions = next_action_values.max(1)[1].unsqueeze(1)
                
                # Evaluate actions using target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        
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
