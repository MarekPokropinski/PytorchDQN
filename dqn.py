import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


class DQN:
    def __init__(self, state_size, num_actions, device='cpu') -> None:
        self.device = torch.device(device)
        self.model = MLP(input_size=state_size, output_size=num_actions).to(self.device)
        self.target_model = MLP(input_size=state_size, output_size=num_actions).to(self.device)
        self.update_target()
        self.target_model.eval()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        # self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=100000)

        self.gamma = torch.tensor(0.95).to(self.device)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((
            state, action, reward, next_state, done
        ))

    def train_step(self, mb_size=64):        
        mb = random.sample(self.memory, k=mb_size)

        states = torch.tensor([x[0] for x in mb], dtype=torch.float, device=self.device)
        actions = torch.tensor([x[1] for x in mb], dtype=torch.long, device=self.device)
        rewards = torch.tensor([x[2] for x in mb], dtype=torch.float, device=self.device)
        next_states = torch.tensor([x[3] for x in mb], dtype=torch.float, device=self.device)
        dones = torch.tensor([x[4] for x in mb], dtype=torch.float, device=self.device)        

        with torch.no_grad():
            next_values = self.target_model(next_states)
            # next_values = self.model(next_states)
            target_values = rewards + (1.0-dones)*self.gamma*next_values.amax(dim=-1)


        values = self.model(states)[torch.arange(0, mb_size), actions]        
        loss = self.loss_fn(values, target_values)
        self.optim.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optim.step()
        return loss.item()

    def act(self, state):
        with torch.no_grad():
            state = state[np.newaxis, ...]
            state = torch.tensor(state, dtype=torch.float)
            
            state = state.to(self.device)
            
            pred = self.model(state)
            return pred.argmax(dim=-1).cpu().item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path, eval=True):
        params = torch.load(path)
        self.model.load_state_dict(params)
        self.target_model.load_state_dict(params)

        if eval:
            self.model.eval()

        self.target_model.eval()
