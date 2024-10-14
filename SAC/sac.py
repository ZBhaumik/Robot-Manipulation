import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from networks import Actor, Critic

class SoftActorCritic:
    def __init__(self, state_dim, action_dim):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = Actor(state_dim, action_dim).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.q1 = Critic(state_dim, action_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim).to(self.device)
        self.q1_t = deepcopy(self.q1)
        self.q2_t = deepcopy(self.q2)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)

        self.alpha = 0.2
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True).to(self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.005
    
    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.policy.forward(state)
        return action.detach().cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Updating q-networks
        with torch.no_grad():
            next_action, next_log_prob = self.policy.forward(next_state)
            target_q1 = self.q1_t(next_state, next_action)
            target_q2 = self.q1_t(next_state, next_action)
            target_q_m = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * target_q_m
        
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        critic1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        critic2_loss.backward()
        self.q2_optimizer.step()

        # Updating policy
        action, log_prob = self.policy.forward(state)
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # Update temperature/alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Update parameters of the target q network.
        for target_param, param in zip(self.q1_t.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_t.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)