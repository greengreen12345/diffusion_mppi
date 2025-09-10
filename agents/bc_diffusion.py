# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP


class Diffusion_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=2e-4,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def to_tensor(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).to(device)
            else:
                x_array = np.array(x)
                x_tensor = torch.tensor(x_array, dtype=torch.float32, device=device)
                return x_tensor
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        else:
            raise TypeError(f"[to_tensor] Unsupported input type: {type(x)}")

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            obs, subgoal_target = replay_buffer.sample(batch_size)
            # state = self.to_tensor(batch["obs"], self.device)
            # action = self.to_tensor(batch["hgg_acts"], self.device)
            # next_state = self.to_tensor(batch["obs_next"], self.device)
            # reward = self.to_tensor(batch["rews"], self.device)
            # not_done = 1 - self.to_tensor(batch["done"], self.device)

            state = self.to_tensor(obs, self.device)
            action = self.to_tensor(subgoal_target, self.device)

            # state = obs
            # action = subgoal_target

            if state.dim() == 3: state = state.view(state.shape[0], -1)
            if action.dim() == 3: action = action.view(action.shape[0], -1)

            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric

    def sample_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def sample_action1(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=15, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            # q_value = self.critic_target.q_min(state_rpt, action).flatten()
            # idx = torch.multinomial(F.softmax(q_value), 1)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))

