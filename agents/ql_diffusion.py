import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if state.ndim == 3:
            state = state.view(state.shape[0], -1)
        if action.ndim == 3:
            action = action.view(action.shape[0], -1)
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                   state_dim, action_dim, max_action, device,
                   discount=0.98, tau=0.002,max_q_backup=False,
                   warmup_actor_steps=10000, warmup_critic_steps=20000,
                   beta_schedule='linear', n_timesteps=100,
                   ema_decay=0.995, update_ema_every=10,
                   lr=1e-4, lr_decay=False, lr_maxt=1000, grad_norm=1.0,
                   eta=0.1, adaptive_eta=True, eta_min=0.05, eta_max=0.3,
                   cql_alpha=20.0, target_action_noise=0.01):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model,
                                max_action=max_action, beta_schedule=beta_schedule,
                                n_timesteps=n_timesteps).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.device = device
        self.discount = discount
        self.tau = tau
        self.grad_norm = grad_norm

        self.warmup_actor_steps = warmup_actor_steps
        self.warmup_critic_steps = warmup_critic_steps
        self.total_warmup_steps = warmup_actor_steps + warmup_critic_steps

        self.step = 0

        min_eta = 0.0
        max_eta = 0.05
     
        # self.eta = 0.04
        self.eta = 0.05
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.adaptive_eta = adaptive_eta

        self.cql_alpha = cql_alpha
        self.target_action_noise = target_action_noise
        # self.step = 0

        self.q_clip_range = 100.0  # Q value clipping range
        self.step_start_ema = 1000

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

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

    def train(self, batch, iterations, batch_size, log_writer=None, offline=False):
        metric = {'bc_loss': [], 'ql_loss': [], 'critic_loss': [], 'cql_loss': []}

        if offline==False:
            iterations = 1

        for _ in range(iterations):
            if offline==True:
                # Sample replay buffer / batch
                state, action, next_state, reward, not_done = batch.sample(batch_size)

            
            else:
                state = self.to_tensor(batch["obs"], self.device)
                action = self.to_tensor(batch["hgg_acts"], self.device)
                next_state = self.to_tensor(batch["obs_next"], self.device)
                reward = self.to_tensor(batch["rews"], self.device)
                not_done = 1 - self.to_tensor(batch["done"], self.device)

            if state.ndim == 3: state = state.view(state.shape[0], -1)
            if action.ndim == 3: action = action.view(action.shape[0], -1)

            reward = reward.view(-1, 1)
            

            # === Critic Update ===
            
            current_q1, current_q2 = self.critic(state, action)

            next_action = self.ema_model(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm,
                                                             norm_type=2)
            self.critic_optimizer.step()

                

            

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            

            q1_new_action, q2_new_action = self.critic(state, new_action)
            q_loss = - torch.min(q1_new_action, q2_new_action).mean()

            
            actor_loss = bc_loss + self.eta * q_loss
            

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm,
                                                            norm_type=2)
            #nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            self.actor_optimizer.step()

            """ Q-guided Behavior Cloning """

            



            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            

            self.step += 1

            if log_writer is not None:
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', float(critic_loss), self.step)
                

            #metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['critic_loss'].append(float(critic_loss))
            #metric['cql_loss'].append(float(cql_loss))
            #metric['ql_loss'].append(float(q_loss))

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=4, dim=0)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            #q_value = self.critic_target.q_min(state_rpt, action).flatten()
            current_q1, current_q2 = self.critic(state_rpt, action)
            q_value = current_q1
            #idx = torch.multinomial(F.softmax(q_value), 1)
            idx = torch.argmax(q_value)
        return action[idx].cpu().data.numpy().flatten()

    def sample_action1(self, state):
        #if self.step > 26000:
        # if self.step > 36000:
        if self.step > 46000:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
            with torch.no_grad():
                action = self.actor.sample(state_rpt)
                # q_value = self.critic_target.q_min(state_rpt, action).flatten()
                current_q1, current_q2 = self.critic(state_rpt, action)
                q_value = current_q1
                # idx = torch.multinomial(F.softmax(q_value), 1)
                idx = torch.argmax(q_value)
            return action[idx].cpu().data.numpy().flatten()


        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()
