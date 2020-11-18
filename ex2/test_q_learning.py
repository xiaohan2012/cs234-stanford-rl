import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from core.q_learning import QN

from configs.q2_linear import config


def grad_norm(model):
    """model: torch.nn.Model"""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def to_tensor(data, dtype):
    return torch.from_numpy(np.asarray(data).astype(dtype))


class Linear(nn.Module):
    def __init__(self, n_hidden, n_out):
        super(Linear, self).__init__()

        self.fc = nn.Linear(n_hidden, n_out)

    def forward(self, state):
        return self.fc(
            torch.flatten(state, start_dim=1)
        )


class LinearQN(QN):
    def get_best_action(self, state):
        state = to_tensor([state], np.float32)
        q_values = self.q_network(state)
        _, idx = torch.max(q_values, dim=1)
        return idx.item(), q_values.detach().numpy()[0]

    def get_q_value(self, state, action):
        q_values = self.q_network(state)

        action_mask = torch.nn.functional.one_hot(
            action, self.env.action_space.n
        ).bool()
        q = torch.masked_select(
            q_values,
            action_mask
        )
        return q
        
    def get_target_q_value(self, state, reward, next_state, done_mask):
        target_q = self.target_q_network(next_state)

        undone_mask = torch.logical_not(done_mask).float()

        max_q, _ = torch.max(target_q, dim=1)
        target_q = reward + undone_mask * self.config.gamma * max_q * self.config.gamma

        return target_q

    def add_optimizer(self):
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.opt_lr,
            betas=[self.config.opt_beta_1, self.config.opt_beta_2],
            eps=self.config.opt_epsilon
        )
        
    def add_models(self):
        n_hidden = np.prod(self.env.observation_space.shape) * self.config.state_history
        n_out = self.env.action_space.n
        self.q_network = Linear(n_hidden, n_out)
        self.target_q_network = Linear(n_hidden, n_out)
        
    def build(self):
        self.add_models()
        self.add_optimizer()

    def update_step(self, t, replay_buffer, lr):
        # zero the parameter gradients
        self.optimizer.zero_grad()
                
        # sample a training batch
        state, action, reward, next_state, done = replay_buffer.sample(self.config.batch_size)

        # conversion to torch.Tensor
        state = to_tensor(state, np.float32)
        action = to_tensor(action, np.int)
        reward = to_tensor(reward, np.float32)
        next_state = to_tensor(next_state, np.float32)
        done = to_tensor(done, np.bool)
        
        # get target q and q values
        target_q_values = self.get_target_q_value(state, reward, next_state, done)
        q_values = self.get_q_value(state, action)

        # get loss
        loss = torch.mean(
            torch.pow(q_values - target_q_values, 2)
        )

        # do gradient update
        loss.backward()

        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_val)

        self.optimizer.step()

        gnorm = grad_norm(self.q_network)

        return loss, gnorm

    def update_target_params(self):
        # self.logger.info('\nupdate target q network params')
        self.target_q_network = deepcopy(self.q_network)
    
if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin,
        config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(
        config.lr_begin, config.lr_end,
        config.lr_nsteps
    )

    # train model
    model = LinearQN(env, config)
    model.run(exp_schedule, lr_schedule)
    
