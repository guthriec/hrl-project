import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_tensor
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        w = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x):
        return x @ self.weight

    def inverse(self, y):
        return y @ self.weight.t()


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dim=256):
        super().__init__()
        self.register_buffer("mask", mask)  # shape [dim], 0/1
        in_dim = int(mask.sum().item())
        out_dim = dim - in_dim
        self.st = MLP(in_dim, 2 * out_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        m = self.mask
        x_id = x[:, m == 1]
        s_t = self.st(x_id)
        s, t = torch.chunk(s_t, 2, dim=-1)
        s = torch.tanh(s)
        x_tr = x[:, m == 0] * torch.exp(s) + t
        out = x.clone()
        out[:, m == 0] = x_tr
        return out

    def inverse(self, y):
        m = self.mask
        y_id = y[:, m == 1]
        s_t = self.st(y_id)
        s, t = torch.chunk(s_t, 2, dim=-1)
        s = torch.tanh(s)
        y_tr = (y[:, m == 0] - t) * torch.exp(-s)
        out = y.clone()
        out[:, m == 0] = y_tr
        return out


class FlowStep(nn.Module):
    def __init__(self, dim, mask, hidden_dim=256):
        super().__init__()
        self.coupling = AffineCoupling(dim, mask, hidden_dim=hidden_dim)
        self.invlin = InvertibleLinear(dim)

    def forward(self, x):
        x = self.coupling(x)
        x = self.invlin(x)
        return x

    def inverse(self, y):
        y = self.invlin.inverse(y)
        y = self.coupling.inverse(y)
        return y


class InvertibleNet(nn.Module):
    def __init__(self, n_layers, state_dim, goal_dim, action_dim, hidden_dim=256, scale=None):
        super().__init__()
        if scale is None:
            scale = torch.ones(action_dim)
        else:
            scale = get_tensor(scale)
        dim = state_dim + goal_dim
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)
        base_mask = torch.zeros(dim)
        base_mask[::2] = 1
        alt_mask = 1 - base_mask
        masks = []
        for i in range(n_layers):
            masks.append(base_mask if i % 2 == 0 else alt_mask)
        self.steps = nn.ModuleList(
            [FlowStep(dim, m, hidden_dim=hidden_dim) for m in masks]
        )
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.output_dummy_dim = self.goal_dim - action_dim

    def forward(self, state, goal):
        """
        Forward pass through invertible network

        Args:
            state: tensor of shape (batch, state_dim)
            goal: tensor of shape (batch, goal_dim) - relative goals (unbounded)

        Returns:
            output: tensor of shape (batch, state_dim + goal_dim)
                   structure: [action (action_dim) | state (state_dim) | latent (goal_dim - action_dim)]
        """
        x = torch.cat([state, goal], dim=1)
        for step in self.steps:
            x = step(x)

        # Only apply tanh and scaling to action dimensions
        output = x.clone()
        output[:, :self.action_dim] = torch.tanh(x[:, :self.action_dim]) * self.scale
        # State and latent dimensions are returned as-is (no tanh)

        return output

    def inverse(self, output_vector):
        """
        Invert the network: given output (action, state, latent), return input (state, goal)

        Args:
            output_vector: tensor of shape (batch, state_dim + goal_dim)
                          structure: [action (action_dim) | state (state_dim) | latent (goal_dim - action_dim)]

        Returns:
            state: tensor of shape (batch, state_dim)
            goal: tensor of shape (batch, goal_dim)
        """
        # Invert tanh and scaling only on action dimensions
        # Process action part separately to avoid in-place operations
        y_action = output_vector[:, :self.action_dim] / self.scale

        # Clamp to prevent arctanh from producing inf/nan (arctanh is undefined at ±1)
        y_action = torch.clamp(y_action, -0.9999, 0.9999)

        # Inverse tanh (arctanh)
        y_action = 0.5 * torch.log((1 + y_action) / (1 - y_action))

        # Reconstruct full tensor (no in-place operation)
        y = torch.cat([y_action, output_vector[:, self.action_dim:]], dim=1)

        # Invert through the flow steps in reverse
        for i, step in enumerate(reversed(self.steps)):
            y = step.inverse(y)

        # Split back into state and goal
        state = y[:, :self.state_dim]
        goal = y[:, self.state_dim:]

        return state, goal