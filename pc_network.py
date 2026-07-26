import torch
import torch.nn as nn
import math

class PCResidualBlock(nn.Module):
    def __init__(self, width, fan_in, a_ell, activation='relu'):
        super().__init__()
        self.width = width
        self.a_ell = a_ell
        std = math.sqrt(2.0 / fan_in)
        self.W = nn.Parameter(torch.randn(width, fan_in) * std)
        self.b = nn.Parameter(torch.zeros(width))
        self.activation = activation

    def forward(self, x):
        pre = torch.nn.functional.linear(x, self.W, self.b)
        if self.activation == 'relu':
            act = torch.relu(pre)
        else:
            act = pre
        return x + self.a_ell * act


class muPCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_width, output_dim, num_blocks):
        super().__init__()
        self.input_dim = input_dim
        self.N = hidden_width
        self.L = num_blocks
        self.output_dim = output_dim
        self.a_1 = 1.0 / math.sqrt(input_dim)
        self.a_hidden = 1.0 / math.sqrt(hidden_width * num_blocks)
        self.a_L = 1.0 / hidden_width
        std_in = math.sqrt(2.0 / input_dim)
        self.W_in = nn.Parameter(torch.rand(hidden_width, input_dim) * std_in)
        self.b_in = nn.Parameter(torch.zeros(hidden_width))
        self.blocks = nn.ModuleList([PCResidualBlock(hidden_width, hidden_width, self.a_hidden) for _ in range(num_blocks)])
        std_out = math.sqrt(2.0 / hidden_width)
        self.W_out = nn.Parameter(torch.rand(output_dim, hidden_width) * std_out)
        self.b_out = nn.Parameter(torch.zeros(output_dim))
        self.states = None
        self.errors = None

    def forward(self, x):
        h = self.a_1 * torch.nn.functional.linear(x, self.W_in, self.b_in)
        h = torch.relu(h)
        for block in self.blocks:
            h = block(h)
        out = torch.nn.functional.linear(h, self.W_out, self.b_out)
        return out

    def init_pc_states(self, x, batch_size):
        device = x.device
        self.states = []
        self.errors = []
        self.states.append(x.clone())
        self.errors.append(torch.zeros(batch_size, self.input_dim, device=device))
        h = self.a_1 * torch.nn.functional.linear(x, self.W_in, self.b_in)
        h = torch.relu(h)
        for i, block in enumerate(self.blocks):
            self.states.append(h.clone())
            self.errors.append(torch.zeros(batch_size, self.N, device=device))
            h = block(h)
        self.states.append(h.clone())
        self.errors.append(torch.zeros(batch_size, self.N, device=device))
        out = torch.nn.functional.linear(h, self.W_out, self.b_out)
        self.states.append(out.clone())
        self.errors.append(torch.zeros(batch_size, self.output_dim, device=device))

    def compute_energy(self):
        energy = 0.0
        for eps in self.errors:
            energy = energy + 0.5 * (eps ** 2).sum()
        return energy

    def _recompute_errors(self, y_target):
        pred_1 = self.a_1 * torch.nn.functional.linear(self.states[0], self.W_in, self.b_in)
        pred_1 = torch.relu(pred_1)
        self.errors[1] = self.states[1] - pred_1
        for i, block in enumerate(self.blocks):
            pred = block(self.states[1 + i])
            self.errors[2 + i] = self.states[2 + i] - pred
        pred_out = torch.nn.functional.linear(self.states[-2], self.W_out, self.b_out)
        if y_target is not None:
            self.states[-1] = y_target.clone()
        self.errors[-1] = self.states[-1] - pred_out

    def pc_inference_step(self, x, y_target, lr=0.1):
        pred_1 = self.a_1 * torch.nn.functional.linear(self.states[0], self.W_in, self.b_in)
        pred_1 = torch.relu(pred_1)
        self.errors[1] = self.states[1] - pred_1
        for i, block in enumerate(self.blocks):
            pred = block(self.states[1 + i])
            self.errors[2 + i] = self.states[2 + i] - pred
        pred_out = torch.nn.functional.linear(self.states[-2], self.W_out, self.b_out)
        self.errors[-1] = self.states[-1] - pred_out
        if y_target is not None:
            self.states[-1] = y_target.clone()
            self.errors[-1] = self.states[-1] - pred_out
        with torch.no_grad():
            for i in range(1, len(self.states) - 1):
                grad = self.errors[i].clone()
                if i < len(self.states) - 2:
                    block_idx = i - 1
                    block = self.blocks[block_idx]
                    grad = grad - block.a_ell * torch.nn.functional.linear(self.errors[i + 1], block.W.t())
                else:
                    grad = grad - torch.nn.functional.linear(self.errors[-1], self.W_out.t())
                mask = (self.states[i] > 0).float()
                grad = grad * mask
                self.states[i] = self.states[i] - lr * grad
        self._recompute_errors(y_target)
        return self.compute_energy()

    def pc_weight_update(self, lr=0.001):
        with torch.no_grad():
            batch_size = self.states[0].shape[0]
            dW_in = self.a_1 * torch.mm(self.errors[1].t(), self.states[0]) / batch_size
            self.W_in += lr * dW_in
            self.b_in += lr * self.errors[1].mean(dim=0)
            for i, block in enumerate(self.blocks):
                dW = block.a_ell * torch.mm(self.errors[2 + i].t(), self.states[1 + i]) / batch_size
                block.W += lr * dW
                block.b += lr * self.errors[2 + i].mean(dim=0)
            dW_out = torch.mm(self.errors[-1].t(), self.states[-2]) / batch_size
            self.W_out += lr * dW_out
            self.b_out += lr * self.errors[-1].mean(dim=0)

    def train_pc(self, x, y_target, inference_steps=20, state_lr=0.1, weight_lr=0.001):
        batch_size = x.shape[0]
        self.init_pc_states(x, batch_size)
        self.states[-1] = y_target.clone()
        energies = []
        for step in range(inference_steps):
            energy = self.pc_inference_step(x, y_target, lr=state_lr)
            energies.append(energy.item())
        self.pc_weight_update(lr=weight_lr)
        return energies

    def measure_activation_variance(self, x):
        variances = []
        h = self.a_1 * torch.nn.functional.linear(x, self.W_in, self.b_in)
        h = torch.relu(h)
        variances.append(h.var().item())
        for block in self.blocks:
            h = block(h)
            variances.append(h.var().item())
        return variances
