import torch
import math
from pc_network import muPCNetwork, PCResidualBlock

def test_forward_pass():
    net = muPCNetwork(784, 256, 10, 5)
    x = torch.randn(32, 784)
    out = net(x)
    assert out.shape == (32, 10), f'Expected (32,10), got {out.shape}'
    print('PASS: Forward pass')

def test_residual_block():
    block = PCResidualBlock(64, 64, a_ell=0.1)
    x = torch.randn(32, 64)
    out = block(x)
    assert out.shape == x.shape, 'Residual block should preserve shape'
    assert (out - x).abs().mean().item() > 0, 'Output should differ from input'
    print('PASS: Residual block')

def test_activation_variance():
    net = muPCNetwork(784, 256, 10, 20)
    x = torch.randn(128, 784)
    variances = net.measure_activation_variance(x)
    print(f'Variances across {len(variances)} layers:')
    for i, v in enumerate(variances):
        print(f' Layer {i}: {v:.4f}')
    ratio = max(variances) / (min(variances) + 1e-10)
    assert ratio < 100.0, f'Variance ratio too large: {ratio:.2f}'
    print(f' Variance ratio: {ratio:.2f} (stable)')
    print('PASS: Activation variance (muP)')

def test_pc_inference():
    net = muPCNetwork(50, 32, 3, 2)
    x = torch.randn(8, 50)
    y = torch.randn(8, 3)
    net.init_pc_states(x, 8)
    net.states[-1] = y.clone()
    energies = []
    for step in range(30):
        energy = net.pc_inference_step(x, y, lr=0.1)
        energies.append(energy.item())
    print(f' Inference energy: {energies[0]:.4f} -> {energies[-1]:.4f}')
    assert energies[-1] < energies[0], 'Energy should decrease'
    print('PASS: PC inference')

def test_pc_training():
    net = muPCNetwork(100, 64, 5, 3)
    x = torch.randn(16, 100)
    y = torch.randn(16, 5)
    energies = net.train_pc(x, y, inference_steps=20, state_lr=0.1, weight_lr=0.001)
    assert len(energies) == 20
    assert energies[-1] < energies[0], f'Energy: {energies[0]:.4f} -> {energies[-1]:.4f}'
    print(f' Energy: {energies[0]:.4f} -> {energies[-1]:.4f}')
    print('PASS: PC training')

if __name__ == '__main__':
    print('=' * 50)
    print('Phase A2: muPC Network Validation Tests')
    print('=' * 50)
    print()
    print('Test 1: Forward Pass')
    test_forward_pass()
    print()
    print('Test 2: Residual Block')
    test_residual_block()
    print()
    print('Test 3: Activation Variance (muP)')
    test_activation_variance()
    print()
    print('Test 4: PC Inference')
    test_pc_inference()
    print()
    print('Test 5: PC Training')
    test_pc_training()
    print()
    print('=' * 50)
    print('All phase A2 tests passed!')
    print('=' * 50)
