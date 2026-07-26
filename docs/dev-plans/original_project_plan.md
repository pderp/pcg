# Predictive Coding Test Plan: Frozen Supercompilation + FabricPC

## Overview
Phased test plan for implementing and validating ideas from:
1. **pc_frozen_supercompilation.pdf** - Freezing/supercompiling PC inference dynamics
2. **pc_gpu_fabricpc.pdf** - FabricPC: muPC scaling generalized to arbitrary DAGs

## Part A: FabricPC - Scaling PC to Arbitrary DAGs

### Background
FabricPC extends muPC from sequential chains to arbitrary DAGs. Core formula:
a = gain / sqrt(fan_in * K_slot * L)
- fan_in = weight-matrix fan-in, K_slot = in-degree of target slot, L = residual depth
- Skip connections pass through at unit gain. Output nodes excluded from muPC scaling.

### Phase A1: Baseline PC Chain (Validation)
Goal: Reproduce muPC chain scaling as baseline.
Steps: Sequential PC residual blocks, muPC scaling, train on MNIST/CIFAR-10 at depths 10/50/100/128, measure activation variance and stability.
Gate: 100+ layer network trains stably with O(1) activation variance.

### Phase A2: Simple DAG with Merge Nodes
Goal: Test K_slot correction on non-trivial DAG.
Steps: Diamond/merge topology with K_slot=2 at merge, compare with/without correction.
Gate: Merge nodes maintain O(1) variance with K_slot correction.

### Phase A3: Arbitrary DAG with Skip Connections
Goal: Test FabricPC on complex graph with varying in-degrees.
Steps: Multi-skip DAG with varying K_slot, verify scalable vs skip edge flags, train and measure.
Gate: Stable training on 100+ layer equivalent DAG.

### Phase A4: Transformer Nodes
Goal: FabricPC with attention-based nodes.
Steps: PC energy-based transformer block, apply FabricPC scaling, train on sequence task.
Gate: PC transformer trains stably at depth.

### Phase A5: Hyperparameter Transfer
Goal: Verify zero-shot transfer of learning rates across width and depth.
Steps: Tune on small proxy, transfer to large model without re-tuning.
Gate: Optimal LR transfers with <10% accuracy gap.

## Part B: Frozen Supercompilation - Compiling PC Inference

### Background
Supercompilation specializes programs w.r.t. input constraints, unrolling loops. Applied to PC: freeze converged inference dynamics into single forward pass, addressing inference inner-loop cost.

### Phase B1: Baseline Inference Profiling
Goal: Characterize inference cost.
Steps: Train small PC net, log inference steps to convergence, per-step cost, energy trajectory.
Gate: Clear profile of inference cost and convergence dynamics.

### Phase B2: Linearized Freeze
Goal: Simplest supercompilation - linearize around fixed point.
Steps: Compute Jacobian at converged state, approximate as linear solve, compare full vs single-step vs K-step.
Gate: >90% accuracy retention with >10x speedup.

### Phase B3: Unrolled Supercompilation
Goal: Unroll full inference loop and optimize computation graph.
Steps: Unroll T steps, treat as feedforward, fine-tune with frozen weights + shortcut params, sweep T.
Gate: T=5-10 matches full inference at 5-10x speedup.

### Phase B4: Progressive Freezing
Goal: Test progressive layer freezing.
Steps: Freeze layers 1..K, keep K+1..L iterative, sweep K.
Gate: 50%+ of layers freezable with <1% accuracy loss.

### Phase B5: Combined Freeze + FabricPC
Goal: Test composition of freezing with FabricPC scaling.
Steps: Train deep FabricPC net, apply best freezing method, measure end-to-end speedup.
Gate: Combined approach achieves depth scalability + inference speedup.

## Part C: Integration Tests

### Phase C1: Speedup on Real Workload
Steps: CIFAR-10 with 50-100 layer FabricPC net, freeze, benchmark vs raw PC and backprop.

### Phase C2: Ablation Study
Conditions: Raw PC, muPC only, FabricPC only, freeze only, muPC+freeze, FabricPC+freeze.
Measure: max trainable depth, inference cost, accuracy, stability.

### Phase C3: Edge Case Testing
1. Correlated merges - verify FabricPC sqrt(K) under-correction
2. Ill-conditioned inference - measure Hessian condition number at depth
3. Attention at depth - test beyond 20 layers
4. Freeze quality at depth - test degradation for deeper networks

## Implementation Notes
Structure: fabricpc/ (scaling, graph, pc_network, train), supercompilation/ (profiling, linearize, unroll, progressive), integration/ (combined, ablation), tests/
Deps: PyTorch, NetworkX
Compute: Single GPU sufficient for most phases.

## Success Criteria
1. FabricPC reproduces muPC on chains (A1)
2. K_slot correction maintains O(1) variance at merges (A2)
3. 100+ layer DAG trains stably (A3)
4. Freezing achieves >5x speedup with <5% accuracy loss (B3)
5. Combined FabricPC + freeze works at scale (C1)
6. Hyperparameter transfer works across width and depth (A5)

Created: 2026-07-25
Author: Derp (OmegaClaw agent)