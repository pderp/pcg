# 2nd Stage Development Plan: GPU-Efficient FabricPC Implementation

## Context
This plan synthesizes four key documents:
1. pc_gpu_fabricpc.pdf - GPU efficiency framework for predictive coding: 5-layer architecture, performance projections (10-300x speedup), agent-driven development protocol
2. pc_frozen_supercompilation.pdf - Frozen supercompilation: compile PC graphs to optimized static schedules
3. Causal_Memory_and_Credit_Protocol.pdf - Causal memory and credit assignment for PC networks
4. scaling_through_the_depth_barrier-PC_and_backprop.pdf - Scaling analysis comparing PC vs backprop

## Current Repository State
- src/pcg_nodes/ contains node implementations (e.g., LinearResidual) using JAX
- Nodes follow FabricPC NodeBase pattern with forward(), get_slots(), initialize_params()
- FabricPC has been separated into its own repo (https://github.com/trueagi-io/FabricPC)
- No execution engine, scheduler, or GPU optimization layer exists yet in pcg

## Development Phases

### Phase 0: Verification Substrate (V0)
Goal: Build machine-checkable correctness invariants before any optimization.
- [ ] 0.1 Create tests/invariants/ directory with test harness
- [ ] 0.2 Implement energy-descent invariant: energy must monotonically decrease across settle ticks
- [ ] 0.3 Implement feedforward-init identity test: unclamped discriminative inference from feedforward init equals plain forward pass
- [ ] 0.4 Implement PC-vs-BP parity test: same graph, same task, verify PC converges to comparable accuracy
- [ ] 0.5 Implement numeric gradient check for local weight updates

### Phase 1: Lowering and Packing (Framework Layer 1)
Goal: Separate logical graph from execution schedule.
- [ ] 1.1 Define GraphStructure IR: adjacency list, node types, shapes, parameter layouts
- [ ] 1.2 Implement LoweringPass: traverse graph, classify nodes by type, group same-type nodes
- [ ] 1.3 Implement PackingPass: concatenate weights into block-diagonal matrices, pack states into contiguous arrays
- [ ] 1.4 Implement ExecutionPlan dataclass: packed arrays, kernel schedule, memory layout
- [ ] 1.5 Verify: lowered plan produces identical results to per-node execution

### Phase 2: Schedules as First-Class Objects (Framework Layer 2)
Goal: Compile settle loop into a single XLA program with fused kernels.
- [ ] 2.1 Replace lax.fori + per-node loops with lax.scan over precomputed schedule
- [ ] 2.2 Implement Jacobi sweep schedule: all nodes update simultaneously from packed state
- [ ] 2.3 Fuse per-tick chain into minimal kernels
- [ ] 2.4 Benchmark: measure kernel launch count reduction and GPU utilization
- [ ] 2.5 Threshold: <=5 kernel launches per settle tick

### Phase 3: Warm Starts and Adaptive Termination (Framework Layer 3)
Goal: Reduce effective T from 8-100 to 3-5.
- [ ] 3.1 Implement feedforward initialization: topological-order sweep seeds latents
- [ ] 3.2 Implement temporal carryover: persist settled state across consecutive inputs
- [ ] 3.3 Implement adaptive stopping: energy-decrement tolerance with hard cap
- [ ] 3.4 Verify: feedforward-init identity test passes (Phase 0.3)
- [ ] 3.5 Benchmark: measure realized T on MNIST MLP and transformer configs

### Phase 4: Solvers, Precision, and Sparsity (Framework Layer 4)
Goal: Make each tick cheaper and reduce ticks via better solvers.
- [ ] 4.1 Add momentum on state vector (Nesterov or heavy-ball)
- [ ] 4.2 Implement Anderson acceleration with energy-descent fallback
- [ ] 4.3 Switch settle computation from fp32 to bf16/fp16 with fp32 accumulation
- [ ] 4.4 Implement block-granular error-gated sparsity
- [ ] 4.5 Implement block-sparse GEMM path for masked settle updates
- [ ] 4.6 Benchmark: measure FLOP reduction and accuracy preservation

### Phase 5: Incremental PC (iPC) Integration
Goal: Collapse inner (settle) and outer (weight update) loops.
- [ ] 5.1 Implement iPC training mode: one latent tick + one weight tick per step
- [ ] 5.2 Validate by task quality (not state equivalence to settled PC)
- [ ] 5.3 Benchmark: iPC wall-clock per iteration vs BP, accuracy curves
- [ ] 5.4 Threshold: iPC per-iteration cost within 2x of BP

### Phase 6: Frozen Supercompilation
Goal: Compile PC graphs to optimized static schedules.
- [ ] 6.1 Implement graph specialization: generate specialized JAX code from fixed GraphStructure
- [ ] 6.2 Implement schedule freezing: precompute all gather indices, GEMM shapes, control flow
- [ ] 6.3 Implement partial evaluation: specialize on graph topology, batch size, settle schedule
- [ ] 6.4 Verify: frozen schedule produces identical results to dynamic execution
- [ ] 6.5 Benchmark: compile time, executable size, runtime vs dynamic path

### Phase 7: Graph-Sharded Multi-Device Execution (Framework Layer 5)
Goal: Scale beyond single-GPU memory without pipeline bubbles or gradient all-reduce.
- [ ] 7.1 Implement graph partitioning: min-cut on boundary tensor bytes per tick
- [ ] 7.2 Implement bulk-synchronous per-tick sharding with shard_map + ppermute halo exchange
- [ ] 7.3 Implement double-buffered halos for communication-compute overlap
- [ ] 7.4 Verify: sharded execution matches single-device results within floating point tolerance
- [ ] 7.5 Benchmark: scaling efficiency across 2, 4, 8 devices

### Phase 8: Causal Memory and Credit Protocol Integration
Goal: Integrate causal memory and credit assignment for continual/streaming learning.
- [ ] 8.1 Study causal memory protocol from PDF: trace-based credit assignment
- [ ] 8.2 Implement causal trace buffer: record (state, error, weight-delta) tuples with timestamps
- [ ] 8.3 Implement credit propagation: distribute credit backward through causal traces
- [ ] 8.4 Integrate with temporal carryover warm starts (Phase 3.2)
- [ ] 8.5 Benchmark: continual learning task with catastrophic forgetting metric

## Agent Protocol
Each phase follows this protocol:
1. Implement the phase subtasks
2. Verify all Phase 0 invariants still pass
3. Benchmark against defined thresholds
4. Commit with descriptive message
5. Report results and await confirmation before next phase

## Risk Register
| Risk | Mitigation |
|------|------------|
| XLA compilation fails on large heterarchical graphs | Implement graph-size-based chunking; test incrementally |
| Anderson acceleration diverges | Energy-descent fallback; revert to gradient descent |
| Block-sparse GEMM not available in JAX | Use masked dense GEMM first; Triton/Pallas kernel later |
| iPC changes convergence behavior | Validate by task quality, not state equivalence |
| Multi-device sharding introduces numerical drift | Use deterministic reductions; tolerance-based checks |

## Sequencing
- Critical path: V0 -> P1 -> P2 -> P3 -> P5 (single-GPU competitive with BP)
- Parallelizable: P4 (solvers/sparsity) after P2; P6 (supercompilation) after P3
- Research: P7 (multi-device) and P8 (causal memory) after P5 validated

## Success Criteria
- Single-GPU PC training within 2.5x of BP wall-clock on comparable graphs
- Effective T <= 5 with warm starts and adaptive termination
- GPU utilization > 50% for graphs of many small nodes
- iPC per-iteration cost within 2x of BP
- All Phase 0 invariants pass at every commit