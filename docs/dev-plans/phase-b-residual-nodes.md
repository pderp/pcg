# PCG FabricPC Experiments - Dev Plan

## Phase B: External muP + Residual Nodes

### Goal
Build external FabricPC nodes implementing muP-compatible residual blocks.

### Components
- `pcg_nodes/nodes.py`: LinearResidual node (get_slots, forward, initialize_params)
- `pcg_nodes/__init__.py`: Package init exporting LinearResidual
- `tests/test_nodes.py`: Tests for slot specs, param init (with/without bias)

### FabricPC Node Contract
External nodes inherit from NodeBase and implement:
- `get_slots() -> Dict[str, SlotSpec]`: Define input/output slots
- `initialize_params(key, node_shape, input_shapes, weight_init, config) -> NodeParams`
- `forward(params, inputs, state, node_info) -> (NodeState, jnp.ndarray)`

### Next Steps
- Add forward() test with dummy NodeState/NodeInfo
- Add muP scaling config support
- Add multi-input residual test
