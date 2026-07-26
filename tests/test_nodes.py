import jax
import jax.numpy as jnp
import pytest
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotSpec
from fabricpc.nodes.base import NodeBase
from pcg_nodes.nodes import LinearResidual


def test_get_slots():
    slots = LinearResidual.get_slots()
    assert "in" in slots
    assert slots["in"].is_multi_input is True


def test_initialize_params():
    key = jax.random.PRNGKey(0)
    node_shape = (8,)
    input_shapes = {"in:0": (4,)}
    weight_init = jax.nn.initializers.normal(0.01)
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init)
    assert "in:0" in params.weights
    assert params.weights["in:0"].shape == (4, 8)
    assert "b" in params.biases
    assert params.biases["b"].shape[-1] == 8


def test_initialize_params_no_bias():
    key = jax.random.PRNGKey(0)
    node_shape = (8,)
    input_shapes = {"in:0": (4,)}
    weight_init = jax.nn.initializers.normal(0.01)
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init, config={"use_bias": False})
    assert "in:0" in params.weights
    assert len(params.biases) == 0
