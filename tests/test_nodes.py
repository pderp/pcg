import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.nodes.base import NodeBase
from pcg_nodes.nodes import LinearResidual


def make_node_info(shape, activation=None, energy=None, node_class=None):
    return NodeInfo(
        name='test_residual',
        shape=shape,
        node_type='linear_residual',
        node_class=node_class or LinearResidual,
        node_config={},
        activation=activation or IdentityActivation(),
        energy=energy or GaussianEnergy(),
        latent_init=None,
        weight_init=None,
        slots=LinearResidual.get_slots(),
        in_degree=1,
        out_degree=0,
        in_edges=('edge0',),
        out_edges=(),
    )

def make_node_state(batch_size, shape):
    z_latent = jnp.zeros((batch_size,) + shape)
    z_mu = jnp.zeros((batch_size,) + shape)
    error = jnp.zeros((batch_size,) + shape)
    energy = jnp.zeros((batch_size,))
    latent_grad = jnp.zeros((batch_size,) + shape)
    return NodeState(
        z_latent=z_latent,
        z_mu=z_mu,
        error=error,
        energy=energy,
        latent_grad=latent_grad,
    )


def test_get_slots():
    slots = LinearResidual.get_slots()
    assert 'in' in slots
    assert slots['in'].is_multi_input == True
    assert slots['in'].is_variance_scalable == False


def test_initialize_params():
    key = jax.random.PRNGKey(42)
    node_shape = (4,)
    input_shapes = {'edge0': (8,)}
    weight_init = NormalInitializer()
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init)
    assert 'edge0' in params.weights
    assert params.weights['edge0'].shape == (8, 4)
    assert 'b' in params.biases


def test_forward_single_input():
    key = jax.random.PRNGKey(42)
    batch_size = 3
    node_shape = (4,)
    input_shapes = {'edge0': (8,)}
    weight_init = NormalInitializer()
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init)
    x = jax.random.normal(key, (batch_size, 8))
    inputs = {'edge0': x}
    state = make_node_state(batch_size, node_shape)
    node_info = make_node_info(node_shape)
    new_state = LinearResidual.forward(params, inputs, state, node_info)
    assert isinstance(new_state, NodeState)
    assert new_state.z_mu.shape == (batch_size, 4)
    np.testing.assert_allclose(np.array(new_state.error), np.array(state.z_latent - new_state.z_mu))
    assert new_state.energy.shape == (batch_size,)


def test_forward_multi_input():
    key = jax.random.PRNGKey(42)
    batch_size = 2
    node_shape = (4,)
    input_shapes = {'edge0': (8,), 'edge1': (6,)}
    weight_init = NormalInitializer()
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init)
    key_x0, key_x1 = jax.random.split(key)
    x0 = jax.random.normal(key_x0, (batch_size, 8))
    x1 = jax.random.normal(key_x1, (batch_size, 6))
    inputs = {'edge0': x0, 'edge1': x1}
    state = make_node_state(batch_size, node_shape)
    node_info = make_node_info(node_shape)
    new_state = LinearResidual.forward(params, inputs, state, node_info)
    assert isinstance(new_state, NodeState)
    assert new_state.z_mu.shape == (batch_size, 4)
    assert new_state.energy.shape == (batch_size,)


def test_mup_scaling_config():
    key = jax.random.PRNGKey(42)
    batch_size = 2
    node_shape = (4,)
    input_shapes = {'edge0': (8,)}
    weight_init = NormalInitializer()
    config = {'use_bias': True, 'flatten_input': False}
    params = LinearResidual.initialize_params(key, node_shape, input_shapes, weight_init, config)
    assert params.weights['edge0'].shape == (8, 4)
    assert 'b' in params.biases
    x = jax.random.normal(key, (batch_size, 8))
    inputs = {'edge0': x}
    state = make_node_state(batch_size, node_shape)
    node_info = make_node_info(node_shape)
    new_state = LinearResidual.forward(params, inputs, state, node_info)
    assert new_state.z_mu.shape == (batch_size, 4)
