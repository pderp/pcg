"""External muP + residual block nodes for FabricPC predictive coding networks."""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotSpec
from fabricpc.nodes.base import NodeBase


class LinearResidual(NodeBase):
    """Linear node with residual (skip) connection: y = f(Wx + b) + x.
    Implements muP-compatible scaling for predictive coding networks.
    The residual path carries the input identity, counting toward muPC depth L.
    """

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[NodeState, jnp.ndarray]:
        """Forward pass with residual connection."""
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        pre_activation = jnp.zeros((batch_size,) + out_shape)
        residual = jnp.zeros((batch_size,) + out_shape)
        for edge_key, x in inputs.items():
            w = params.weights[edge_key]
            pre_activation = pre_activation + jnp.matmul(x, w)
            residual = residual + x
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]
        pre_activation = pre_activation + residual
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state, z_mu

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialize weights and biases for residual block."""
        if config is None:
            config = {}
        key_w, key_b = jax.random.split(key)
        weights_dict = {}
        rand_key_w = dict(
            zip(input_shapes.keys(), jax.random.split(key_w, len(input_shapes)))
        )
        for edge_key, in_shape in input_shapes.items():
            in_features = in_shape[-1]
            out_features = node_shape[-1]
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = weight_init(rand_key_w[edge_key], weight_shape)
        use_bias = config.get("use_bias", True)
        biases = {}
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            biases["b"] = jnp.zeros(bias_shape)
        return NodeParams(weights=weights_dict, biases=biases)
