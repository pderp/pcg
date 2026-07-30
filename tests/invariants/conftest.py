import pytest
import jax
import jax.numpy as njp
import numpy as np

from fabricpc.core.activations import ReLUActivation
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph_assembly.graph_construction import graph as build_graph
from fabricpc.nodes.linear import Linear
from fabricpc.graph_initialization.state_initializer import initialize_graph_state

@pytest.fixture
def simple_mlp():
    input_node = Linear(shape=(784,), name='input')
    hidden_node = Linear(shape=(256,), name='hidden', activation=ReLUActivation())
    output_node = Linear(shape=(10,), name='output')
    structure = build_graph(
        nodes=[input_node, hidden_node, output_node],
        edges=[
            Edge(input_node, hidden_node),
            Edge(hidden_node, output_node),
        ],
        task_map={'x': 'input', 'y': 'output'},
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    return structure

@pytest.fixture
def small_batch():
    rng = jax.random.PRNGKey(42)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.normal(rng_x, (4, 784))
    y = jax.nn.one_hot(jax.random.randint(rng_y, (4,), 0, 10), 10)
    return {'x': x, 'y': y}
