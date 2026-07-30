import pytest
import jax
import jax.numpy as jnp
import numpy as np
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.graph_initialization.params_initializer import initialize_params

class TestEnergyMonotonicity:
    def test_energy_decreases(self, simple_mlp, small_batch):
        structure = simple_mlp
        key = jax.random.PRNGKey(0)
        batch_size = small_batch['x'].shape[0]
        params = initialize_params(structure, key)
        state = initialize_graph_state(structure, batch_size, key, params=params)
        result = run_inference(params, state, small_batch, structure)
        energies = [ns.energy for ns in result.nodes.values()]
        if len(energies) > 1:
            diffs = jnp.diff(jnp.array([e.mean() for e in energies]))
            assert jnp.all(diffs <= 1e-6), 'Energy increased'

class TestSettledGradients:
    def test_settled_grads_near_zero(self, simple_mlp, small_batch):
        structure = simple_mlp
        key = jax.random.PRNGKey(0)
        batch_size = small_batch['x'].shape[0]
        params = initialize_params(structure, key)
        state = initialize_graph_state(structure, batch_size, key, params=params)
        result = run_inference(params, state, small_batch, structure)
        for name, ns in result.nodes.items():
            assert jnp.max(jnp.abs(ns.latent_grad)) < 1e-3, f'Latent grad not zero for {name}'

class TestParameterminism:
    def test_deterministic_seed(self, simple_mlp, small_batch):
        structure = simple_mlp
        key = jax.random.PRNGKey(42)
        batch_size = small_batch['x'].shape[0]
        params = initialize_params(structure, key)
        s1 = initialize_graph_state(structure, batch_size, key, params=params)
        r1 = run_inference(params, s1, small_batch, structure)
        params2 = initialize_params(structure, key)
        s2 = initialize_graph_state(structure, batch_size, key, params=params2)
        r2 = run_inference(params2, s2, small_batch, structure)
        for k in r1.nodes:
            if k in r2.nodes:
                np.testing.assert_allclose(r1.nodes[k].z_latent, r2.nodes[k].z_latent, atol=1e-7)

class TestPredictionClamping:
    def test_output_clamped(self, simple_mlp, small_batch):
        structure = simple_mlp
        key = jax.random.PRNGKey(0)
        batch_size = small_batch['x'].shape[0]
        params = initialize_params(structure, key)
        state = initialize_graph_state(structure, batch_size, key, clamps={'ay': small_batch['y']}, params=params)
        result = run_inference(params, state, small_batch, structure)
        output = result.nodes.get('ay', None)
        if output is not None:
            np.testing.assert_allclose(output.z_latent, small_batch['y'], atol=1e-5)

class TestBatchConsistency:
    def test_batch_vs_single(self, simple_mlp, small_batch):
        structure = simple_mlp
        key = jax.random.PRNGKey(0)
        batch_size = small_batch['x'].shape[0]
        params = initialize_params(structure, key)
        state = initialize_graph_state(structure, batch_size, key, params=params)
        batch_result = run_inference(params, state, small_batch, structure)
        for i in range(small_batch['x'].shape[0]):
            single = {'x': small_batch['x'][i:i+1], 'y': small_batch['y'][i:i+1]}
            single_state_init = initialize_graph_state(structure, 1, key, params=params)
            single_result = run_inference(params, single_state_init, single, structure)
            for k in batch_result.nodes:
                if k in single_result.nodes:
                    np.testing.assert_allclose(batch_result.nodes[k].z_latent[i], single_result.nodes[k].z_latent[0], atol=1e-5)
