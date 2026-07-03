import jax
import jax.numpy as jnp
import jax.random as jr

from kozax.genetic_programming import GeneticProgramming


class DummyFitness:
    def __init__(self):
        self.n_objectives = 1

    def __call__(self, candidate, data, tree_evaluator):
        del candidate, data, tree_evaluator
        return jnp.array(0.0)


def _make_strategy(population_size=6):
    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.4),
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.3),
        ("sin", lambda x: jnp.sin(x), 1, 0.3),
    ]
    variable_list = [["x0", "x1"]]

    return GeneticProgramming(
        num_generations=1,
        population_size=population_size,
        fitness_function=DummyFitness(),
        operator_list=operator_list,
        variable_list=variable_list,
        layer_sizes=jnp.array([1]),
        num_populations=1,
        max_init_depth=3,
        max_nodes=15,
        constant_optimization=False,
        max_fitness=1e6,
    )


def _assert_tree_reference_invariants(tree):
    node_ids = tree[:, 0]
    active = node_ids != 0

    child_refs = tree[:, 1:3].astype(jnp.int32)

    # Child references are either -1 or valid indices within bounds.
    within_bounds = (child_refs == -1) | ((child_refs >= 0) & (child_refs < tree.shape[0]))
    assert jnp.all(within_bounds)

    # For active nodes, children (if present) should point to lower indices in depth-first storage.
    parent_idx = jnp.arange(tree.shape[0])[:, None]
    valid_child = child_refs >= 0
    ordering_ok = (~valid_child) | (child_refs < parent_idx)
    ordering_ok = (~active[:, None]) | ordering_ok
    assert jnp.all(ordering_ok)


def test_mutate_trees_jit_and_invariants_hold():
    strategy = _make_strategy(population_size=6)
    key = jr.PRNGKey(10)
    init_key, mutate_key = jr.split(key)

    population = strategy.initialize_population(init_key)
    trees = population[0, 0]  # Shape: (num_trees, max_nodes, 4)

    keys = jr.split(mutate_key, trees.shape[0])
    reproduction_probability = 1.0
    variable_array = strategy.variable_array

    jit_mutate = jax.jit(strategy.mutate_trees)
    mutated = jit_mutate(trees, keys, reproduction_probability, variable_array)

    assert mutated.shape == trees.shape
    assert jnp.all(jnp.isfinite(mutated))

    for i in range(mutated.shape[0]):
        _assert_tree_reference_invariants(mutated[i])
