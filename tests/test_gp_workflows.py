import jax
import jax.numpy as jnp
import jax.random as jr

from kozax.genetic_programming import GeneticProgramming


class SimpleRegressionFitness:
    def __init__(self):
        self.n_objectives = 1

    def __call__(self, candidate, data, tree_evaluator):
        x, y = data

        def predict_row(row):
            # Single-tree setup: take index 0 from tree evaluator output.
            return tree_evaluator(candidate, row)[0]

        pred = jax.vmap(predict_row)(x)
        return jnp.mean((pred - y) ** 2)


class MultiObjectiveRegressionFitness:
    def __init__(self):
        self.n_objectives = 2

    def __call__(self, candidate, data, tree_evaluator):
        x, y = data

        def predict_row(row):
            return tree_evaluator(candidate, row)[0]

        pred = jax.vmap(predict_row)(x)
        mse = jnp.mean((pred - y) ** 2)
        mae = jnp.mean(jnp.abs(pred - y))
        return jnp.array([mse, mae])


def _toy_data(n=16):
    x0 = jnp.linspace(-1.0, 1.0, n)
    x1 = jnp.linspace(1.0, -1.0, n)
    x = jnp.stack([x0, x1], axis=-1)
    y = x0 + 0.5 * x1
    return x, y


def _make_strategy(
    fitness_function,
    num_generations=2,
    num_populations=1,
    population_size=6,
    complexity_objective=False,
    constant_optimization=False,
    constant_optimization_steps=1,
    optimize_constants_elite=100,
    constant_step_size=0.1,
    operator_list=None,
):
    if operator_list is None:
        operator_list = [{"string": "+", "fn": lambda x, y: jnp.add(x, y), "arity": 2, "prob": 0.1}, 
                                     {"string": "-", "fn": lambda x, y: jnp.subtract(x, y), "arity": 2, "prob": 0.1},
                                     {"string": "*", "fn": lambda x, y: jnp.multiply(x, y), "arity": 2, "prob": 0.1},
                                     ]
    variable_list = [["x0", "x1"]]

    return GeneticProgramming(
        num_generations=num_generations,
        population_size=population_size,
        fitness_function=fitness_function,
        operator_list=operator_list,
        variable_list=variable_list,
        layer_sizes=jnp.array([1]),
        num_populations=num_populations,
        max_init_depth=3,
        max_nodes=15,
        complexity_objective=complexity_objective,
        constant_sd=1.0,
        constant_optimization=constant_optimization,
        constant_optimization_steps=constant_optimization_steps,
        optimize_constants_elite=optimize_constants_elite,
        constant_step_size=constant_step_size,
        max_fitness=1e6,
    )


def test_initialize_evaluate_evolve_single_objective_smoke():
    key = jr.PRNGKey(0)
    init_key, eval_key, evolve_key = jr.split(key, 3)

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=2, num_populations=1, population_size=6)

    population = strategy.initialize_population(init_key)
    assert population.shape[0] == 1
    assert population.shape[1] == 6

    data = _toy_data(n=16)
    fitness_values, evaluated_population = strategy.evaluate_population(population, data, eval_key)

    assert fitness_values.shape == (1, 6, 1)
    assert evaluated_population.shape == population.shape
    assert jnp.all(jnp.isfinite(fitness_values))

    offspring = strategy.evolve_population(evaluated_population, fitness_values, evolve_key)
    assert offspring.shape == population.shape


def test_multi_objective_evaluation_shape_and_finiteness():
    key = jr.PRNGKey(1)
    init_key, eval_key = jr.split(key, 2)

    fitness = MultiObjectiveRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=2, num_populations=1, population_size=6)

    population = strategy.initialize_population(init_key)
    fitness_values, _ = strategy.evaluate_population(population, _toy_data(12), eval_key)

    assert fitness_values.shape == (1, 6, 2)
    assert jnp.all(jnp.isfinite(fitness_values))


def test_fit_smoke_runs_end_to_end():
    key = jr.PRNGKey(2)
    fit_key = jr.PRNGKey(3)

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=2, num_populations=1, population_size=6)

    strategy.fit(fit_key, _toy_data(10), verbose=0)

    pareto_fitness, pareto_solutions = strategy.pareto_front
    assert pareto_solutions.shape[1:] == (1, strategy.max_nodes, 4)
    assert jnp.all(jnp.isfinite(pareto_fitness))


def test_expression_to_string_returns_symbolic_output():
    key = jr.PRNGKey(4)

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=1, num_populations=1, population_size=6)

    population = strategy.initialize_population(key)
    candidate = population[0, 0]
    expr = strategy.expression_to_string(candidate)

    assert str(expr)


def test_constant_optimization_updates_values_and_fitness():
    key = jr.PRNGKey(5)
    init_key, eval_key = jr.split(key, 2)
    data = _toy_data(n=16)

    fitness = SimpleRegressionFitness()
    strategy_no_opt = _make_strategy(
        fitness,
        num_generations=1,
        num_populations=1,
        population_size=6,
        constant_optimization=False,
    )
    strategy_opt = _make_strategy(
        fitness,
        num_generations=1,
        num_populations=1,
        population_size=6,
        constant_optimization=True,
        constant_optimization_steps=3,
        optimize_constants_elite=6,
        constant_step_size=0.05,
    )

    # Use the same initial population to isolate the effect of constant optimization.
    population = strategy_no_opt.initialize_population(init_key)

    fitness_no_opt, pop_no_opt = strategy_no_opt.evaluate_population(population, data, eval_key)
    fitness_opt, pop_opt = strategy_opt.evaluate_population(population, data, eval_key)

    values_no_opt = pop_no_opt[..., -1]
    values_opt = pop_opt[..., -1]

    assert jnp.any(jnp.abs(values_opt - values_no_opt) > 1e-8)
    assert jnp.any(jnp.abs(fitness_opt - fitness_no_opt) > 1e-10)
    assert jnp.all(jnp.isfinite(fitness_opt))


def test_initialized_trees_have_no_internal_empty_rows():
    key = jr.PRNGKey(6)

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=1, num_populations=1, population_size=8)

    population = strategy.initialize_population(key)

    # Trees are stored in postorder with active nodes packed at the end.
    # Therefore empty rows are allowed only as a contiguous prefix.
    empty_mask = population[..., 0] == 0
    non_empty_seen = jnp.cumsum((~empty_mask).astype(jnp.int32), axis=-1) > 0
    internal_empty = empty_mask & non_empty_seen

    assert not bool(jnp.any(internal_empty))


def test_complexity_objective_multiobjective_evolve_smoke():
    key = jr.PRNGKey(7)
    init_key, eval_key, evolve_key = jr.split(key, 3)

    fitness = MultiObjectiveRegressionFitness()
    strategy = _make_strategy(
        fitness,
        num_generations=2,
        num_populations=1,
        population_size=6,
        complexity_objective=True,
    )

    population = strategy.initialize_population(init_key)
    fitness_values, evaluated_population = strategy.evaluate_population(population, _toy_data(12), eval_key)

    # evaluate_population keeps original number of fitness objectives
    assert fitness_values.shape == (1, 6, 2)
    offspring = strategy.evolve_population(evaluated_population, fitness_values, evolve_key)
    assert offspring.shape == population.shape

def test_complexity_objective_singleobjective_evolve_smoke():
    key = jr.PRNGKey(7)
    init_key, eval_key, evolve_key = jr.split(key, 3)

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(
        fitness,
        num_generations=2,
        num_populations=1,
        population_size=6,
        complexity_objective=True,
    )

    population = strategy.initialize_population(init_key)
    fitness_values, evaluated_population = strategy.evaluate_population(population, _toy_data(12), eval_key)

    # evaluate_population keeps original number of fitness objectives
    assert fitness_values.shape == (1, 6, 1)
    offspring = strategy.evolve_population(evaluated_population, fitness_values, evolve_key)
    assert offspring.shape == population.shape


def test_higher_arity_operators_initialize_evaluate_evolve_and_structure():
    key = jr.PRNGKey(8)
    init_key, eval_key, evolve_key = jr.split(key, 3)

    operator_list = [
        {"string":"sum3", "fn": lambda x, y, z: x + y + z, "arity": 3, "prob": 0.34},
        {"string":"+", "fn": lambda x, y: x + y, "arity": 2, "prob": 0.33},
        {"string":"neg", "fn": lambda x: -x, "arity": 1, "prob": 0.33},
    ]

    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(
        fitness,
        num_generations=1,
        num_populations=1,
        population_size=8,
        operator_list=operator_list,
    )

    population = strategy.initialize_population(init_key)
    assert population.shape[-1] == 2 + strategy.max_arity

    fitness_values, evaluated_population = strategy.evaluate_population(population, _toy_data(12), eval_key)
    assert fitness_values.shape == (1, 8, 1)

    offspring = strategy.evolve_population(evaluated_population, fitness_values, evolve_key)
    assert offspring.shape == population.shape

    # In postorder storage, empty rows must be a contiguous prefix.
    empty_mask = offspring[..., 0] == 0
    non_empty_seen = jnp.cumsum((~empty_mask).astype(jnp.int32), axis=-1) > 0
    internal_empty = empty_mask & non_empty_seen
    assert not bool(jnp.any(internal_empty))

    # Child references must point to non-empty rows.
    refs = offspring[..., 1:-1].astype(jnp.int32)
    has_ref = refs >= 0
    safe_refs = jnp.where(has_ref, refs, 0)
    ref_fidx = jax.vmap(lambda ind: jnp.take_along_axis(offspring[..., 0], ind, axis=-1), in_axes=[-1], out_axes=-1)(safe_refs)
    dangling = has_ref & (ref_fidx == 0)
    assert not bool(jnp.any(dangling))

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
    fitness = SimpleRegressionFitness()
    strategy = _make_strategy(fitness, num_generations=2, num_populations=1, population_size=100)
    key = jr.PRNGKey(10)
    init_key, mutate_key = jr.split(key)

    population = strategy.initialize_population(init_key)
    trees = population[0, 0]  # Shape: (num_trees, max_nodes, 4)

    keys = jr.split(mutate_key, trees.shape[0])
    reproduction_probability = 1.0
    variable_array = strategy.variable_array
    constand_sd_array = strategy.constant_sd_array

    jit_mutate = jax.jit(strategy.mutate_trees)
    mutated = jit_mutate(trees, keys, reproduction_probability, variable_array, constand_sd_array)

    assert mutated.shape == trees.shape
    assert jnp.all(jnp.isfinite(mutated))

    for i in range(mutated.shape[0]):
        _assert_tree_reference_invariants(mutated[i])