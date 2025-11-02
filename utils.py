import re

import diffrax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sympy as sp

from evaluators.sho_evaluator import SHOEvaluator
from kozax.genetic_programming import GeneticProgramming


def analyze_expression_dependencies(u_str):
    return bool(re.search(r'\ba0\b', u_str)), bool(re.search(r'\ba1\b', u_str))


def get_unique_solutions(solutions, fitness_values=None, tolerance=1e-10):
    if len(solutions) == 0:
        return (solutions, fitness_values) if fitness_values is not None else solutions

    solutions_np = np.array(solutions) if hasattr(solutions, 'device') else solutions
    unique_solutions, unique_indices = [], []

    for i, solution in enumerate(solutions_np):
        if len(solution) < 3:
            continue
        a0, a1, u = solution[0], solution[1], solution[2]
        is_duplicate = False
        u_str = str(u)
        uses_a0, uses_a1 = analyze_expression_dependencies(u_str)

        for j, unique_sol in enumerate(unique_solutions):
            ua0, ua1, uu = unique_sol[0], unique_sol[1], unique_sol[2]
            uu_str = str(uu)
            u_uses_a0, u_uses_a1 = analyze_expression_dependencies(uu_str)

            try:
                u_same = sp.simplify(sp.sympify(u_str) - sp.sympify(uu_str)) == 0
            except:
                u_same = u_str.strip() == uu_str.strip()

            if not u_same:
                continue

            variables_same = True
            if (uses_a0 or u_uses_a0) and np.abs(float(a0) - float(ua0)) >= tolerance:
                variables_same = False
            if (uses_a1 or u_uses_a1) and np.abs(float(a1) - float(ua1)) >= tolerance:
                variables_same = False

            if variables_same:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_solutions.append(solution)
            unique_indices.append(i)

    unique_solutions = np.array(unique_solutions)

    if fitness_values is not None:
        fv = np.array(fitness_values) if hasattr(fitness_values, 'device') else fitness_values
        return unique_solutions, fv[unique_indices]

    return unique_solutions


def get_data(key, env, batch_size, dt, T, param_setting, dynamic_target=False):
    init_key, noise_key, param_key = jr.split(key, 3)
    ts = jnp.arange(0, T, dt)
    x0, targets = env.sample_init_states(batch_size, ts, init_key, dynamic_target=dynamic_target)
    noise_keys = jr.split(noise_key, batch_size)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, noise_keys, params


def validate(models, strategy, data):
    models = get_unique_solutions(models)

    n_models = models.shape[0]
    padding_needed = (10 - (n_models % 10)) % 10
    if padding_needed > 0:
        last_model = models[-1:]
        padding_models = jnp.repeat(last_model, padding_needed, axis=0)
        models = jnp.concatenate([models, padding_models], axis=0)
    fitnesses = strategy.jit_eval(models, data)

    best_idx = jnp.argmin(fitnesses)

    return fitnesses[best_idx], models[best_idx]


def get_strategy(env, operator_list, dt0, reward_fn):
    variable_list = [
        # Latent memory
        [f"y{i}" for i in range(env.n_obs)] +
        [f"a{i}" for i in range(2)] +
        [f"u{i}" for i in range(env.n_control_inputs)] +

        # Control readout
        [f"a{i}" for i in range(2)]
    ]
    if env.n_targets > 0:
        if reward_fn is None:
            for var_list in variable_list:
                var_list.append("tar")
        else:
            for var_list in variable_list:
                var_list.append("r")

        layer_sizes = jnp.array([2, env.n_control_inputs])

        fitness_function = SHOEvaluator(env, 2, dt0, reward_fn=reward_fn, solver=diffrax.GeneralShARK())

        strategy = GeneticProgramming(
            fitness_function=fitness_function,
            num_generations=1,
            population_size=2,
            operator_list=operator_list,
            variable_list=variable_list,
            layer_sizes=layer_sizes,
        )
    return strategy, fitness_function
