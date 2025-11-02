import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr

from environments.harmonic_oscillator import HarmonicOscillator
from evaluators.sho_evaluator import SHOEvaluator
from kozax.genetic_programming import GeneticProgramming
from utils import get_data


def run_experiment(args, operator_list=None, reward_fn=None, save_path=None):
    device = jax.devices()[0].platform

    # env parameters
    param_setting = args.get("param_setting")
    batch_size = args.get("batch_size")
    n_obs = args.get("n_obs")
    process_noise = args.get("process_noise")
    obs_noise = args.get("obs_noise")
    dt = 0.2
    T = 40

    # GP parameters
    num_generations = args.get("num_generations")
    population_size = args.get("population_size")
    num_populations = args.get("num_populations")
    if operator_list is None:
        operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.3),
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5)
        ]
    variable_list = [
        # Latent memory
        [f"y{i}" for i in range(env.n_obs)] +
        [f"a{i}" for i in range(state_size, env.n_obs)] +
        [f"u{i}" for i in range(env.n_control_inputs)] +
        # Control readout
        [f"a{i}" for i in range(state_size)]
    ]
    if reward_fn is None:
        for var_list in variable_list:
            var_list.append("tar")
    else:
        for var_list in variable_list:
            var_list.append("r")

    # Evaluation parameters
    state_size = 2
    layer_sizes = jnp.array([state_size, env.n_control_inputs])
    dt0 = 0.02
    max_steps = 2000

    env = HarmonicOscillator(process_noise, obs_noise, n_obs)

    fitness_function = SHOEvaluator(env, state_size, dt0, reward_fn=reward_fn, solver=diffrax.GeneralShARK(), max_steps=max_steps)
    strategy = GeneticProgramming(
        num_generations,
        population_size, 
        fitness_function, 
        operator_list, 
        variable_list,
        layer_sizes,
        num_populations,
        device_type=device
    )

    key = jr.PRNGKey(100)
    key, init_key, data_key = jr.split(key, 3)

    best_fitnesses = []

    # Get the data
    data = get_data(data_key, env, batch_size, dt, T, param_setting)

    # Run the evolution
    best_fitnesses = strategy.fit(init_key, data)

    return strategy, best_fitnesses