"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
print("These device(s) are detected: ", jax.devices())
from jax import Array

import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import optax
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import sympy
from typing import Tuple, Callable
import time

import numpy as np

from kozax.genetic_operators.crossover import crossover_trees
from kozax.genetic_operators.initialization import sample_population, sample_tree
from kozax.genetic_operators.mutation import initialize_mutation_functions
from kozax.genetic_operators.reproduction import evolve_populations, evolve_population

# Function containers
def lambda_operator_arity1(f):
    return lambda x, y, _data: f(x)

def lambda_operator_arity2(f):
    return lambda x, y, _data: f(x, y)

def lambda_leaf(i):
    return lambda x, y, _data: _data[i]

class GeneticProgramming:
    """Genetic programming strategy of symbolic expressions.

    Parameters
    ----------
    num_generations : int
        The number of generations over which to evolve the population.
    population_size : int
        Number of candidates in the population.
    fitness_function : Callable
        Function that evaluates a candidate and assigns a fitness.
    operator_list : list, optional
        List of operators that can be included in the trees.
    variable_list : list, optional
        List of variables that can be included in the trees, which can vary between layers of a candidate.
    layer_sizes : Array, optional
        Size of each layer in a candidate.
    num_populations : int, optional
        Number of subpopulations.
    max_init_depth : int, optional
        Highest depth of a tree during initialization.
    max_nodes : int, optional
        Maximum number of nodes in a tree.
    device_type : str, optional
        Type of device on which the evaluation and evolution takes place.
    constant_sd : float, optional
        Standard deviation to sample constants.
    migration_period : int, optional
        Number of generations after which populations are migrated.
    migration_size : float, optional
        Number of candidates to migrate.
    tournament_size: int, optional
        Size of the tournament.
    constant_optimization_method : str, optional
        Method for optimizing constants. Options are "evolution", "gradient", or None.
    constant_optimization_N_offspring : int, optional
        Number of offspring for Evolution Strategies.
    constant_optimization_steps : int, optional
        The number of iterations to optimize the constants.
    start_constant_optimization : int, optional
        After which generation the optimization of constants should start.
    optimize_constants_elite : int, optional
        The number of the best candidates of which the constants are optimized.
    optimizer_class : optional
        optimizer for constant optimization.
    constant_step_size_init : float, optional
        Initial step size for the optimizer.
    constant_step_size_decay : float, optional
        Decay rate for the step size.
    max_fitness : float, optional
        Maximum fitness value.
    reproduction_probability_factors : Tuple[float], optional
        The reproduction probability for each subpopulation.
    crossover_probability_factors : Tuple[float], optional
        The crossover probability for each subpopulation.
    mutation_probability_factors : Tuple[float], optional
        The mutation probability for each subpopulation.
    sample_probability_factors : Tuple[float], optional
        The probability to sample a new candidate for each subpopulation.
    """

    def __init__(self, 
                 num_generations: int, 
                 population_size: int, 
                 fitness_function: Callable, 
                 operator_list: list = None,
                 variable_list: list = None,
                 layer_sizes: Array = jnp.array([1]),
                 num_populations: int = 1,
                 max_init_depth: int = 4,
                 max_nodes: int = 15,
                 device_type: str = 'cpu',
                 complexity_objective: bool = False,
                 constant_sd: float = 1.0,
                 migration_period: int = 5,
                 migration_size: float = 10,
                 tournament_size: int = 7,
                 constant_optimization_method: str = None,
                 constant_optimization_N_offspring: int = 50,
                 constant_optimization_steps: int = 1,
                 start_constant_optimization: int = 0,
                 optimize_constants_elite: int = 100,
                 optimizer_class = optax.adam,
                 constant_step_size_init: float = 0.1,
                 constant_step_size_decay: float = 0.99,
                 max_fitness: float = 1e8,
                 reproduction_probability_factors: float | Tuple[float] = (0.75, 0.75),
                 crossover_probability_factors: float | Tuple[float] = (0.9, 0.1),
                 mutation_probability_factors: float | Tuple[float] = (0.1, 0.9),
                 sample_probability_factors: float | Tuple[float] = (0.0, 0.0)) -> None:
        
        self.layer_sizes = layer_sizes
        assert num_populations > 0, "The number of populations should be larger than 0"
        self.num_populations = num_populations
        assert population_size > 0 and population_size % 2 == 0, "The population_size should be larger than 0 and an even number"
        self.population_size = population_size
        assert max_init_depth > 0, "The max initial depth should be larger than 0"
        self.max_init_depth = max_init_depth
        assert max_nodes > 0, "The max number of nodes should be larger than 0"
        self.max_nodes = max_nodes
        self.num_trees = jnp.sum(self.layer_sizes).item()
        assert self.num_trees > 0, "The number of trees should be larger than 0"
        assert tournament_size > 1, "The tournament size should be larger than 1"
        self.tournament_size = tournament_size

        assert num_generations > 0, "The number of generations should be larger than 0"
        self.num_generations = num_generations

        self.complexity_objective = complexity_objective
        assert constant_sd > 0, "The standard deviation of the constants should be larger than 0"
        self.constant_sd = constant_sd

        assert migration_period > 1, "The migration period should be larger than 1"
        self.migration_period = migration_period
        assert isinstance(migration_size, int), "The migration size should be an integer"
        self.migration_size = migration_size
        
        if isinstance(crossover_probability_factors, float):
            crossover_probability_factors = (crossover_probability_factors, crossover_probability_factors)
        assert (crossover_probability_factors[0] >= 0) & (crossover_probability_factors[1] >= 0), "The crossover probability should not be negative"

        if isinstance(mutation_probability_factors, float):
            mutation_probability_factors = (mutation_probability_factors, mutation_probability_factors)
        assert (mutation_probability_factors[0] >= 0) & (mutation_probability_factors[1] >= 0), "The mutation probability should not be negative"

        if isinstance(sample_probability_factors, float):
            sample_probability_factors = (sample_probability_factors, sample_probability_factors)
        assert (sample_probability_factors[0] >= 0) & (sample_probability_factors[1] >= 0), "The sample probability should not be negative"

        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(*crossover_probability_factors, self.num_populations),
                                                           jnp.linspace(*mutation_probability_factors, self.num_populations),
                                                           jnp.linspace(*sample_probability_factors, self.num_populations)]).T
        
        if isinstance(reproduction_probability_factors, float):
            reproduction_probability_factors = (reproduction_probability_factors, reproduction_probability_factors)
        assert (reproduction_probability_factors[0] > 0) & (reproduction_probability_factors[1] > 0), "The reproduction probability should be larger than 0"

        self.reproduction_probabilities = jnp.linspace(*reproduction_probability_factors, self.num_populations)

        self.max_fitness = max_fitness
        self.complexities = jnp.arange(self.num_trees * self.max_nodes)

        # Create a mapping from indices in breadth first space to depth first space
        self.map_b_to_d = self.map_breadth_indices_to_depth_indices(jnp.maximum(self.max_init_depth, 3))

        # Define general tree structures
        self.tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:, None], reps=(1, 4))
        self.empty_tree = jnp.tile(jnp.array([0.0, -1.0, -1.0, 0.0]), (self.max_nodes, 1))

        self.initialize_node_library(operator_list, variable_list)

        print(f"Input data should be formatted as: {[self.node_to_string[i.item()] for i in self.variable_indices[1:]]}.")

        # Define jittable reproduction functions
        self.sample_args = (self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.constant_sd, 
                            self.map_b_to_d)
                
        self.sample_tree = partial(sample_tree,
                                   max_nodes=self.max_nodes, 
                                   tree_size=2**jnp.maximum(self.max_init_depth, 3) - 1,
                                   args=self.sample_args)
        
        self.sample_population = partial(sample_population, 
                                         num_trees=self.num_trees, 
                                         max_init_depth=self.max_init_depth, 
                                         variable_array=self.variable_array,
                                         sample_function=self.sample_tree)

        self.mutate_args = (self.sample_tree, 
                            self.max_nodes, 
                            self.max_init_depth, 
                            self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.constant_sd)
        
        self.mutate_trees = initialize_mutation_functions(self.mutate_args)

        self.partial_crossover = partial(crossover_trees, 
                                         operator_indices=self.operator_indices, 
                                         max_nodes=self.max_nodes)

        self.reproduction_functions = [self.partial_crossover, self.mutate_pair, self.sample_pair]

        self.jit_evolve_population = jax.jit(jax.vmap(partial(evolve_population, 
                                                     reproduction_functions=self.reproduction_functions, 
                                                     num_trees=self.num_trees, 
                                                     population_size=population_size,
                                                     tournament_size=self.tournament_size),
                                                    in_axes=[0, 0, 0, 0, 0, None]))
        
        self.jit_simplify_constants = jax.jit(jax.vmap(jax.vmap(jax.vmap(self.simplify_constants))))

        # Define partial fitness function for evaluation
        self.jit_evaluate_row_from_tree = partial(self.evaluate_row_from_tree, node_function_list=self.node_function_list)
        self.partial_fitness_function = lambda constants, nodes, data: fitness_function(jnp.concatenate([nodes, constants], axis=-1), data, self.tree_evaluator)

        # Define parallel evaluation functions
        self.vmap_trees = jax.vmap(self.partial_fitness_function, in_axes=[0, 0, None])
        self.vmap_gradients = jax.vmap(jax.value_and_grad(self.partial_fitness_function), in_axes=[0, 0, None])

        assert device_type in ["cpu", "gpu", "tpu"], "The device type is not supported"
        devices = mesh_utils.create_device_mesh((len(jax.devices(device_type)),))
        self.mesh = Mesh(devices, axis_names=('i'))
        self.data_mesh = NamedSharding(self.mesh, P())
        
        # Define hyperparameters for constant optimization
        self.constant_step_size_init = constant_step_size_init
        self.constant_step_size_decay = constant_step_size_decay
        self.optimize_constants_elite = optimize_constants_elite
        self.start_constant_optimization = start_constant_optimization
        self.optimizer_class = optimizer_class

        assert constant_optimization_method in ["evolution", "gradient", None], "This optimization method is not implemented"
        if constant_optimization_method == "evolution":
            assert constant_optimization_N_offspring > 0, "The offspring size for constant optimization should be larger than 0"
            self.constant_optimization_steps = constant_optimization_steps
            self.optimize_constants_function = jax.vmap(partial(self.optimize_constants_with_evolution, n_iterations=constant_optimization_steps), in_axes=[0, None, 0, None])
            self.n_offspring = constant_optimization_N_offspring
            self.constant_optimization = True
        elif constant_optimization_method == "gradient":
            self.optimize_constants_function = partial(self.optimize_constants_with_gradients, n_epoch=constant_optimization_steps)
            self.constant_optimization = True
            
        else:
            self.optimize_constants_function = None
            self.constant_optimization = False

        # Define sharded functions for evaluation and optimization
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=P('i'), check_rep=False)
        def shard_eval(array, data):
            fitness = self.vmap_trees(array[..., 3:], array[..., :3], data)

            # Regularize invalid solutions
            nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
            fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape) * self.max_fitness, fitness)
            
            return jnp.minimum(fitness, self.max_fitness * jnp.ones_like(fitness))
            
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None), P('i'), P()), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_optimize(array, data, keys, step_size):
            result, _array = self.optimize_constants_function(array, data, keys, step_size)
            return result, _array
        
        self.jit_eval = jax.jit(shard_eval)
        self.jit_optimize = jax.jit(shard_optimize)

        self.reset()
        
    def _warm_up_jit_functions(self, dummy_population: Array, data: Tuple) -> None:
        """
        Warm up JIT compilation with actual data shapes.

        Parameters
        ----------
        dummy_population : Array
            A JAX array representing a sample population with the same
            shape as will be used in actual evolution.
        data : Tuple
            A tuple containing the data used for evaluating fitness functions. This method prints the compilation start and completion time as a diagnostic.

        Returns
        -------
        None            
        """

        print("Compiling code for evaluation and evolution...")
        start = time.time()

        dummy_key = jr.PRNGKey(0)
        
        # Use actual data for compilation
        flat_populations = dummy_population.reshape(self.num_populations * self.population_size, *dummy_population.shape[2:])
        dummy_fitness, dummy_population = self.evaluate_population(dummy_population, data, dummy_key)

        self.jit_simplify_constants(dummy_population)
        
        # Warm up evolve function
        dummy_population = self.evolve_population(dummy_population, dummy_fitness, dummy_key)
        jax.block_until_ready(dummy_population)
        print(f"Finished compilation in {(time.time() - start):.2f} seconds")

    def map_breadth_indices_to_depth_indices(self, depth: int) -> Array:
        """
        Creates a mapping from the breadth-first index to depth-first index given a depth.

        Parameters
        ----------
        depth : int
            Depth of the tree.

        Returns
        -------
        Array
            Index mapping.
        """
        max_nodes = 2**depth - 1
        current_depth = 0
        map_b_to_d = jnp.zeros(max_nodes)

        for i in range(max_nodes):
            if i > 0:
                parent = (i + (i % 2) - 2) // 2  # Determine parent position
                value = map_b_to_d[parent]
                if (i % 2) == 0:  # Right child
                    new_value = value + 2**(depth - current_depth + 1)
                else:  # Left child
                    new_value = value + 1
                map_b_to_d = map_b_to_d.at[i].set(new_value)
            current_depth += i == (2**current_depth - 1)  # If last node at current depth is reached, increase current depth

        return max_nodes - 1 - map_b_to_d  # Inverse the mapping

    def initialize_node_library(self, operator_list: list, variable_list: list) -> None:
        """
        Initializes the node library.

        Parameters
        ----------
        operator_list : list
            List of operators that can be included in the trees.
        variable_list : list
            List of variables that can be included in the trees, which can vary between layers of a candidate.
        """
        string_to_node = {}  # Maps string to node
        node_to_string = {}
        node_function_list = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        if operator_list is None:
            operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.1), 
                             ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1),
                             ("*", lambda x, y: jnp.multiply(x, y), 2, 0.1),
                             ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
                             ("**", lambda x, y: jnp.power(x, y), 2, 0.1)
                             ]
            
        if variable_list is None:
            assert len(self.layer_sizes) == 1, "If more than one type of tree are optimized, you have to specify the input variables"
            variable_list = [["x" + str(i) for i in range(self.layer_sizes[0])]]

        n_operands = [0, 0] #Add 0 for empty node and constant node
        index = 2
        operator_probabilities = jnp.zeros(len(operator_list))

        assert len(operator_list) > 0, "No operators were given"

        for operator_tuple in operator_list:
            string = operator_tuple[0]
            f = operator_tuple[1] #Executable function
            arity = operator_tuple[2] #Number of children
            if len(operator_tuple) == 4:
                probability = operator_tuple[3] #Probability of function to be sampled
            else:
                probability = 0.1

            if string not in string_to_node:
                string_to_node[string] = index
                node_to_string[index] = string
                if arity == 1:
                    node_function_list.append(lambda_operator_arity1(f))
                    n_operands.append(1)
                elif arity == 2:
                    node_function_list.append(lambda_operator_arity2(f))
                    n_operands.append(2)
                operator_probabilities = operator_probabilities.at[index - 2].set(probability)
                index += 1

        self.operator_probabilities = operator_probabilities
        self.operator_indices = jnp.arange(2, index) #Store the indices corresponding to operator nodes
        var_start_index = index #The leaf nodes are appended to the operator list

        data_index = 0
        assert len(self.layer_sizes) == len(variable_list), "There is not a set of expressions for every type of layer"

        for var_list in variable_list:
            if len(var_list) == 0:
                continue     
            for var_or_tuple in var_list:
                if isinstance(var_or_tuple, str): #Variables may be provided with or without probability
                    var = var_or_tuple
                else:
                    var = var_or_tuple[0]
                if var not in string_to_node:
                    try:
                        sympy.parsing.sympy_parser.parse_expr(f"1.0*{var}"), "Variable is not a valid expression"
                        string_to_node[var] = index
                        node_to_string[index] = var
                        node_function_list.append(lambda_leaf(data_index))
                        n_operands.append(0) #Leaf nodes have no children
                        index += 1
                        data_index += 1
                    except:
                        raise ValueError(f"Variable {var} is not a valid expression. Please use a valid expression or remove it from the variable list.")
        
        self.variable_indices = jnp.concatenate([jnp.array([1]), jnp.arange(var_start_index, index)]) #Store the indices corresponding to leaf nodes
        variable_array = jnp.zeros((self.num_trees, data_index + 1))

        counter = 0
        for layer_i, var_list in enumerate(variable_list):
            p = jnp.zeros((data_index + 1))
            p = p.at[0].set(0.1)
            for var_or_tuple in var_list:
                if isinstance(var_or_tuple, str): #Variables may be provided with or without probability
                    var = var_or_tuple
                    var_p = 0.1
                else:
                    var = var_or_tuple[0]
                    var_p = var_or_tuple[1]
                p = p.at[string_to_node[var] - var_start_index + 1].set(var_p)

            for _ in range(self.layer_sizes[layer_i]):
                variable_array = variable_array.at[counter].set(p)
                counter += 1

        self.slots = jnp.array(n_operands)
        self.string_to_node = string_to_node
        self.node_to_string = node_to_string
        self.node_function_list = node_function_list
        self.variable_array = variable_array

        print(self.variable_indices, self.variable_array)

    def fit(self, key: PRNGKey, data: Tuple, verbose: int = 0, save_pareto_front: bool = False, path_to_file: str = None) -> None:
        """
        Fits the genetic programming algorithm to the data.

        Parameters
        ----------
        data : Tuple
            Data required for evaluation.
        key : PRNGKey
            Random key.
        verbose : int, optional
            Whether to print the best fitness and solution at each generation. By choosing a positive integer, you can control the frequency of the output.
        save_pareto_front : bool, optional
            Whether to save the Pareto front to a file.
        path_to_file : str, optional
            Name of the file to save the Pareto front.
        """

        # Check if path exists when saving is requested
        if save_pareto_front:
            assert path_to_file is not None, "A file name must be provided when saving the Pareto front"
            import os
            # Check if directory exists and create it if needed
            directory = os.path.dirname(path_to_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

        key, init_key = jr.split(key)

        self.reset()
        
        # Warm up JIT functions with actual data shapes
        dummy_population = self.initialize_population(init_key)
        self._warm_up_jit_functions(dummy_population, data)

        population = dummy_population

        for g in range(self.num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            fitness, population = self.evaluate_population(population, data, eval_key)

            if verbose:
                if g % verbose == 0:
                    print(f"In generation {g+1}")
                    self.print_pareto_front()

            if g < (self.num_generations-1):
                population = self.evolve_population(population, fitness, sample_key)

        print("Final pareto front:")
        self.print_pareto_front(save_pareto_front, path_to_file)
        # print(eqx.debug.get_num_traces(self.jit_eval))

    def reset(self) -> None:
        """Resets the state of the genetic programming algorithm."""
        self.current_generation = 0
        self.constant_step_size = self.constant_step_size_init

        # The Pareto front keeps track of the best solutions at each complexity level
        self.pareto_front = (jnp.array([self.max_fitness]), jnp.ones((1, self.num_trees, self.max_nodes, 4)))        

    def initialize_population(self, key: PRNGKey) -> Array:
        """
        Randomly initializes the population.

        Parameters
        ----------
        key : PRNGKey
            Random key.

        Returns
        -------
        Array
            Population.
        """
        keys = jr.split(key, self.num_populations)
        populations = jax.vmap(self.sample_population, in_axes=[0, None])(keys, self.population_size)

        return self.jit_simplify_constants(populations)

    def evolve_population(self, populations: Array, fitness: Array, key: PRNGKey) -> Array:
        """
        Evolves each population independently.

        Parameters
        ----------
        populations : Array
            Populations of candidates.
        fitness : Array
            Fitness of candidates.
        key : PRNGKey
            Random key.

        Returns
        -------
        Array
            Evolved populations.
        """

        populations, fitness = jax.vmap(self.punish_duplicates)(populations, fitness) #Give duplicate candidates poor fitness

        fitness = jnp.expand_dims(fitness, axis=2) #Expand fitness to match the shape of the populations
        
        if self.complexity_objective:
            complexities = jax.vmap(lambda population: jax.vmap(lambda candidate: jnp.sum(candidate[:,:,0]!=0))(population))(populations)
            complexities = jnp.maximum(complexities, 5*self.num_trees*jnp.ones_like(complexities))
            fitness = jnp.concatenate([fitness, complexities[:,:,None]], axis=-1)

        new_populations = evolve_populations(self.jit_evolve_population, 
                                             populations, 
                                             fitness,
                                             key, 
                                             self.current_generation, 
                                             self.migration_period, 
                                             self.migration_size, 
                                             self.reproduction_type_probabilities, 
                                             self.reproduction_probabilities)
        
        self.current_generation += 1
        if self.constant_optimization and self.current_generation >= self.start_constant_optimization:
            self.constant_step_size = jnp.maximum(self.constant_step_size * self.constant_step_size_decay, 0.001) #Update step size for constant optimization
        return self.jit_simplify_constants(new_populations)
    
    def mutate_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Mutates a pair of candidates.

        Parameters
        ----------
        parent1 : Array
            First parent candidate.
        parent2 : Array
            Second parent candidate.
        keys : Array
            Random keys.
        reproduction_probability : float
            Probability of a tree to be mutated.

        Returns
        -------
        Tuple[Array, Array]
            Pair of candidates after mutation.
        """
        offspring = jax.vmap(self.mutate_trees, in_axes=[0,1,None,None])(jnp.stack([parent1, parent2]), keys, reproduction_probability, self.variable_array)
        return offspring[0], offspring[1]

    def sample_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Samples a pair of candidates.

        Parameters
        ----------
        parent1 : Array
            First parent candidate.
        parent2 : Array
            Second parent candidate.
        keys : Array
            Random keys.
        reproduction_probability : float
            Probability of a tree to be sampled.

        Returns
        -------
        Tuple[Array, Array]
            Pair of candidates.
        """
        offspring = jax.vmap(lambda _keys: jax.vmap(self.sample_tree, in_axes=[0, None, 0])(_keys, self.max_init_depth, self.variable_array), in_axes=[1])(keys)
        return offspring[0], offspring[1]

    def simplify_constants_in_row(self, i: int, carry: Tuple[Array, Array, Array]) -> Tuple[Array, Array, Array]:
        """
        Simplifies the constants in a tree.

        Parameters
        ----------
        i : int
            Index of the node.
        carry : Tuple[Array, Array, Array]
            Tuple containing the tree, tree indices, and empty tree.

        Returns
        -------
        Tuple[Array, Array, Array]
            Simplified tree.
        """
        tree, tree_indices, empty_tree = carry

        last_node_idx = jnp.sum(tree[:,0]==0)
        f_idx, a_idx, b_idx, constant = tree[i]

        evaluated_subtree = tree.at[i].set(jnp.array([1.0, -1.0, -1.0, jax.lax.switch(f_idx.astype(int), self.node_function_list, tree[a_idx.astype(int), -1], tree[b_idx.astype(int), -1], jnp.zeros(1))]))
        
        one_branch_tree = jnp.where((tree_indices < i) & (tree_indices >= last_node_idx + 1), jnp.roll(tree, 1, axis=0), evaluated_subtree)
        one_branch_tree = jnp.where(tree_indices < last_node_idx + 1, empty_tree, one_branch_tree)
        one_branch_tree = one_branch_tree.at[:,1:3].set(jnp.where((one_branch_tree[:,1:3] < a_idx) & (one_branch_tree[:,1:3] > -1), one_branch_tree[:,1:3] + 1, one_branch_tree[:,1:3]))
        
        two_branch_tree = jnp.where((tree_indices < i) & (tree_indices >= last_node_idx + 2), jnp.roll(tree, 2, axis=0), evaluated_subtree)
        two_branch_tree = jnp.where(tree_indices < last_node_idx + 2, empty_tree, two_branch_tree)
        two_branch_tree = two_branch_tree.at[:,1:3].set(jnp.where((two_branch_tree[:,1:3] < b_idx) & (two_branch_tree[:,1:3] > -1), two_branch_tree[:,1:3] + 2, two_branch_tree[:,1:3]))
        
        new_tree = jax.lax.select((tree[a_idx.astype(int), 0] == 1) & (b_idx == -1), one_branch_tree, tree)
        new_tree = jax.lax.select((tree[a_idx.astype(int), 0] == 1) & (tree[b_idx.astype(int), 0] == 1), two_branch_tree, new_tree)

        new_tree = jax.lax.select(a_idx > -1, new_tree, tree)

        return (new_tree, tree_indices, empty_tree)

    def simplify_constants(self, tree: Array) -> Array:
        """
        Simplifies a tree by evaluating constant subtrees.

        Parameters
        ----------
        tree : Array
            Tree to be simplified.

        Returns
        -------
        Array
            Simplified tree.
        """
        tree, _, _ = jax.lax.fori_loop(0, self.max_nodes, self.simplify_constants_in_row, (tree, self.tree_indices, self.empty_tree))

        return tree

    def evaluate_row_from_tree(self, i: int, carry: Tuple[Array, Array], node_function_list: list) -> Tuple[Array, Array]:
        """
        Evaluates a node given inputs.

        Parameters
        ----------
        i : int
            Index of the node.
        carry : Tuple[Array, Array]
            Tuple containing the tree and data.
        node_function_list : list
            List of functions corresponding to nodes.

        Returns
        -------
        Tuple[Array, Array]
            Evaluated node.
        """

        tree, data = carry
        f_idx, a_idx, b_idx, constant = tree[i]  # Get node function, index of first and second operand, and constant value of node (which will be 0 if the node function is not 1)

        x = tree[a_idx.astype(int), 3]  # Value of first operand
        y = tree[b_idx.astype(int), 3]  # Value of second operand
        value = jax.lax.select(f_idx == 1, constant, jax.lax.switch(f_idx.astype(int), node_function_list, x, y, data))  # Computes value of the node

        tree = tree.at[i, 3].set(value)  # Store value

        return (tree, data)

    def iterate_through_tree(self, tree: Array, data: Array) -> Array:
        """
        Loops through a tree to compute the value of each node bottom up.

        Parameters
        ----------
        tree : Array
            Tree to be evaluated.
        data : Array
            Data for evaluation.

        Returns
        -------
        Array
            Value of the root node.
        """

        x, _ = jax.lax.fori_loop(0, self.max_nodes, self.jit_evaluate_row_from_tree, (tree, data)) #Iterate over the tree to compute the value of each node
        return x[-1, -1] #Return the value of the root node

    def tree_evaluator(self, candidate: Array, data: Array) -> Array:
        """
        Calls the evaluation function for each tree in a candidate.

        Parameters
        ----------
        candidate : Array
            Candidate to be evaluated.
        data : Array
            Data for evaluation.

        Returns
        -------
        Array
            Result of each tree.
        """

        # data = jnp.atleast_1d(data)
        result = jax.vmap(self.iterate_through_tree, in_axes=[0, None])(candidate, data) #Evaluate each tree in the candidate
        # return jnp.squeeze(result)
        return result

    def evaluate_population(self, populations: Array, data: Tuple, key: PRNGKey) -> Tuple[Array, Array]:
        """
        Evaluates every candidate in population and assigns a fitness. Optionally, the constants in the candidates are optimized.

        Parameters
        ----------
        populations : Array
            Population of candidates.
        data : Tuple
            The data required to evaluate the population.
        key : PRNGKey
            Random key.

        Returns
        -------
        Tuple[Array, Array]
            Fitness and evaluated or optimized population.
        """

        # Flatten the populations so they can be distributed over the devices
        flat_populations = populations.reshape(self.num_populations * self.population_size, *populations.shape[2:])
        data = jax.device_put(data, self.data_mesh)

        fitness = self.jit_eval(flat_populations, data)  # Evaluate the candidates

        # Optimize constants of the best candidates in the current generation
        if self.constant_optimization and self.current_generation >= self.start_constant_optimization:
            self.optimizer = self.optimizer_class(self.constant_step_size)

            # Get best candidates
            best_candidates_idx = jnp.argsort(fitness)[:self.optimize_constants_elite]
            best_candidates = flat_populations[best_candidates_idx]

            # Optimize constants of the best candidates
            optimized_fitness, optimized_population = self.jit_optimize(best_candidates, data, jr.split(key, self.optimize_constants_elite), self.constant_step_size)

            optimized_population = jax.block_until_ready(optimized_population)

            # Store updated candidates and fitness
            flat_populations = flat_populations.at[best_candidates_idx].set(optimized_population)
            fitness = fitness.at[best_candidates_idx].set(optimized_fitness)

        self.update_pareto_front(fitness, flat_populations)

        # Reshape the populations into the subpopulations
        fitness = fitness.reshape((self.num_populations, self.population_size))
        populations = flat_populations.reshape((self.num_populations, self.population_size, *flat_populations.shape[1:]))

        return fitness, populations
    
    def optimize_constants_epoch(self, carry: Tuple[Array, Array, Tuple], x: int) -> Tuple[Tuple[Array, Array, Tuple], Tuple[Array, Array]]:
        """
        Applies one step of constant optimization to a batch of candidates.

        Parameters
        ----------
        carry : Tuple[Array, Array, Tuple]
            Tuple containing candidates, states, and data.
        x : int
            Unused parameter for scan.

        Returns
        -------
        Tuple[Tuple[Array, Array, Tuple], Tuple[Array, Array]]
            Tuple containing updated candidates, states, and data, and the original candidates and loss.
        """
        candidates, states, data = carry
        loss, gradients = self.vmap_gradients(candidates[..., 3:], candidates[..., :3], data)  # Parallel computation of the loss and gradients

        # Regularize invalid solutions
        nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(loss)
        loss = jnp.where(nan_or_inf, jnp.ones(loss.shape) * self.max_fitness, loss)

        loss = jnp.minimum(loss, self.max_fitness * jnp.ones_like(loss))

        updates, states = jax.vmap(self.optimizer.update)(gradients, states, candidates[..., 3])  # Parallel computation of the updates
        new_candidates = candidates.at[..., 3:].set(jax.vmap(lambda t, u: t + u)(candidates[..., 3:], updates))  # Parallel updating of constants

        return (new_candidates, states, data), (candidates, loss)

    def optimize_constants_with_gradients(self, candidates: Array, data: Tuple, key: PRNGKey, step_size: float, n_epoch: int) -> Tuple[Array, Array]:
        """
        optimizes the constants in the candidates.

        Parameters
        ----------
        candidates : Array
            Candidate solutions.
        data : Tuple
            The data required to evaluate the population.
        key : PRNGKey
            Random key.
        n_epoch : int
            Number of steps to optimize constants.

        Returns
        -------
        Tuple[Array, Array]
            Optimized and evaluated candidate.
        """

        states = jax.vmap(self.optimizer.init)(candidates[..., 3:])  # Initialize optimizers for each candidate

        _, out = jax.lax.scan(self.optimize_constants_epoch, (candidates, states, data), length=n_epoch)

        new_candidates, loss = out

        fitness = jnp.min(loss, axis=0)  # Get best fitness during constant optimization
        candidates = jax.vmap(lambda t, i: t[i], in_axes=[1, 0])(new_candidates, jnp.argmin(loss, axis=0))  # Get best candidate during constant optimization

        return fitness, candidates

    def optimize_constants_generation(self, carry: Tuple[Array, Tuple, PRNGKey], x: int) -> Tuple[Tuple[Array, Tuple, PRNGKey], float]:
        """
        optimizes a generation of candidates.

        Parameters
        ----------
        carry : Tuple[Array, Tuple, PRNGKey]
            Tuple containing candidate, data, and key.
        x : int
            Unused parameter for scan.

        Returns
        -------
        Tuple[Tuple[Array, Tuple, PRNGKey], float]
            Tuple containing updated candidate, data, and key, and the best fitness.
        """

        candidate, data, key, step_size = carry

        key, sample_key = jr.split(key)

        mask = candidate[..., 0] == 1.0 #Only samples mutations for the nodes that contain a constant
        mutations = jax.vmap(lambda _key: step_size * jr.normal(_key, shape=(self.num_trees, self.max_nodes,)) * mask)(jr.split(sample_key, self.n_offspring))
        mutations = jnp.vstack([jnp.zeros((1, self.num_trees, self.max_nodes)), mutations])

        offspring = jax.vmap(lambda m: candidate.at[..., 3].set(candidate[..., 3] + m))(mutations)

        fitness = self.vmap_trees(offspring[..., 3:], offspring[..., :3], data)

        # Regularize invalid solutions
        nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(self.n_offspring + 1) * self.max_fitness, fitness)

        fitness = jnp.minimum(fitness, self.max_fitness * jnp.ones_like(fitness))

        return (offspring[jnp.argmin(fitness)], data, key, step_size), jnp.min(fitness)

    def optimize_constants_with_evolution(self, candidate: Array, data: Tuple, key: PRNGKey, step_size: float, n_iterations: int) -> Tuple[float, Array]:
        """
        optimizes a candidate using Evolution Strategies (ES).

        Parameters
        ----------
        candidate : Array
            Candidate solution.
        data : Tuple
            The data required to evaluate the population.
        key : PRNGKey
            Random key.
        n_iterations : int
            Number of iterations for optimization.

        Returns
        -------
        Tuple[float, Array]
            Best fitness and optimized candidate.
        """

        (new_candidate, _, _, _), fitness = jax.lax.scan(self.optimize_constants_generation, (candidate, data, key, step_size), length=n_iterations)

        return jnp.min(fitness), new_candidate

    def punish_duplicates(self, population: Array, fitness: Array) -> Tuple[Array, Array]:
        """
        Punishes duplicate candidates by setting their fitness to the maximum fitness.

        Parameters
        ----------
        population : Array
            Population of candidates.
        fitness : Array
            Fitness of candidates.

        Returns
        -------
        Tuple[Array, Array]
            Population and fitness with duplicates punished.
        """

        _, indices, counts = jnp.unique(population, return_index=True, return_counts=True, axis=0, size=self.population_size)
        population = population[indices]
        fitness = fitness[indices]
        return population, jnp.where(counts > 0, fitness, self.max_fitness)

    def update_pareto_front(self, fitness: Array, population: Array) -> None:
        """
        Updates the Pareto front with the current population.

        Parameters
        ----------
        current_fitness : Array
            Fitness of the current population.
        current_population : Array
            Current population.
        """
        current_pareto_fitness, current_pareto_solutions = self.pareto_front

        _fitness = jnp.concatenate([fitness, current_pareto_fitness], axis=0)
        _population = jnp.concatenate([population, current_pareto_solutions], axis=0)

        # Compute complexity of the current population
        complexity = jax.vmap(lambda array: jnp.sum(array[:, :, 0] != 0))(_population)[:,None]
        metrics = jnp.concatenate([_fitness[:,None], complexity], axis=-1)

        # For each solution i, check if it's dominated by any other solution j
        
        # Reshape metrics for broadcasting. Shape: (n_solutions, 1, n_objectives)
        metrics_i = jnp.expand_dims(metrics, axis=1)
        
        # Shape: (1, n_solutions, n_objectives)
        metrics_j = jnp.expand_dims(metrics, axis=0)
        
        # Check if j dominates i: 
        # 1. j is better or equal in all objectives
        # 2. j is strictly better in at least one objective
        j_better_or_equal = (metrics_j <= metrics_i)
        j_strictly_better = (metrics_j < metrics_i)
        
        # For each pair (i,j), check if j dominates i
        j_dominates_i = jnp.all(j_better_or_equal, axis=2) & jnp.any(j_strictly_better, axis=2)
        
        # Don't compare solutions with themselves
        mask = ~jnp.eye(metrics.shape[0], dtype=bool)
        j_dominates_i = j_dominates_i & mask
        
        # A solution is non-dominated if it's not dominated by any other solution
        dominated_by_others = ~jnp.any(j_dominates_i, axis=1)
        
        pareto_indices = jnp.nonzero(dominated_by_others)[0]

        pareto_solutions, unique_indices = jnp.unique(_population[pareto_indices], return_index=True, axis=0)

        self.pareto_front = (_fitness[pareto_indices][unique_indices], pareto_solutions)

    def print_pareto_front(self, save: bool = False, path_to_file: str = None) -> None:
        """
        Prints the Pareto front.

        Parameters
        ----------
        save : bool, optional
            Whether to save the Pareto front to a file.
        path_to_file : str, optional
            Name of the file to save the Pareto front.
        """

        # Check if path exists when saving is requested
        if save:
            assert path_to_file is not None, "A file name must be provided when saving the Pareto front"
            import os
            # Check if directory exists and create it if needed
            directory = os.path.dirname(path_to_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

        pareto_fitness, pareto_solutions = self.pareto_front

        complexities = jax.vmap(lambda array: jnp.sum(array[:, :, 0] != 0))(pareto_solutions)

        sorted_indices = jnp.argsort(complexities)
        pareto_fitness = pareto_fitness[sorted_indices]
        pareto_solutions = pareto_solutions[sorted_indices]
        complexities = complexities[sorted_indices]

        if save:
            pareto_table = ['Complexity', 'Fitness']
            for i in range(self.num_trees):
                pareto_table.append(f'Equation {i}')
            pareto_table = [tuple(pareto_table)]

        expression_strings = []

        # Group solutions by complexity and get the best fitness for each
        unique_complexities = jnp.unique(complexities)
        printed_complexities = set()  # Track which complexity levels we've printed

        for c in range(complexities.shape[0]):
            string_equations = self.expression_to_string(pareto_solutions[c])
            complexity = complexities[c].item()
            
            # Always save all solutions
            if save:
                if self.num_trees > 1:
                    temp = (complexity, pareto_fitness[c])
                    for tree in string_equations:
                        temp += (tree,)
                        pareto_table.append(temp)
                else:
                    pareto_table.append((complexity, pareto_fitness[c], string_equations))
            
            # Only print if we haven't printed this complexity level yet
            if complexity not in printed_complexities and string_equations not in expression_strings:
                printed_complexities.add(complexity)
                expression_strings.append(string_equations)
                
                if self.num_trees > 1:
                    print(f"Complexity: {complexity}, fitness: {pareto_fitness[c]}, equations: {string_equations}")
                else:
                    print(f"Complexity: {complexity}, fitness: {pareto_fitness[c]}, equation: {string_equations}")

        if save:
            np.savetxt(f'{path_to_file}/pareto_front.csv', pareto_table, delimiter=',', fmt='%s')
    
    def tree_to_string(self, tree: Array) -> str:
        """
        Maps tree to string.

        Parameters
        ----------
        tree : Array
            Tree to be converted to string.

        Returns
        -------
        str
            String representation of tree.
        """

        if tree[-1, 0] == 1:  # constant
            return str(tree[-1, 3])
        elif tree[-1, 1] < 0:  # Variable
            return self.node_to_string[tree[-1, 0].astype(int).item()]
        elif tree[-1, 2] < 0:  # Operator with one operand
            substring = self.tree_to_string(tree[:tree[-1, 1].astype(int) + 1])
            operator_string = self.node_to_string[tree[-1, 0].astype(int).item()]

            if operator_string[0].isalpha() or operator_string[0].isdigit():
                return f"{operator_string}({substring})"
            else:
                return f"({substring}){operator_string}"
            
        else:  # Operator with two operands
            substring1 = self.tree_to_string(tree[:tree[-1, 1].astype(int) + 1])
            substring2 = self.tree_to_string(tree[:tree[-1, 2].astype(int) + 1])
            operator_string = self.node_to_string[tree[-1, 0].astype(int).item()]

            if operator_string in ["+", "-", "*", "/", "**"]:
                return f"({substring1}){self.node_to_string[tree[-1, 0].astype(int).item()]}({substring2})"
            else:
                return f"{operator_string}({substring1}, {substring2})"

    def expression_to_string(self, candidate: Array) -> str:
        """
        Maps trees in a candidate to string.

        Parameters
        ----------
        candidate : Array
            Candidate to be converted to string.

        Returns
        -------
        str
            String representation of candidate.
        """
        if self.num_trees == 1:
            simplified_expression = sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(candidate[0]))

            rounded_expression = simplified_expression

            for a in sympy.preorder_traversal(simplified_expression):
                if isinstance(a, sympy.Float):
                    rounded_expression = rounded_expression.subs(a, sympy.Float(a, 3))

            return rounded_expression

        string_output = []
        for tree in candidate:
            simplified_expression = sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(tree))

            rounded_expression = simplified_expression

            for a in sympy.preorder_traversal(simplified_expression):
                if isinstance(a, sympy.Float):
                    rounded_expression = rounded_expression.subs(a, sympy.Float(a, 3))

            string_output.append(rounded_expression)

        return string_output