"""
Kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
import equinox as eqx

from typing import Tuple, Callable
import time

from Kozax.genetic_operators.crossover import crossover_trees
from Kozax.genetic_operators.initialization import sample_population, sample_tree
from Kozax.genetic_operators.mutation import initialize_mutation_functions
from Kozax.genetic_operators.reproduction import evolve_populations, evolve_population

#Function containers
def lambda_operator_arity1(f):
    return lambda x, y, _data: f(x)

def lambda_operator_arity2(f):
    return lambda x, y, _data: f(x, y)

def lambda_leaf(i):
    return lambda x, y, _data: _data[i]

class GeneticProgramming:
    """Genetic programming strategy of symbolic expressions.

        :param num_generations: The number of generations over which to evolve the population
        :param population_size: Number of candidates in the population
        :param fitness_function: Function that evaluates a candidate and assigns a fitness
        :param operator_list: List of operators that can be included in the trees
        :param variable_list: List of variables that can be included in the trees, which can vary between layers of a candidate
        :param layer_sizes: Size of each layer in a candidate
        :param num_populations: Number of subpopulations
        :param max_init_depth: Highest depth of a tree during initialization
        :param max_nodes: Maximum number of nodes in a tree
        :param device_type: Type of device on which the evaluation and evolution takes place
        :param tournament_size: Size of the tournament
        :param size_parsimony: Parsimony factor that increases the fitness of a candidate based on its size
        :param coefficient_sd: Standard deviation to sample coefficients
        :param migration_period: Number of generations after which populations are migrated
        :param migration_percentage: Number of candidates to migrate
        :param elite_percentage: Percentage of elite candidates that procceed to the next population
        :param coefficient_optimisation: If the coefficients are optimised with gradients
        :param gradient_steps: For how many steps the coefficients are optimised
        :param start_coefficient_optimisation: After which generation the optimisation of coefficients should start
        :param optimise_coefficients_elite: The number of the best candidates of which the coefficients are optimised
        :param optimiser: Optimiser for coefficient optimisation
        :param selection_pressure_factors: The selection pressure for each subpopulation
        :param reproduction_probability_factors: The reproduction probability for each subpopulation
        :param crossover_probability_factors: The crossover probability for each subpopulation
        :param mutation_probability_factors: The mutation probability for each subpopulation
        :param sample_probability_factor: The probability to sample a new candidate for each subpopulation
        """
    def __init__(self, num_generations: int, 
                 population_size: int, 
                 fitness_function: Callable, 
                 operator_list: list = None,
                 variable_list: list = None,
                 layer_sizes: Array = jnp.array([1]),
                 num_populations: int = 1,
                 max_init_depth: int = 4,
                 max_nodes: int = 30,
                 device_type: str = 'cpu',
                 tournament_size: int = 7, 
                 size_parsimony: float = 0.0, 
                 coefficient_sd: float = 1.0,
                 migration_period: int = 10,
                 migration_percentage: float = 0.1,
                 elite_size: int = 10,
                 coefficient_optimisation: str = None,
                 gradient_steps: int = 10,
                 ES_n_offspring: int = 50,
                 ES_n_iterations: int = 5,
                 start_coefficient_optimisation: int = 50,
                 optimise_coefficients_elite: int = 100,
                 optimiser_class = optax.adam,
                 init_learning_rate: float = 0.1,
                 learning_rate_decay: float = 0.99,
                 max_fitness: float = 1e8,
                 selection_pressure_factors: Tuple[float] = (0.9, 0.9),
                 reproduction_probability_factors: Tuple[float] = (0.75, 0.75),
                 crossover_probability_factors: Tuple[float] = (0.9, 0.1),
                 mutation_probability_factors: Tuple[float] = (0.1, 0.9),
                 sample_probability_factors: Tuple[float] = (0.0, 0.0)) -> None:
        
        self.layer_sizes = layer_sizes
        assert num_populations>0, "The number of populations should be larger than 0"
        self.num_populations = num_populations
        assert population_size>0 and population_size%2==0, "The population_size should be larger than 0 and an even number"
        self.population_size = population_size
        assert max_init_depth>0, "The max initial depth should be larger than 0"
        self.max_init_depth = max_init_depth
        assert max_nodes>0, "The max number of nodes should be larger than 0"
        self.max_nodes = max_nodes
        self.num_trees = jnp.sum(self.layer_sizes)
        assert self.num_trees>0, "The number of trees should be larger than 0"

        self.current_generation = 0
        assert num_generations>0, "The number of generations should be larger than 0"
        self.num_generations = num_generations
        self.best_fitnesses = jnp.zeros(self.num_generations)
        self.best_solutions = jnp.zeros((self.num_generations, self.num_trees, self.max_nodes, 4))

        self.size_parsimony = size_parsimony
        self.coefficient_sd = coefficient_sd

        assert migration_period>1, "The migration period should be larger than 1"
        self.migration_period = migration_period
        assert migration_percentage*population_size%1==0, "The migration size should be an integer"
        self.migration_size = int(migration_percentage*population_size)

        assert tournament_size>1, "The number of gradient steps should be larger than 1"
        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(*selection_pressure_factors, self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(*crossover_probability_factors, self.num_populations),
                                                           jnp.linspace(*mutation_probability_factors, self.num_populations),
                                                           jnp.linspace(*sample_probability_factors, self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(*reproduction_probability_factors, self.num_populations)

        self.elite_size = elite_size
        assert self.elite_size%2==0, "The elite size should be a multiple of two"

        self.coefficient_optimisation = coefficient_optimisation
        if coefficient_optimisation:
            assert gradient_steps>0, "The number of gradient steps should be larger than 0"
        self.gradient_steps = gradient_steps
        self.optimiser_class = optimiser_class
        self.learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.max_fitness = max_fitness

        self.max_complexity = self.num_trees * self.max_nodes
        self.complexities = jnp.arange(self.max_complexity)
        self.pareto_front = (jnp.ones(self.max_complexity) * self.max_fitness, jnp.zeros((self.max_complexity, self.num_trees, self.max_nodes, 4)))

        self.map_b_to_d = self.create_map_b_to_d(jnp.maximum(self.max_init_depth, 3))

        self.optimise_coefficients_elite = optimise_coefficients_elite
        self.start_coefficient_optimisation = start_coefficient_optimisation

        #Initialize library of nodes
        string_to_node = {} #Maps string to node
        node_to_string = {}
        node_function_list = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        if operator_list is None:
            operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                 ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1),
                 ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                 ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
                 ("**", lambda x, y: jnp.power(x, y), 2, 0.1)
                 ]
        if variable_list is None:
            assert len(self.layer_sizes) == 1, "If you want to optimise more than one type of tree, you have to specify the input variables"
            variable_list = [["x" + str(i) for i in range(self.layer_sizes[0])] for _ in range(self.layer_sizes[0])]

        n_operands = [0, 0]
        index = 2
        operator_probabilities = jnp.zeros(len(operator_list))

        assert len(operator_list)>0, "No operators were given"

        for operator_tuple in operator_list:
            string = operator_tuple[0]
            f = operator_tuple[1]
            arity = operator_tuple[2]
            if len(operator_tuple)==4:
                probability = operator_tuple[3]
            else:
                probability = 0.1

            if string not in string_to_node:
                string_to_node[string] = index
                node_to_string[index] = string
                if arity==1:
                    node_function_list.append(lambda_operator_arity1(f))
                    n_operands.append(1)
                elif arity==2:
                    node_function_list.append(lambda_operator_arity2(f))
                    n_operands.append(2)
                operator_probabilities = operator_probabilities.at[index-2].set(probability)
                index += 1

        self.operator_probabilities = operator_probabilities
        self.operator_indices = jnp.arange(2, index)
        var_start_index = index

        data_index = 0
        assert len(layer_sizes) == len(variable_list), "There is not a set of expressions for every type of layer"

        for var_list in variable_list:
            assert len(var_list)>0, "An empty set of variables was given"
            for var_or_tuple in var_list:
                if isinstance(var_or_tuple, str):
                    var = var_or_tuple
                else:
                    var = var_or_tuple[0]
                if var not in string_to_node:
                    string_to_node[var] = index
                    node_to_string[index] = var
                    node_function_list.append(lambda_leaf(data_index))
                    n_operands.append(0)
                    index += 1
                    data_index += 1
        
        self.variable_indices = jnp.arange(var_start_index, index)
        variable_array = jnp.zeros((self.num_trees, data_index))

        counter = 0
        for layer_i, var_list in enumerate(variable_list):
            p = jnp.zeros((data_index))
            for var_or_tuple in var_list:
                if isinstance(var_or_tuple, str):
                    var = var_or_tuple
                    var_p = 1.0
                else:
                    var = var_or_tuple[0]
                    var_p = var_or_tuple[1]
                p = p.at[string_to_node[var] - var_start_index].set(var_p)

            for _ in range(layer_sizes[layer_i]):
                variable_array = variable_array.at[counter].set(p)
                counter += 1

        self.slots = jnp.array(n_operands)
        self.string_to_node = string_to_node
        self.node_to_string = node_to_string
        self.node_function_list = node_function_list
        self.variable_array = variable_array
        self.tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        self.empty_tree = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))

        print(f"Input data should be formatted as: {[self.node_to_string[i.item()] for i in self.variable_indices]}.")

        #Define jittable reproduction functions
        self.sample_args = (self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.coefficient_sd, 
                            self.map_b_to_d)
                
        self.sample_tree = partial(sample_tree,
                                   max_nodes = self.max_nodes, 
                                   tree_size = 2**jnp.maximum(self.max_init_depth, 3)-1,
                                   args = self.sample_args)
        
        self.sample_population = partial(sample_population, 
                                         num_trees = self.num_trees, 
                                         max_init_depth = self.max_init_depth, 
                                         variable_array = self.variable_array,
                                         sample_function = self.sample_tree)

        self.mutate_args = (self.sample_tree, 
                            self.max_nodes, 
                            self.max_init_depth, 
                            self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.coefficient_sd)
        
        self.mutate_trees = initialize_mutation_functions(self.mutate_args)

        self.partial_crossover = partial(crossover_trees, 
                                         operator_indices = self.operator_indices, 
                                         max_nodes = self.max_nodes)

        self.reproduction_functions = [self.partial_crossover, self.mutate_pair, self.sample_pair]

        self.jit_evolve_population = jax.jit(eqx.debug.assert_max_traces(jax.vmap(partial(evolve_population, 
                                                     reproduction_functions = self.reproduction_functions, 
                                                     elite_size = self.elite_size, 
                                                     tournament_size = self.tournament_size, 
                                                     num_trees = self.num_trees, 
                                                     population_size=population_size),
                                                    in_axes=[0, 0, 0, 0, 0, 0, None]), max_traces=1))
        
        self.jit_simplification = jax.jit(eqx.debug.assert_max_traces(jax.vmap(jax.vmap(jax.vmap(self.simplify_tree))), max_traces=1))

        #Define partial fitness function for evaluation
        self.jit_body_fun = partial(self.body_fun, node_function_list = self.node_function_list)
        self.partial_ff = lambda coefficients, nodes, data: fitness_function(jnp.concatenate([nodes, coefficients], axis=-1), data, self.vmap_foriloop)

        #Define parallel evaluation functions
        self.vmap_trees = jax.vmap(self.partial_ff, in_axes=[0, 0, None])
        self.vmap_gradients = jax.vmap(jax.value_and_grad(self.partial_ff), in_axes=[0, 0, None])
        self.gradients_f = jax.value_and_grad(self.partial_ff)

        devices = mesh_utils.create_device_mesh((len(jax.devices(device_type))))
        self.mesh = Mesh(devices, axis_names=('i'))
        self.data_mesh = NamedSharding(self.mesh, P())

        assert coefficient_optimisation in ["ES", "BP", None], "This optimisation method is not implemented"
        if coefficient_optimisation == "ES":
            self.optimise_coefficients_function = jax.vmap(partial(self.optimise_ES, n_iterations = ES_n_iterations),  in_axes=[0,None,0])
            self.n_offspring = ES_n_offspring
            self.coefficient_optimisation = True
        elif coefficient_optimisation == "BP":
            self.optimise_coefficients_function = partial(self.optimise_gradient, n_epoch = gradient_steps)
            self.coefficient_optimisation = True
        else:
            self.optimise_coefficients_function = None
            self.coefficient_optimisation = False

        #Define sharded functions for evaluation and optimisation
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=P('i'), check_rep=False)
        def shard_eval(array, data):
            fitness = self.vmap_trees(array[...,3:], array[...,:3], data)

            nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
            fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
            
            return jnp.clip(fitness,0,self.max_fitness)
            
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None), P('i')), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_optimise(array, data, keys):
            # result, _array = self.optimise(array, data, keys, self.gradient_steps)
            result, _array = self.optimise_coefficients_function(array, data, keys)
            return result, _array
        
        self.jit_eval = jax.jit(eqx.debug.assert_max_traces(shard_eval, max_traces=1))
        self.jit_optimise = jax.jit(eqx.debug.assert_max_traces(shard_optimise, max_traces=1))

    def create_map_b_to_d(self, depth: int) -> Array:
        """
        Creates a mapping from the breadth first index to depth first index given a depth

        :param depth

        Returns: Index mapping
        """
                
        max_nodes = 2**depth-1
        current_depth = 0
        map_b_to_d = jnp.zeros(max_nodes)
        
        for i in range(max_nodes):
            if i>0:
                parent = (i + (i%2) - 2)//2 #Determine parent position
                value = map_b_to_d[parent]
                if (i % 2)==0: #Right child
                    new_value = value + 2**(depth-current_depth+1) 
                else: #Left child
                    new_value = value + 1
                map_b_to_d = map_b_to_d.at[i].set(new_value)
            current_depth += i==(2**current_depth-1) #If last node at current depth is reached, increase current depth

        return max_nodes - 1 - map_b_to_d #Inverse the mapping

    def initialize_population(self, key: PRNGKey) -> Array:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jr.split(key, self.num_populations)
        populations = jax.vmap(self.sample_population, in_axes=[0, None])(keys, self.population_size)

        return self.jit_simplification(populations)
    
    def tree_to_string(self, tree: Array) -> str:
        """
        Maps tree to string

        :param tree

        Returns: String representation of tree
        """
        if tree[-1,0]==1: #Coefficient
            return str(tree[-1,3])
        elif tree[-1,1]<0: #Variable
            return self.node_to_string[tree[-1,0].astype(int).item()]
        elif tree[-1,2]<0: #Operator with one operand
            substring = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            operator_string = self.node_to_string[tree[-1,0].astype(int).item()]

            if operator_string[0].isalpha() or operator_string[0].isdigit():
                return f"{operator_string}({substring})"
            else:
                return f"({substring}){operator_string}"
        else: #Operator with two operands
            substring1 = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            substring2 = self.tree_to_string(tree[:tree[-1,2].astype(int)+1])
            operator_string = self.node_to_string[tree[-1,0].astype(int).item()]
            if operator_string in ["+", "-", "*", "/", "**"]:
                return f"({substring1}){self.node_to_string[tree[-1,0].astype(int).item()]}({substring2})"
            else:
                return f"{operator_string}({substring1}, {substring2})"
        
    def to_string(self, candidate: Array) -> str:
        """
        Maps trees in a candidate to string
        
        :param candidate

        Returns: String representation of candidate
        """
        string_output = ""
        tree_index = 0
        layer_index = 0

        if jnp.sum(self.layer_sizes) == 1:
            simplified_expression = sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(candidate[0]))

            rounded_expression = simplified_expression

            for a in sympy.preorder_traversal(simplified_expression):
                if isinstance(a, sympy.Float):
                    rounded_expression = rounded_expression.subs(a, sympy.Float(a, 3))

            return rounded_expression

        for tree in candidate:
            if tree_index==0: #Begin layer of trees
                string_output += "["
            simplified_expression = sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(tree))

            rounded_expression = simplified_expression

            for a in sympy.preorder_traversal(simplified_expression):
                if isinstance(a, sympy.Float):
                    rounded_expression = rounded_expression.subs(a, sympy.Float(a, 3))

            string_output += str(rounded_expression) #Map tree to string
            if tree_index < (self.layer_sizes[layer_index] - 1): #Continue layer of trees
                string_output += ", "
                tree_index += 1
            else: #End layer of trees
                string_output += "]"
                if layer_index < (self.layer_sizes.shape[0] - 1): #Begin new layer
                    string_output += ", "
                tree_index = 0
                layer_index += 1
        return string_output
    
    def body_fun(self, i, carry: Tuple[Array, Array], node_function_list):
        """
        Evaluates a node given inputs
        
        :param tree
        :param data
        :param node_function_list: Maps nodes to callable functions

        Returns: Evaluated node
        """

        tree, data = carry
        f_idx, a_idx, b_idx, coefficient = tree[i] #Get node function, index of first and second operand, and coefficient value of node (which will be 0 if the node function is not 1)
    
        x = tree[a_idx.astype(int), 3] #Value of first operand
        y = tree[b_idx.astype(int), 3] #Value of second operand
        value = jax.lax.select(f_idx == 1, coefficient, jax.lax.switch(f_idx.astype(int), node_function_list, x, y, data)) #Computes value of the node
        
        tree = tree.at[i, 3].set(value) #Store value

        return (tree, data)

    def foriloop(self, tree: Array, data: Array) -> Array:
        """
        Loops through a tree to compute the value of each node bottom up 
        
        :param tree
        :param data

        Returns: Value of the root node
        """
        x, _ = jax.lax.fori_loop(0, self.max_nodes, self.jit_body_fun, (tree, data))
        return x[-1, -1]

    def vmap_foriloop(self, candidate: Array, data: Array) -> Array:
        """
        Calls the evaluation function for each tree in a candidate

        :param candidate
        :param data

        Returns: Result of each tree
        """

        result = jax.vmap(self.foriloop, in_axes=[0, None])(candidate, data)
        return result
       
    def evaluate_population(self, populations: Array, data: Tuple, key: PRNGKey) -> Tuple[Array, Array]:
        """Evaluates every candidate in population and assigns a fitness. Optionally the coefficients in the candidates are optimised

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated or optimised population.
        """

        flat_populations = populations.reshape(self.num_populations*self.population_size, *populations.shape[2:]) #Flatten the populations so they can be distributed over the devices
        # populations = jax.device_put(populations, NamedSharding(self.mesh, P('i')))
        data = jax.device_put(data, self.data_mesh)
        
        # fitness = self.jit_eval(populations, data) #Evaluate the candidates
        # print("eval")
        fitness = self.jit_eval(flat_populations, data) #Evaluate the candidates

        #Optimise coefficients of the best candidates given conditions
        if (self.coefficient_optimisation & (self.current_generation>=self.start_coefficient_optimisation)):
            self.optimiser = self.optimiser_class(self.learning_rate)
            best_candidates_idx = jnp.argsort(fitness)[:self.optimise_coefficients_elite]
            # best_candidates_idx = jr.choice(jr.PRNGKey(self.current_generation), jnp.arange(0, flat_populations.shape[0]), shape=(self.optimise_coefficients_elite,), p=1/fitness)
            best_candidates = flat_populations[best_candidates_idx]
            # print("optimise ", jnp.mean(fitness))
            optimised_fitness, optimised_population = self.jit_optimise(best_candidates, data, jr.split(key, self.optimise_coefficients_elite))
            flat_populations = flat_populations.at[best_candidates_idx].set(optimised_population)
            fitness = fitness.at[best_candidates_idx].set(optimised_fitness)
            # print("done ", jnp.mean(optimised_fitness))

        # fitness, flat_populations = self.jit_optimise(flat_populations, data, jr.split(key, self.num_populations*self.population_size))
            
        self.update_pareto_front(fitness, flat_populations)

        fitness = fitness + jax.vmap(lambda array: self.size_parsimony * jnp.sum(array[:,:,0]!=0))(flat_populations) #Increase fitness based on the size of the candidate

        best_solution = flat_populations[jnp.argmin(fitness)]
        best_fitness = jnp.min(fitness)

        #Store best fitness and solution
        self.best_solutions = self.best_solutions.at[self.current_generation].set(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        fitness = fitness.reshape((self.num_populations, self.population_size))
        populations = flat_populations.reshape((self.num_populations, self.population_size, *flat_populations.shape[1:]))

        # print(jnp.min(fitness, axis=1))


        return fitness, populations
    
    def update_pareto_front(self, current_fitness, current_population):
        current_population_complexity = jax.vmap(lambda array: jnp.sum(array[:,:,0]!=0))(current_population)
        pareto_fitness, pareto_solutions = self.pareto_front
        new_pareto_front = jax.vmap(self.find_best_solution_given_complexity_level, in_axes=[0, None, None, None, 0, 0])(self.complexities, 
                                                                                                             current_fitness, 
                                                                                                             current_population, 
                                                                                                             current_population_complexity,
                                                                                                             pareto_fitness,
                                                                                                             pareto_solutions)

        self.pareto_front = new_pareto_front

    def find_best_solution_given_complexity_level(self, complexity, current_fitness, current_population, current_population_complexity, best_fitness, best_solution):
        fitness_at_complexity_level = jnp.where(current_population_complexity == complexity, current_fitness, jnp.ones_like(current_fitness) * self.max_fitness)
        best_fitness_at_complexity_level = jnp.min(fitness_at_complexity_level)
        best_solution_at_complexity_level = current_population[jnp.argmin(fitness_at_complexity_level)]

        new_best_fitness = jax.lax.select(best_fitness_at_complexity_level > best_fitness, best_fitness, best_fitness_at_complexity_level)
        new_best_solution = jax.lax.select(best_fitness_at_complexity_level > best_fitness, best_solution, best_solution_at_complexity_level)

        return new_best_fitness, new_best_solution
    
    def print_pareto_front(self):
        pareto_fitness, pareto_solutions = self.pareto_front
        best_pareto_fitness = jnp.inf

        for c in range(self.complexities):
            if pareto_fitness[c] < best_pareto_fitness: 
                print(f"Complexity: {c}, fitness: {pareto_fitness[c]}, equations: {self.to_string(pareto_solutions[c])}")
                best_pareto_fitness = pareto_fitness[c]
            
    def optimise_epoch(self, carry: Tuple[Array, Array, Tuple], x: int) -> Tuple[Tuple[Array, Array, Tuple], Tuple[Array, Array]]:
        """
        Applies one step of coefficient optimisation to a batch of candidates

        :param candidates
        :param states: Optimiser states of each candidate
        :param data
        
        Returns: Candidates with optimised coefficients
        """

        candidates, states, data = carry
        loss, gradients = self.vmap_gradients(candidates[...,3:], candidates[...,:3], data) #Compute loss and gradients parallely

        nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(loss)
        loss = jnp.where(nan_or_inf, jnp.ones(loss.shape)*self.max_fitness, loss)
            
        loss = jnp.clip(loss,0,self.max_fitness)

        updates, states = jax.vmap(self.optimiser.update)(gradients, states, candidates[...,3]) #Compute updates parallely
        new_candidates = candidates.at[...,3:].set(jax.vmap(lambda t, u: t + u)(candidates[...,3:], updates)) #Apply updates to coefficients parallely
        
        return (new_candidates, states, data), (candidates, loss)

    def optimise_gradient(self, candidates: Array, data: Tuple, key: PRNGKey, n_epoch: int):
        """Optimises the constants in the candidates

        :param candidates: Candidate solutions
        :param data: The data required to evaluate the population
        :param n_epoch: Number of steps to optimise coefficients

        Returns: Optimised and evaluated candidate.
        """

        states = jax.vmap(self.optimiser.init)(candidates[...,3:]) #Initialize optimisers for each candidate

        _, out = jax.lax.scan(self.optimise_epoch, (candidates, states, data), length=n_epoch)

        new_candidates, loss = out

        fitness = jnp.min(loss, axis=0) #Get best fitness during optimisation
        candidates = jax.vmap(lambda t, i: t[i], in_axes=[1,0])(new_candidates, jnp.argmin(loss, axis=0)) #Get best candidate during optimisation

        return fitness, candidates
    
    def optimise_generation(self, carry, x: int):
        candidate, data, key = carry

        key, sample_key = jr.split(key)

        mask = candidate[...,0] == 1.0
        mutations = jax.vmap(lambda _key: self.learning_rate*jr.normal(_key, shape=(self.num_trees, self.max_nodes,)) * mask)(jr.split(sample_key, self.n_offspring))
        mutations = jnp.vstack([jnp.zeros((1, self.num_trees, self.max_nodes)), mutations])

        offspring = jax.vmap(lambda m: candidate.at[...,3].set(candidate[...,3] + m))(mutations)

        fitness = self.vmap_trees(offspring[...,3:], offspring[...,:3], data)

        nan_or_inf = jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(self.n_offspring + 1)*self.max_fitness, fitness)
            
        fitness = jnp.clip(fitness,0,self.max_fitness)

        return (offspring[jnp.argmin(fitness)], data, key), jnp.min(fitness)


    def optimise_ES(self, candidate: Array, data: Tuple, key: PRNGKey, n_iterations: int):
        (new_candidate, _, _), fitness = jax.lax.scan(self.optimise_generation, (candidate, data, key), length=n_iterations)

        return jnp.min(fitness), new_candidate

    def increase_generation(self):
        self.current_generation += 1

    def punish_duplicates(self, population, fitness):
        _, indices, counts = jnp.unique(population, return_index=True, return_counts=True, axis=0, size=self.population_size)
        population = population[indices]
        fitness = fitness[indices]
        return population, jnp.where(counts > 0, fitness, self.max_fitness)

    def evolve(self, populations: Array, fitness: Array, key: PRNGKey) -> Array:
        """
        Evolves each population independently

        :param population: Populations of candidates
        :param fitness: Fitness of candidates
        :param key

        Returns: Evolved populations
        
        """
        populations, fitness = jax.vmap(self.punish_duplicates)(populations, fitness)

        new_populations = evolve_populations(self.jit_evolve_population, 
                                         populations, 
                                         fitness, 
                                         key, 
                                         self.current_generation, 
                                         self.migration_period, 
                                         self.migration_size, 
                                         self.reproduction_type_probabilities, 
                                         self.reproduction_probabilities, 
                                         self.tournament_probabilities)
        self.increase_generation()
        self.learning_rate = jnp.maximum(self.learning_rate * self.learning_rate_decay, 0.001)
        return self.jit_simplification(new_populations)
    
    def mutate_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Mutates a pair of candidates

        :param parent1
        :param parent2
        :param keys
        :param reproduction_probability: Probability of a tree to be mutated
        
        Returns: Pair of candidates after mutation
        """
        offspring = jax.vmap(self.mutate_trees, in_axes=[0,1,None,None])(jnp.stack([parent1, parent2]), keys, reproduction_probability, self.variable_array)
        return offspring[0], offspring[1]

    def sample_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Samples a pair of candidates

        :param parent1
        :param parent2
        :param keys
        :param reproduction_probability: Probability of a tree to be mutated
        
        Returns: Pair of candidates 
        """
        offspring = jax.vmap(lambda _keys: jax.vmap(self.sample_tree, in_axes=[0, None, 0])(_keys, self.max_init_depth, self.variable_array), in_axes=[1])(keys)
        return offspring[0], offspring[1]
    
    def get_statistics(self, generation: int = None) -> Tuple[Array | int, Array]:
        """Returns best fitness and best solution.

        :param generation: Generation of which the best fitness and solution are required. If None, returns all best fitness and solutions.

        Returns: Best fitness and best solution.
        """
        if generation is not None:
            return self.best_fitnesses[generation], self.best_solutions[generation]
        else:
            return self.best_fitnesses, self.best_solutions
        
    def simplify_coefficients(self, i, carry):
        tree, tree_indices, empty_tree = carry

        last_node_idx = jnp.sum(tree[:,0]==0)
        f_idx, a_idx, b_idx, coefficient = tree[i]

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

    def simplify_tree(self, tree):
        tree, _, _ = jax.lax.fori_loop(0, self.max_nodes, self.simplify_coefficients, (tree, self.tree_indices, self.empty_tree))

        return tree
    
    def generation(self, carry, _):
        populations, key, data = carry
        populations = jax.device_put(populations, NamedSharding(self.mesh, P('i')))
        
        fitness = self.jit_eval(populations, data) #Evaluate the candidates

        #Optimise coefficients of the best candidates given conditions
        # if (self.coefficient_optimisation & (self.current_generation>=self.start_coefficient_optimisation)):
        #     best_candidates_idx = jnp.argsort(fitness)[:,:self.optimise_coefficients_elite]
        #     # best_candidates = flat_populations[best_candidates_idx]
        #     best_candidates = jax.vmap(lambda pop, idx: pop[idx])(populations, best_candidates_idx)
        #     optimised_fitness, optimised_population = self.jit_optimise(best_candidates, data)
        #     # flat_populations = flat_populations.at[best_candidates_idx].set(optimised_population)
        #     populations = jax.vmap(lambda pop, idx, opt_pop: pop.at[idx].set(opt_pop))(populations, best_candidates_idx, optimised_population)
        #     fitness = jax.vmap(lambda f, idx, opt_f: f.at[idx].set(opt_f))(fitness, best_candidates_idx, optimised_fitness)

        fitness = fitness + self.size_parsimony * jnp.sum(populations[:,:,0]!=0, axis=[2,3]) #Increase fitness based on the size of the candidate

        best_idx = jnp.argmin(fitness)
        best_solution = populations[best_idx//self.population_size, best_idx%self.population_size]
        best_fitness = jnp.min(fitness)

        key, sample_key = jr.split(key)
        populations = self.evolve(populations, fitness, sample_key)

        next_carry = (populations, key, data)

        return next_carry, (best_fitness, best_solution)
    
    def fit(self, key, data):
        init_key, scan_key = jr.split(key)
        init_population = self.initialize_population(init_key)
        data = jax.device_put(data, self.data_mesh)
        init_carry = (init_population, scan_key, data)
        scan_fn = jax.jit(self.generation)
        carry, output = jax.lax.scan(scan_fn, init_carry, length = self.num_generations)

        best_fitness, best_solutions = output

        return best_fitness, best_solutions