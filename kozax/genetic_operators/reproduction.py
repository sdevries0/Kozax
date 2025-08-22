"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
from typing import Callable, List, Tuple

def evolve_trees(parent1: Array, 
                 parent2: Array, 
                 keys: Array, 
                 type: int, 
                 reproduction_probability: float, 
                 reproduction_functions: List[Callable]) -> Tuple[Array, Array]:
    """Applies reproduction function to pair of candidates.

    Parameters
    ----------
    parent1 : Array
        First parent candidate.
    parent2 : Array
        Second parent candidate.
    keys : Array
        Random keys.
    type : int
        Type of reproduction function to apply.
    reproduction_probability : float
        Probability of a tree to be mutated.
    reproduction_functions : List[Callable]
        Functions that can be applied for reproduction.

    Returns
    -------
    Tuple[Array, Array]
        Pair of reproduced candidates.
    """
    child0, child1 = jax.lax.switch(type, reproduction_functions, parent1, parent2, keys, reproduction_probability)

    return child0, child1

def tournament_selection(population: Array, 
                         ranks: Array,
                         key: PRNGKey, 
                         tournament_size: int, 
                         population_indices: Array) -> Array:
    """Selects a candidate for reproduction from a tournament.

    Parameters
    ----------
    population : Array
        Population of candidates.
    ranks : Array
        Ranks of candidates.
    key : PRNGKey
        Random key.
    tournament_size : int
        Size of the tournament.
    population_indices : Array
        Indices of the population.

    Returns
    -------
    Array
        Candidate that won the tournament.
    """
    tournament_key, winner_key = jr.split(key)
    indices = jr.choice(tournament_key, population_indices, shape=(tournament_size,), replace=False)

    tournament_ranks = ranks[indices] + 1

    winner_index = jr.choice(winner_key, indices, p=1/tournament_ranks)

    return population[winner_index]

def identify_non_dominated(metrics: Array) -> Array:
    """
    Identifies non-dominated solutions in a set using vectorized operations.
    
    Parameters
    ----------
    metrics : Array
        Array of shape (n_solutions, n_objectives) with metrics for each solution.
        
    Returns
    -------
    Array
        Boolean mask identifying non-dominated solutions.
    """

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
    
    return dominated_by_others

def nsga2(metrics: Array) -> Array:
    """
    Selects individuals for the next generation using NSGA-II without crowding distance.

    Parameters
    ----------
    metrics : Array
        Array of shape (population_size, n_objectives) with metrics for each individual.
    selection_size : int
        The number of individuals to select.

    Returns
    -------
    Array
        Indices of the selected individuals.
    """
    all_indices = jnp.ones(metrics.shape[0])

    def cond_fun(state):
        selected_count, remaining_indices, _ = state
        return jnp.sum(remaining_indices) > 0

    def body_fun(state):
        current_rank, remaining_indices, ranks = state
        remaining_metrics = jnp.where(remaining_indices[:, None], metrics, jnp.inf * jnp.ones_like(metrics))
        non_dominated = identify_non_dominated(remaining_metrics)
        
        ranks = jnp.where(non_dominated, current_rank * jnp.ones_like(ranks), ranks)
        
        remaining_indices = jnp.where(non_dominated, jnp.zeros_like(remaining_indices), remaining_indices)
        
        return current_rank + 1, remaining_indices, ranks

    initial_state = (0, all_indices, all_indices * 0)
    
    _, _, ranks = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    return ranks

def evolve_population(population: Array, 
                      ranks: Array,
                      key: PRNGKey, 
                      reproduction_type_probabilities: Array, 
                      reproduction_probability: float, 
                      population_indices: Array, 
                      population_size: int, 
                      num_trees: int, 
                      tournament_size: int,
                      reproduction_functions: List[Callable]) -> Array:
    """Reproduces pairs of candidates to obtain a new population.

    Parameters
    ----------
    population : Array
        Population of candidates.
    ranks : Array
        Ranks of the candidates.
    key : PRNGKey
        Random key.
    reproduction_type_probabilities : Array
        Probability of each reproduction function to be applied.
    reproduction_probability : float
        Probability of a tree to be mutated.
    population_indices : Array
        Indices of the population.
    population_size : int
        Size of the population.
    num_trees : int
        Number of trees in a candidate.
    tournament_size : int
        Number of candidates that compete in a tournament.
    reproduction_functions : List[Callable]
        Functions that can be applied for reproduction.

    Returns
    -------
    Array
        Evolved population.
    """
    
    left_key, right_key, repro_key, evo_key = jr.split(key, 4)

    # Sample parents for reproduction
    left_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None])(population, 
                                                                                             ranks,
                                                                                             jr.split(left_key, population_size//2), 
                                                                                             tournament_size,
                                                                                             population_indices)
    
    right_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None])(population, 
                                                                                              ranks,
                                                                                              jr.split(right_key, population_size//2), 
                                                                                              tournament_size,
                                                                                              population_indices)
    # Sample which reproduction function to apply to the parents
    reproduction_type = jr.choice(repro_key, jnp.arange(3), shape=(population_size//2,), p=reproduction_type_probabilities)

    left_children, right_children = jax.vmap(evolve_trees, in_axes=[0, 0, 0, 0, None, None])(left_parents, 
                                                                                             right_parents, 
                                                                                             jr.split(evo_key, (population_size//2, num_trees, 2)), 
                                                                                             reproduction_type, 
                                                                                             reproduction_probability, 
                                                                                             reproduction_functions)
    
    evolved_population = jnp.where(ranks[:,None,None,None] == 0, population, jnp.concatenate([left_children, right_children], axis=0))

    return evolved_population

def migrate_population(receiver: Array, 
                       sender: Array, 
                       receiver_metrics: Array, 
                       sender_metrics: Array, 
                       receiver_ranks: Array,
                       sender_ranks: Array,
                       migration_size: int, 
                       population_indices: Array) -> Tuple[Array, Array]:
    """Unfit candidates from one population are replaced with fit candidates from another population.

    Parameters
    ----------
    receiver : Array
        Population that receives new candidates.
    sender : Array
        Population that sends fit candidates.
    receiver_fitness : Array
        Fitness of the candidates in the receiving population.
    sender_fitness : Array
        Fitness of the candidates in the sending population.
    receiver_ranks : Array
        Dominance ranks of candidates in the receiving population.
    sender_ranks : Array
        Dominance ranks of candidates in the sending population.
    migration_size : int
        How many candidates are migrated to new population.
    population_indices : Array
        Indices of the population.

    Returns
    -------
    Tuple[Array, Array, Array]
        Population after migration and the corresponding fitness and dominance ranks.
    """
    # Identify the Pareto front of the sender population
    sender_dominated_ranks = jnp.argsort(sender_ranks, descending=False)
    receiver_dominated_ranks = jnp.argsort(receiver_ranks, descending=True)

    # Replace the selected locations in the receiver with Pareto front individuals
    migrated_population = jnp.where((population_indices < migration_size)[:,None,None,None], sender[sender_dominated_ranks], receiver[receiver_dominated_ranks])
    migrated_metrics = jnp.where((population_indices < migration_size)[:,None], sender_metrics[sender_dominated_ranks], receiver_metrics[receiver_dominated_ranks])

    new_ranks = nsga2(migrated_metrics)

    return migrated_population, migrated_metrics, new_ranks

def evolve_populations(jit_evolve_population: Callable, 
                       populations: Array, 
                       metrics: Array, 
                       key: PRNGKey, 
                       current_generation: int, 
                       migration_period: int, 
                       migration_size: int, 
                       reproduction_type_probabilities: Array, 
                       reproduction_probabilities: Array) -> Array:
    """Evolves each population independently.

    Parameters
    ----------
    jit_evolve_population : Callable
        Function for evolving trees that is jittable and parallelizable.
    populations : Array
        Populations of candidates.
    fitness : Array
        Fitness of candidates.
    key : PRNGKey
        Random key.
    current_generation : int
        Current generation number.
    migration_period : int
        After how many generations migration occurs.
    migration_size : int
        How many candidates are migrated to new population.
    reproduction_type_probabilities : Array
        Probability of each reproduction function to be applied.
    reproduction_probabilities : Array
        Probability of a tree to be mutated.

    Returns
    -------
    Array
        Evolved populations.
    """
    num_populations, population_size, _, _, _ = populations.shape
    population_indices = jnp.arange(population_size)

    nsga_ranks = jax.vmap(nsga2)(metrics)

    # Migrate candidates between populations. The populations and fitnesses are rolled for circular migration.
    populations, metrics, nsga_ranks = jax.lax.cond((num_populations > 1) & (((current_generation+1)%migration_period) == 0), 
                                    jax.vmap(migrate_population, in_axes=[0, 0, 0, 0, 0, 0, None, None]), 
                                    jax.vmap(lambda receiver, sender, receiver_fitness, sender_fitness, receiver_ranks, sender_ranks, migration_size, population_indices: (receiver, receiver_fitness, receiver_ranks), 
                                             in_axes=[0, 0, 0, 0, 0, 0, None, None]), 
                                        populations, 
                                        jnp.roll(populations, 1, axis=0), 
                                        metrics, 
                                        jnp.roll(metrics, 1, axis=0), 
                                        nsga_ranks,
                                        jnp.roll(nsga_ranks, 1, axis=0), 
                                        migration_size, 
                                        population_indices)
    
    new_population = jit_evolve_population(populations, 
                                        nsga_ranks,
                                        jr.split(key, num_populations), 
                                        reproduction_type_probabilities, 
                                        reproduction_probabilities, 
                                        population_indices)
    return new_population