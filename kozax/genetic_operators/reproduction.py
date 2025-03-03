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
                         fitness: Array, 
                         key: PRNGKey, 
                         tournament_probabilities: Array, 
                         tournament_size: int, 
                         population_indices: Array) -> Array:
    """Selects a candidate for reproduction from a tournament.

    Parameters
    ----------
    population : Array
        Population of candidates.
    fitness : Array
        Fitness of candidates.
    key : PRNGKey
        Random key.
    tournament_probabilities : Array
        Probability of each of the ranks in the tournament to be selected for reproduction.
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
    indices = jr.choice(tournament_key, population_indices, shape=(tournament_size,))

    index = jr.choice(winner_key, indices[jnp.argsort(fitness[indices])], p=tournament_probabilities)
    return population[index]

def evolve_population(population: Array, 
                      fitness: Array, 
                      key: PRNGKey, 
                      reproduction_type_probabilities: Array, 
                      reproduction_probability: float, 
                      tournament_probabilities: Array, 
                      population_indices: Array, 
                      population_size: int, 
                      tournament_size: int, 
                      num_trees: int, 
                      elite_size: int, 
                      reproduction_functions: List[Callable]) -> Array:
    """Reproduces pairs of candidates to obtain a new population.

    Parameters
    ----------
    population : Array
        Population of candidates.
    fitness : Array
        Fitness of candidates.
    key : PRNGKey
        Random key.
    reproduction_type_probabilities : Array
        Probability of each reproduction function to be applied.
    reproduction_probability : float
        Probability of a tree to be mutated.
    tournament_probabilities : Array
        Probability of each of the ranks in the tournament to be selected for reproduction.
    population_indices : Array
        Indices of the population.
    population_size : int
        Size of the population.
    tournament_size : int
        Size of the tournament.
    num_trees : int
        Number of trees in a candidate.
    elite_size : int
        Number of candidates that remain in the new population without reproduction.
    reproduction_functions : List[Callable]
        Functions that can be applied for reproduction.

    Returns
    -------
    Array
        Evolved population.
    """
    left_key, right_key, repro_key, evo_key = jr.split(key, 4)
    elite = population[jnp.argsort(fitness)[:elite_size]]

    # Sample parents for reproduction
    left_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, 
                                                                                             fitness, 
                                                                                             jr.split(left_key, (population_size - elite_size)//2), 
                                                                                             tournament_probabilities, 
                                                                                             tournament_size, 
                                                                                             population_indices)
    
    right_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, 
                                                                                              fitness, 
                                                                                              jr.split(right_key, (population_size - elite_size)//2), 
                                                                                              tournament_probabilities, 
                                                                                              tournament_size, 
                                                                                              population_indices)
    # Sample which reproduction function to apply to the parents
    reproduction_type = jr.choice(repro_key, jnp.arange(3), shape=((population_size - elite_size)//2,), p=reproduction_type_probabilities)

    left_children, right_children = jax.vmap(evolve_trees, in_axes=[0, 0, 0, 0, None, None])(left_parents, 
                                                                                             right_parents, 
                                                                                             jr.split(evo_key, ((population_size - elite_size)//2, num_trees, 2)), 
                                                                                             reproduction_type, 
                                                                                             reproduction_probability, 
                                                                                             reproduction_functions)
    
    evolved_population = jnp.concatenate([elite, left_children, right_children], axis=0)
    return evolved_population

def migrate_population(receiver: Array, 
                       sender: Array, 
                       receiver_fitness: Array, 
                       sender_fitness: Array, 
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
    migration_size : int
        How many candidates are migrated to new population.
    population_indices : Array
        Indices of the population.

    Returns
    -------
    Tuple[Array, Array]
        Population after migration and their fitness.
    """
    sorted_receiver = receiver[jnp.argsort(receiver_fitness, descending=True)]
    sorted_sender = sender[jnp.argsort(sender_fitness, descending=False)]
    migrated_population = jnp.where((population_indices < migration_size)[:,None,None,None], sorted_sender, sorted_receiver)
    migrated_fitness = jnp.where(population_indices < migration_size, jnp.sort(sender_fitness, descending=False), jnp.sort(receiver_fitness, descending=True))
    return migrated_population, migrated_fitness

def evolve_populations(jit_evolve_population: Callable, 
                       populations: Array, 
                       fitness: Array, 
                       key: PRNGKey, 
                       current_generation: int, 
                       migration_period: int, 
                       migration_size: int, 
                       reproduction_type_probabilities: Array, 
                       reproduction_probabilities: Array, 
                       tournament_probabilities: Array) -> Array:
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
    tournament_probabilities : Array
        Probability of each of the ranks in the tournament to be selected for reproduction.

    Returns
    -------
    Array
        Evolved populations.
    """
    num_populations, population_size, num_trees, _, _ = populations.shape
    population_indices = jnp.arange(population_size)

    # Migrate candidates between populations. The populations and fitnesses are rolled for circular migration.
    populations, fitness = jax.lax.cond((num_populations > 1) & (((current_generation+1)%migration_period) == 0), 
                                    jax.vmap(migrate_population, in_axes=[0, 0, 0, 0, None, None]), 
                                    jax.vmap(lambda receiver, sender, receiver_fitness, sender_fitness, migration_size, population_indices: (receiver, receiver_fitness), in_axes=[0, 0, 0, 0, None, None]), 
                                        populations, 
                                        jnp.roll(populations, 1, axis=0), 
                                        fitness, 
                                        jnp.roll(fitness, 1, axis=0), 
                                        migration_size, 
                                        population_indices)
    
    new_population = jit_evolve_population(populations, 
                                        fitness, 
                                        jr.split(key, num_populations), 
                                        reproduction_type_probabilities, 
                                        reproduction_probabilities, 
                                        tournament_probabilities, 
                                        population_indices)
    return new_population