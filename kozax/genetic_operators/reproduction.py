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
                         crowding_distances: Array,
                         key: PRNGKey, 
                         tournament_size: int, 
                         population_indices: Array) -> Array:
    """Selects a candidate for reproduction from a tournament using NSGA-II selection.

    Parameters
    ----------
    population : Array
        Population of candidates.
    ranks : Array
        Dominance ranks of candidates.
    crowding_distances : Array
        Crowding distances of candidates.
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

    tournament_ranks = ranks[indices]
    tournament_distances = crowding_distances[indices]
    
    # NSGA-II selection: prefer lower rank, then higher crowding distance
    # Create selection probabilities based on dominance and crowding
    min_rank = jnp.min(tournament_ranks)
    best_rank_mask = (tournament_ranks == min_rank)

    max_distance = jnp.max(tournament_distances * best_rank_mask)
    best_distance_mask = (tournament_distances * best_rank_mask) == max_distance
    
    winner_index = jr.choice(winner_key, indices, p=best_distance_mask)

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

def calculate_crowding_distance(metrics: Array, front_mask: Array) -> Array:
    """
    Calculate crowding distance for solutions in a Pareto front.
    
    Parameters
    ----------
    metrics : Array
        Array of shape (n_solutions, n_objectives) with metrics for each solution.
    front_mask : Array
        Boolean mask indicating which solutions belong to the current front.
        
    Returns
    -------
    Array
        Crowding distances for all solutions (0 for solutions not in the front).
    """
    n_solutions, n_objectives = metrics.shape
    distances = jnp.zeros(n_solutions)
    
    def compute_objective_distance(obj_idx):
        # Extract metrics for this objective
        obj_metrics = metrics[:, obj_idx]
        
        # Sort front solutions by this objective
        front_obj_metrics = jnp.where(front_mask, obj_metrics, jnp.inf)
        sorted_indices = jnp.argsort(front_obj_metrics)
        sorted_values = front_obj_metrics[sorted_indices]
        
        # Mask for valid (finite) values
        is_valid = (sorted_values < jnp.inf)
        
        # Calculate range for normalization
        valid_values = jnp.where(is_valid, sorted_values, 0.0)
        min_val = jnp.min(jnp.where(is_valid, sorted_values, jnp.inf))
        max_val = jnp.max(valid_values)
        obj_range = jnp.maximum(max_val - min_val, 1e-10)
        
        # Calculate distances using vectorized operations
        prev_values = jnp.roll(sorted_values, 1)
        next_values = jnp.roll(sorted_values, -1)
        prev_valid = jnp.roll(is_valid, 1)
        next_valid = jnp.roll(is_valid, -1)
        
        # Find boundary positions using cumulative sum
        cumsum_valid = jnp.cumsum(is_valid)
        total_valid = cumsum_valid[-1]
        is_first = is_valid & (cumsum_valid == 1)
        is_last = is_valid & (cumsum_valid == total_valid)
        is_boundary = is_first | is_last
        
        # Calculate crowding contribution for each sorted position
        crowding_contrib = jnp.where(
            is_valid & is_boundary,
            jnp.inf,  # Boundary points get infinite distance
            jnp.where(
                is_valid & prev_valid & next_valid & (~is_boundary),
                (next_values - prev_values) / obj_range,  # Internal points
                0.0  # Invalid or edge cases
            )
        )
        
        # Map back to original indices using scatter_add
        obj_distances = jnp.zeros(n_solutions).at[sorted_indices].add(crowding_contrib)
        
        return obj_distances
    
    # Vectorize over all objectives and sum distances
    obj_distances_all = jax.vmap(compute_objective_distance)(jnp.arange(n_objectives))
    distances = jnp.sum(obj_distances_all, axis=0)
    
    return distances

def nsga2(metrics: Array) -> Tuple[Array, Array]:
    """
    Performs NSGA-II ranking with crowding distance calculation.

    Parameters
    ----------
    metrics : Array
        Array of shape (population_size, n_objectives) with metrics for each individual.

    Returns
    -------
    Tuple[Array, Array]
        Dominance ranks and crowding distances for each individual.
    """
    # JIT-friendly NSGA-II: precompute pairwise dominance matrix once,
    # then iteratively extract non-dominated fronts using boolean masks.
    n_solutions = metrics.shape[0]

    # Pairwise comparison: does solution j dominate i?
    metrics_i = jnp.expand_dims(metrics, axis=1)  # (n,1,d)
    metrics_j = jnp.expand_dims(metrics, axis=0)  # (1,n,d)

    j_better_or_equal = (metrics_j <= metrics_i)
    j_strictly_better = (metrics_j < metrics_i)
    j_dominates_i = jnp.all(j_better_or_equal, axis=2) & jnp.any(j_strictly_better, axis=2)
    # avoid self-dominance
    j_dominates_i = j_dominates_i & (~jnp.eye(n_solutions, dtype=bool))

    # Initialize state
    alive = jnp.ones(n_solutions, dtype=bool)
    ranks = -jnp.ones(n_solutions, dtype=jnp.int32)
    distances = jnp.zeros(n_solutions)
    current_rank = jnp.array(0, dtype=jnp.int32)

    def cond_fun(state):
        alive_mask, _, _, _ = state
        return jnp.any(alive_mask)

    def body_fun(state):
        alive_mask, ranks, distances, current_rank = state

        # Count how many alive solutions dominate each i
        dominated_count = jnp.sum(j_dominates_i & alive_mask[None, :], axis=1)

        # Front: alive and dominated_count == 0
        front = (dominated_count == 0) & alive_mask

        # Compute crowding for this front
        front_crowding = calculate_crowding_distance(metrics, front)

        ranks = jnp.where(front, current_rank, ranks)
        distances = jnp.where(front, front_crowding, distances)

        alive_mask = alive_mask & (~front)
        current_rank = current_rank + 1

        return alive_mask, ranks, distances, current_rank

    alive_mask, ranks, distances, _ = jax.lax.while_loop(cond_fun, body_fun, (alive, ranks, distances, current_rank))

    return ranks, distances

def evolve_population(population: Array, 
                      ranks: Array,
                      crowding_distances: Array,
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
        Dominance ranks of the candidates.
    crowding_distances : Array
        Crowding distances of the candidates.
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
    left_parents = jax.vmap(tournament_selection, in_axes=[None, None, None, 0, None, None])(population, 
                                                                                             ranks,
                                                                                             crowding_distances,
                                                                                             jr.split(left_key, population_size//2), 
                                                                                             tournament_size,
                                                                                             population_indices)
    
    right_parents = jax.vmap(tournament_selection, in_axes=[None, None, None, 0, None, None])(population, 
                                                                                              ranks,
                                                                                              crowding_distances,
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
    order = jnp.lexsort((-crowding_distances, ranks))
    evolved_population = jnp.where((population_indices < 5)[:,None,None,None], population[order], jnp.concatenate([left_children, right_children], axis=0))

    return evolved_population

def migrate_population(receiver: Array, 
                       sender: Array, 
                       receiver_metrics: Array, 
                       sender_metrics: Array, 
                       receiver_ranks: Array,
                       sender_ranks: Array,
                       receiver_distances: Array,
                       sender_distances: Array,
                       migration_size: int,
                       population_indices: Array) -> Tuple[Array, Array, Array, Array]:
    """Replace worst receiver candidates with best sender candidates.

    Ordering:
    - Best: rank ascending, then crowding distance descending.
    - Worst: rank descending, then crowding distance ascending.
    """
    # Best sender individuals: low rank first, and within rank high crowding distance first
    sender_order = jnp.lexsort((-sender_distances, sender_ranks))

    # Worst receiver individuals: high rank first, and within rank low crowding distance first
    receiver_order = jnp.lexsort((receiver_distances, -receiver_ranks))

    migrated_population = jnp.where((population_indices < migration_size)[:,None,None,None], sender[sender_order], receiver[receiver_order])
    migrated_metrics = jnp.where((population_indices < migration_size)[:,None], sender_metrics[sender_order], receiver_metrics[receiver_order])

    new_ranks, new_crowding_distances = nsga2(migrated_metrics)

    return migrated_population, migrated_metrics, new_ranks, new_crowding_distances

def evolve_populations(jit_evolve_population: Callable, 
                       populations: Array, 
                       metrics: Array, 
                       key: PRNGKey, 
                       current_generation: int, 
                       migration_period: int, 
                       migration_size: int, 
                       reproduction_type_probabilities: Array, 
                       reproduction_probabilities: Array,
                       population_indices: Array) -> Array:
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

    nsga_ranks, crowding_distances = jax.vmap(nsga2, out_axes=(0, 0))(metrics)

    # Migrate candidates between populations. The populations and fitnesses are rolled for circular migration.
    populations, metrics, nsga_ranks, crowding_distances = jax.lax.cond((num_populations > 1) & (((current_generation+1)%migration_period) == 0), 
                                    jax.vmap(migrate_population, in_axes=[0, 0, 0, 0, 0, 0, 0, 0, None, None]), 
                                    lambda pops, rolled_pops, mets, rolled_mets, ranks, rolled_ranks, distances, rolled_distances, migration_size, indices: (pops, mets, ranks, crowding_distances), 
                                        populations, 
                                        jnp.roll(populations, 1, axis=0), 
                                        metrics, 
                                        jnp.roll(metrics, 1, axis=0), 
                                        nsga_ranks,
                                        jnp.roll(nsga_ranks, 1, axis=0),
                                        crowding_distances, 
                                        jnp.roll(crowding_distances, 1, axis=0),
                                        migration_size,
                                        population_indices)

    new_population = jit_evolve_population(populations, 
                                        nsga_ranks,
                                        crowding_distances,
                                        jr.split(key, num_populations), 
                                        reproduction_type_probabilities, 
                                        reproduction_probabilities, 
                                        population_indices)
    return new_population