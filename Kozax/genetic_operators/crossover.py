"""
kozax: Genetic programming framework in JAX

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
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, Callable
from jax import Array
from jax.random import PRNGKey

def sample_indices(carry: Tuple[PRNGKey, Array, float]) -> Tuple[PRNGKey, Array, float]:
    """
    Samples indices of the trees in a candidate that will be mutated
    
    :param key
    :param indices: Indices of trees
    :param reproduction_probability: Probability that determines how likely a tree is mutated

    Returns: The indices of the trees to be mutated
    """

    key, indices, reproduction_probability = carry
    indices = jr.bernoulli(key, p=reproduction_probability, shape=indices.shape)*1.0
    return (jr.split(key, 1)[0], indices, reproduction_probability)

def find_end_idx(carry: Tuple[Array, int, int]) -> Tuple[Array, int, int]:
    """
    Finds the index of the last node in a subtree

    :param tree
    :param open_slots: The number of open slots in the tree that need to be matched with a node
    :param counter: The index of the current node

    Returns: The index of the last node in the subtree
    """

    tree, open_slots, counter = carry
    _, idx1, idx2, _ = tree[counter]
    open_slots -= 1 #Reduce open slot for current node
    open_slots = jax.lax.select(idx1 < 0, open_slots, open_slots+1) #Increase the open slots for a child
    open_slots = jax.lax.select(idx2 < 0, open_slots, open_slots+1) #Increase the open slots for a child
    counter -= 1
    return (tree, open_slots, counter)

def check_invalid_cx_nodes(carry: Tuple[Array, Array, Array, int, int, Array, Array]) -> bool:
    """
    Checks if the sampled subtrees are different and if the trees after crossover are valid 
    
    :param tree1
    :param tree
    :param node_idx1: Index of node in the first tree
    :param node_idx2: Index of node in the second tree
    
    Returns: If the sampled nodes are valid nodes for crossover
    """

    tree1, tree2, _, node_idx1, node_idx2, _, _ = carry

    _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree1, 1, node_idx1))
    _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree2, 1, node_idx2))

    subtree_size1 = node_idx1 - end_idx1
    subtree_size2 = node_idx2 - end_idx2

    empty_nodes1 = jnp.sum(tree1[:,0]==0)
    empty_nodes2 = jnp.sum(tree2[:,0]==0)

    #Check if the subtrees can be inserted
    return (empty_nodes1 < subtree_size2 - subtree_size1) | (empty_nodes2 < subtree_size1 - subtree_size2)

def sample_cx_nodes(carry: Tuple[Array, Array, Array, int, int, Array, Array]) -> Tuple[Array, Array, Array, int, int, Array, Array]:
    """
    Samples nodes in a pair of trees for crossover
    
    :param tree1
    :param tree
    :param keys
    :param node_ids: Indices of all the nodes in the trees
    :param operator_indices: The indices that belong to operator nodes
    
    Returns: Sampled nodes
    """

    tree1, tree2, keys, _, _, node_ids, operator_indices = carry
    key1, key2 = keys

    #Sample nodes from the non-empty nodes, with higher probability for operator nodes
    cx_prob1 = jnp.isin(tree1[:,0], operator_indices)
    cx_prob1 = jnp.where(tree1[:,0]==0, cx_prob1, cx_prob1+1)
    node_idx1 = jr.choice(key1, node_ids, p = cx_prob1*1.)

    cx_prob2 = jnp.isin(tree2[:,0], operator_indices)
    cx_prob2 = jnp.where(tree2[:,0]==0, cx_prob2, cx_prob2+1)
    node_idx2 = jr.choice(key2, node_ids, p = cx_prob2*1.)

    return (tree1, tree2, jr.split(key1), node_idx1, node_idx2, node_ids, operator_indices)

def tree_crossover(tree1: Array, 
              tree2: Array, 
              keys: Array,
              node_ids: Array,
              operator_indices: Array) -> Tuple[Array, Array]:
    """
    Applies crossover to a pair of trees to produce two new trees
    
    :param tree1
    :param tree2
    :param keys
    :param max_nodes: Max number of nodes in a tree
    :param operator_indices: The indices that belong to operator nodes
    
    Returns: Pair of new trees
    """

    #Define indices of the nodes
    tree_indices = jnp.tile(node_ids[:,None], reps=(1,4))
    key1, key2 = keys

    #Define last node in tree
    last_node_idx1 = jnp.sum(tree1[:,0]==0)
    last_node_idx2 = jnp.sum(tree2[:,0]==0)

    #Randomly select nodes for crossover
    _, _, _, node_idx1, node_idx2, _, _ = sample_cx_nodes((tree1, tree2, jr.split(key1), 0, 0, node_ids, operator_indices))

    #Reselect until valid crossover nodes have been found
    _, _, _, node_idx1, node_idx2, _, _ = jax.lax.while_loop(check_invalid_cx_nodes, sample_cx_nodes, (tree1, tree2, jr.split(key2), node_idx1, node_idx2, node_ids, operator_indices))

    #Retrieve subtrees of selected nodes
    _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree1, 1, node_idx1))
    _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree2, 1, node_idx2))

    #Initialize children
    child1 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (len(node_ids), 1))
    child2 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (len(node_ids), 1))

    #Compute subtree sizes
    subtree_size1 = node_idx1 - end_idx1
    subtree_size2 = node_idx2 - end_idx2

    #Insert nodes before subtree in children
    child1 = jnp.where(tree_indices >= node_idx1 + 1, tree1, child1)
    child2 = jnp.where(tree_indices >= node_idx2 + 1, tree2, child2)
    
    #Align nodes after subtree with first open spot after new subtree in children
    rolled_tree1 = jnp.roll(tree1, subtree_size1 - subtree_size2, axis=0)
    rolled_tree2 = jnp.roll(tree2, subtree_size2 - subtree_size1, axis=0)

    #Insert nodes after subtree in children
    child1 = jnp.where((tree_indices >= node_idx1 - subtree_size2 - (end_idx1 - last_node_idx1)) & (tree_indices < node_idx1 + 1 - subtree_size2), rolled_tree1, child1)
    child2 = jnp.where((tree_indices >= node_idx2 - subtree_size1 - (end_idx2 - last_node_idx2)) & (tree_indices < node_idx2 + 1 - subtree_size1), rolled_tree2, child2)

    #Update index references to moved nodes in staying nodes
    child1 = child1.at[:,1:3].set(jnp.where((child1[:,1:3] < (node_idx1 - subtree_size1 + 1)) & (child1[:,1:3] > -1), child1[:,1:3] + (subtree_size1-subtree_size2), child1[:,1:3]))
    child2 = child2.at[:,1:3].set(jnp.where((child2[:,1:3] < (node_idx2 - subtree_size2 + 1)) & (child2[:,1:3] > -1), child2[:,1:3] + (subtree_size2-subtree_size1), child2[:,1:3]))

    #Align subtree with the selected node in children
    rolled_subtree1 = jnp.roll(tree1, node_idx2 - node_idx1, axis=0)
    rolled_subtree2 = jnp.roll(tree2, node_idx1 - node_idx2, axis=0)

    #Update index references in subtree
    rolled_subtree1 = rolled_subtree1.at[:,1:3].set(jnp.where(rolled_subtree1[:,1:3] > -1, rolled_subtree1[:,1:3] + (node_idx2 - node_idx1), -1))
    rolled_subtree2 = rolled_subtree2.at[:,1:3].set(jnp.where(rolled_subtree2[:,1:3] > -1, rolled_subtree2[:,1:3] + (node_idx1 - node_idx2), -1))

    #Insert subtree in selected node in children
    child1 = jnp.where((tree_indices >= node_idx1 + 1 - subtree_size2) & (tree_indices < node_idx1 + 1), rolled_subtree2, child1)
    child2 = jnp.where((tree_indices >= node_idx2 + 1 - subtree_size1) & (tree_indices < node_idx2 + 1), rolled_subtree1, child2)
    
    return child1, child2

def full_crossover(tree1: Array, 
              tree2: Array, 
              keys: Array,
              node_ids: Array,
              operator_indices: Array) -> Tuple[Array, Array]:

    return tree2, tree1

def crossover(tree1: Array, 
              tree2: Array, 
              keys: Array,
              node_ids: Array,
              operator_indices: Array,
              crossover_types: int) -> Tuple[Array, Array]:

    return jax.lax.cond(crossover_types, tree_crossover, full_crossover, tree1, tree2, keys, node_ids, operator_indices)

def check_different_tree(parent1, parent2, child1, child2):
    size1 = jnp.sum(child1[:,0]!=0)
    size2 = jnp.sum(child2[:,0]!=0)
    
    check1 = (jnp.all(parent1 == child1) | jnp.all(parent2 == child1))
    check2 = (jnp.all(parent1 == child2) | jnp.all(parent2 == child2))

    return ((check1 | check2) & ((size1 > 1) & (size2 > 1))) | (size1 == 0)

def check_different_trees(carry):
    parent1, parent2, child1, child2, _, _, _, _ = carry

    return jnp.all(jax.vmap(check_different_tree)(parent1, parent2, child1, child2))

def safe_crossover(carry: Tuple[Array, Array, Array, float, Array, Array]):
    parent1, parent2, _, _, keys, reproduction_probability, node_ids, operator_indices = carry
    index_key, type_key, new_key = jr.split(keys[0,0], 3)
    _, cx_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (index_key, jnp.zeros(parent1.shape[0]), reproduction_probability))
    crossover_types = jr.bernoulli(type_key, p = 0.9, shape = (parent1.shape[0],))
    offspring1, offspring2 = jax.vmap(crossover, in_axes=[0,0,0,None,None, 0])(parent1, parent2, keys, node_ids, operator_indices, crossover_types)
    child1 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent1), offspring1, parent1)
    child2 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent2), offspring2, parent2)

    keys = jr.split(new_key, keys.shape[:-1])

    return parent1, parent2, child1, child2, keys, reproduction_probability, node_ids, operator_indices

def crossover_trees(parent1: Array, 
                    parent2: Array, 
                    keys: Array, 
                    reproduction_probability: float, 
                    max_nodes: int, 
                    operator_indices: Array) -> Tuple[Array, Array]:
    """
    Applies crossover to the trees in a pair of candidates

    :param parent1
    :param parent2
    :param keys
    :param reproduction_probability: Probability of a tree to be mutated
    :param max_nodes: Max number of nodes in a tree
    :param operator_indices: The indices that belong to operator nodes 
    
    Returns: Pair of candidates after crossover    
    """

    #Determine to which trees in the candidates crossover is applied

    _, _, child1, child2, _, _, _, _ = jax.lax.while_loop(check_different_trees, safe_crossover, (
        parent1, parent2, jnp.zeros_like(parent1), jnp.zeros_like(parent2), keys, reproduction_probability, jnp.arange(max_nodes), operator_indices))
    
    return child1, child2