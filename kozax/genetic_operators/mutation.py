"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple, Callable, List
from jax import Array
from jax.random import PRNGKey

def find_end_idx(carry: Tuple[Array, int, int]) -> Tuple[Array, int, int]:
    """Finds the index of the last node in a subtree.

    Parameters
    ----------
    carry : Tuple[Array, int, int]
        Tuple containing the tree, the number of open slots, and the current node index.

    Returns
    -------
    Tuple[Array, int, int]
        Updated tuple with the tree, open slots, and current node index.
    """
    tree, open_slots, counter = carry
    row = tree[counter]
    open_slots -= 1
    open_slots += jnp.sum(row[1:-1] >= 0)
    counter -= 1
    return (tree, open_slots, counter)

def sample_indices(carry: Tuple[PRNGKey, Array, float]) -> Tuple[PRNGKey, Array, float]:
    """Samples indices of the trees in a candidate that will be mutated.

    Parameters
    ----------
    carry : Tuple[PRNGKey, Array, float]
        Tuple containing the random key, indices of trees, and reproduction probability.

    Returns
    -------
    Tuple[PRNGKey, Array, float]
        Updated tuple with the random key, indices of trees to be mutated, and reproduction probability.
    """
    key, indices, reproduction_probability = carry
    indices = jr.bernoulli(key, p=reproduction_probability, shape=indices.shape) * 1.0
    return (jr.split(key, 1)[0], indices, reproduction_probability)

def sample_leaf_node(carry: Tuple[Array, PRNGKey, int, int, Array, Array]) -> Tuple[Array, PRNGKey, int, int, Array, Array]:
    """Samples a leaf node to be replaced in the tree and a new leaf node.

    Parameters
    ----------
    carry : Tuple[Array, PRNGKey, int, int, Array, Array]
        Tuple containing the tree, random key, mutate index, new leaf, variable array, and variable indices.

    Returns
    -------
    Tuple[Array, PRNGKey, int, int, Array, Array]
        Updated tuple with the tree, random key, mutate index, new leaf, variable array, and variable indices.
    """
    tree, key, _, _, variable_array, variable_indices = carry
    key, select_key, sample_key, variable_key = jr.split(key, 4)
    node_ids = tree[:, 0]
    is_leaf = jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=is_leaf * 1.0)
    new_leaf = jr.choice(variable_key, variable_indices, shape=(), p=variable_array)
    return (tree, key, mutate_idx, new_leaf, variable_array, variable_indices)

def check_equal_leaves(carry: Tuple[Array, PRNGKey, int, int, Array, Array]) -> bool:
    """Checks that the old and new leaf node are different.

    Parameters
    ----------
    carry : Tuple[Array, PRNGKey, int, int, Array, Array]
        Tuple containing the tree, random key, mutate index, new leaf, variable array, and variable indices.

    Returns
    -------
    bool
        Whether the old and new leaf are different.
    """
    tree, _, mutate_idx, new_leaf, _, _ = carry
    return (tree[mutate_idx, 0] == new_leaf) & (new_leaf != 1)

def check_invalid_operator_node(carry: Tuple[Array, PRNGKey, int, int, int, Array, Array]) -> bool:
    """Checks that the old and new operator node are different and that the tree does not exceed the maximum size after sampling a new subtree.

    Parameters
    ----------
    carry : Tuple[Array, PRNGKey, int, int, int, Array, Array]
        Tuple containing the tree, random key, mutate index, new operator, slots, operator indices, and operator probabilities.

    Returns
    -------
    bool
        Whether the old and new operator are different and a valid subtree can be sampled.
    """
    tree, _, mutate_idx, new_operator, slots, operator_indices, _ = carry
    _, _, end_idx = jax.lax.while_loop(lambda c: c[1] > 0, find_end_idx, (tree, 1, mutate_idx))
    subtree_size = mutate_idx - end_idx
    empty_nodes = jnp.sum(tree[:, 0] == 0)

    # replace_with_subtrees samples one depth-1 subtree per required operand,
    # so the replacement needs (1 + arity) nodes total.
    required_nodes = 1 + slots[new_operator]

    return ((tree[mutate_idx, 0] == new_operator) & (len(operator_indices) > 1)) | (empty_nodes + subtree_size < required_nodes)

def sample_operator_node(carry: Tuple[Array, PRNGKey, int, int, int, Array, Array]) -> Tuple[Array, PRNGKey, int, int, int, Array, Array]:
    """Samples an operator node to be replaced in the tree and a new operator node.

    Parameters
    ----------
    carry : Tuple[Array, PRNGKey, int, int, int, Array, Array]
        Tuple containing the tree, random key, mutate index, new operator, slots, operator indices, and operator probabilities.

    Returns
    -------
    Tuple[Array, PRNGKey, int, int, int, Array, Array]
        Updated tuple with the tree, random key, mutate index, new operator, slots, operator indices, and operator probabilities.
    """
    tree, key, _, _, slots, operator_indices, operator_probabilities = carry
    key, select_key, sample_key = jr.split(key, 3)
    node_ids = tree[:, 0]
    is_operator = jnp.isin(node_ids, operator_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=is_operator * 1.0)
    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities)
    return (tree, key, mutate_idx, new_operator, slots, operator_indices, operator_probabilities)

def add_subtree(tree: Array, 
                key: PRNGKey, 
                variable_array: Array, 
                args: Tuple) -> Array:
    """Replaces a leaf node with a random subtree.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    select_key, sample_key = jr.split(key, 2)

    # Sample node to be mutated
    node_ids = tree[:, 0]
    is_leaf = jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=is_leaf * 1.0)
    subtree = sample_tree(sample_key, jnp.minimum(max_init_depth, 2), variable_array)
    subtree_size = jnp.sum(subtree[:, 0] != 0)
    remaining_size = mutate_idx - jnp.sum(tree[:, 0] == 0)
    
    # Create new tree
    empty_row = -jnp.ones(tree.shape[1]).at[0].set(0.0).at[-1].set(0.0)
    child = jnp.tile(empty_row, (max_nodes, 1))
    child = jnp.where(tree_indices > mutate_idx, tree, child)
    rolled_tree = jnp.roll(tree, -subtree_size + 1, axis=0)

    # Insert subtree in new tree
    child = jnp.where((tree_indices <= mutate_idx - subtree_size) & (tree_indices > mutate_idx - subtree_size - remaining_size), rolled_tree, child)
    child = child.at[:, 1:-1].set(jnp.where((child[:, 1:-1] < mutate_idx) & (child[:, 1:-1] > -1), child[:, 1:-1] - (subtree_size - 1), child[:, 1:-1]))

    # Update references to subtrees
    subtree = jnp.roll(subtree, -(max_nodes - mutate_idx - 1), axis=0)
    subtree = subtree.at[:, 1:-1].set(jnp.where(subtree[:, 1:-1] > -1, subtree[:, 1:-1] + (mutate_idx - max_nodes + 1), -1))

    child = jnp.where((tree_indices <= mutate_idx) & (tree_indices > mutate_idx - subtree_size), subtree, child)
    return child

def mutate_leaf(tree: Array, 
                key: PRNGKey, 
                variable_array: Array, 
                args: Tuple) -> Array:
    """Replaces a leaf node with a different leaf node.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    select_key, sample_key, coefficient_key, variable_key = jr.split(key, 4)

    # Sample node to be mutated
    node_ids = tree[:, 0]
    is_leaf = jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=is_leaf * 1.0)
    new_leaf = jr.choice(variable_key, variable_indices, shape=(), p=variable_array)
    coefficient = jr.normal(coefficient_key) * coefficient_sd

    # Check that the old and new leaf node are different
    _, _, mutate_idx, new_leaf, _, _ = jax.lax.while_loop(check_equal_leaves, sample_leaf_node, (tree, jr.fold_in(key, 0), mutate_idx, new_leaf, variable_array, variable_indices))
    
    # Insert new leaf node
    child = tree.at[mutate_idx, 0].set(new_leaf)
    child = jax.lax.select(new_leaf == 1, child.at[mutate_idx, -1].set(coefficient), child.at[mutate_idx, -1].set(0))
    return child

def replace_with_subtrees(tree: Array,
                          key: PRNGKey,
                          mutate_idx: int,
                          operator: int,
                          variable_array: Array,
                          args: Tuple,
                          num_subtrees: int) -> Array:
    """Replaces an operator node with a new operator and sampled subtrees."""
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    max_arity = tree.shape[1] - 2

    # Determine subtree to be replaced
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1] > 0, find_end_idx, (tree, 1, mutate_idx))
    remaining_size = end_idx - jnp.sum(tree[:, 0] == 0) + 1
    empty_row = -jnp.ones(tree.shape[1]).at[0].set(0.0).at[-1].set(0.0)

    child_cols = jnp.arange(max_arity)
    use_branch = child_cols < num_subtrees

    branch_keys = jr.split(key, max_arity)
    sampled_subtrees = jax.vmap(lambda k: sample_tree(k, 1, variable_array))(branch_keys)
    sampled_sizes = jnp.sum(sampled_subtrees[:, :, 0] != 0, axis=1).astype(jnp.int32)
    branch_sizes = jnp.where(use_branch, sampled_sizes, 0)
    total_subtree_size = jnp.sum(branch_sizes)
    cumulative_prev = jnp.cumsum(branch_sizes) - branch_sizes
    branch_roots = jnp.where(use_branch, mutate_idx - 1 - cumulative_prev, -1)

    # Create new tree
    child = jnp.tile(empty_row, (max_nodes, 1))
    child = jnp.where(tree_indices >= mutate_idx, tree, child)

    # Insert the preserved part of the old tree after the new operator/subtrees.
    rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - total_subtree_size - 1), axis=0)
    child = jnp.where(
        (tree_indices < mutate_idx - total_subtree_size) &
        (tree_indices >= mutate_idx - total_subtree_size - remaining_size),
        rolled_tree,
        child,
    )
    child = child.at[:, 1:-1].set(
        jnp.where(
            (child[:, 1:-1] <= end_idx) & (child[:, 1:-1] > -1),
            child[:, 1:-1] + (mutate_idx - end_idx - total_subtree_size - 1),
            child[:, 1:-1],
        )
    )

    # Insert the new operator node.
    child = child.at[mutate_idx, 0].set(operator)
    child = child.at[mutate_idx, 1:-1].set(-1)
    child = child.at[mutate_idx, 1:-1].set(branch_roots)

    def place_branch(i, carry):
        current_child = carry
        size = branch_sizes[i]
        root_idx = branch_roots[i]
        branch = sampled_subtrees[i]
        shift = root_idx - (max_nodes - 1)

        rolled = jnp.roll(branch, shift, axis=0)
        rolled = rolled.at[:, 1:-1].set(jnp.where(rolled[:, 1:-1] > -1, rolled[:, 1:-1] + shift, -1))
        mask = (tree_indices <= root_idx) & (tree_indices > root_idx - size)
        return jax.lax.select(use_branch[i], jnp.where(mask, rolled, current_child), current_child)

    child = jax.lax.fori_loop(0, max_arity, place_branch, child)

    return child

def mutate_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """Replaces an operator node with a different operator node. The arity of the operator might change, therefore new subtrees may be sampled.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    select_key, sample_key, subtree_key = jr.split(key, 3)
    node_ids = tree[:, 0]
    is_operator = jnp.isin(node_ids, operator_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=is_operator * 1.0)  # Sample node to be mutated

    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities)  # Sample new operator

    # Check that the new operator is different from the old operator
    _, _, mutate_idx, new_operator, _, _, _ = jax.lax.while_loop(check_invalid_operator_node, sample_operator_node, (tree, 
                                                                                                                     jr.fold_in(key, 0),
                                                                                                                     mutate_idx, 
                                                                                                                     new_operator, 
                                                                                                                     slots, 
                                                                                                                     operator_indices, 
                                                                                                                     operator_probabilities))

    current_slots = slots[node_ids[mutate_idx].astype(int)]
    new_slots = slots[new_operator]

    # Keep exact-arity replacements in-place; otherwise rebuild the local subtree
    # with exactly the required number of operands.
    child = jax.lax.cond(
        current_slots == new_slots,
        lambda _: tree.at[mutate_idx, 0].set(new_operator),
        lambda _: replace_with_subtrees(tree, subtree_key, mutate_idx, new_operator, variable_array, args, new_slots),
        operand=None,
    )

    return child

def delete_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """Replaces an operator and operands with a leaf node.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    select_key, sample_key, coefficient_key, variable_key = jr.split(key, 4)

    # Sample node to be mutated (non-root operator only).
    node_ids = tree[:, 0]
    is_operator = jnp.isin(node_ids, operator_indices)
    deletable_mask = is_operator.at[-1].set(False)
    has_deletable = jnp.any(deletable_mask)

    def do_delete(_):
        delete_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=deletable_mask.astype(jnp.float32))

        # Determine subtree to be replaced.
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1] > 0, find_end_idx, (tree, 1, delete_idx))
        removed_count = (delete_idx - end_idx - 1).astype(jnp.int32)  # descendants removed; root becomes leaf

        coefficient = jr.normal(coefficient_key) * coefficient_sd
        new_leaf = jr.choice(variable_key, variable_indices, shape=(), p=variable_array)  # Sample coefficient or variable

        empty_row = -jnp.ones(tree.shape[1]).at[0].set(0.0).at[-1].set(0.0)
        child = jnp.tile(empty_row, (max_nodes, 1))

        # Keep all active rows except the deleted root and descendants; map indices explicitly.
        row_ids = jnp.arange(max_nodes)
        active_mask = tree[:, 0] != 0
        removed_desc_mask = (row_ids > end_idx) & (row_ids < delete_idx)
        keep_mask = active_mask & (~removed_desc_mask) & (row_ids != delete_idx)

        new_row_ids = jnp.where(row_ids <= end_idx, row_ids + removed_count, row_ids).astype(jnp.int32)
        safe_new_ids = jnp.clip(new_row_ids, 0, max_nodes - 1)

        def scatter_kept(i, current_child):
            return jax.lax.cond(
                keep_mask[i],
                lambda c: c.at[safe_new_ids[i]].set(tree[i]),
                lambda c: c,
                current_child,
            )

        child = jax.lax.fori_loop(0, max_nodes, scatter_kept, child)

        # Insert replacement leaf at the original root location.
        child = child.at[delete_idx, 0].set(new_leaf)
        child = child.at[delete_idx, 1:-1].set(-1)
        child = jax.lax.select(new_leaf == 1, child.at[delete_idx, -1].set(coefficient), child.at[delete_idx, -1].set(0))

        # Remap all child references to the new indices.
        refs = child[:, 1:-1].astype(jnp.int32)
        valid_ref = refs >= 0
        remapped_refs = jnp.where(
            refs <= end_idx,
            refs + removed_count,
            jnp.where((refs > end_idx) & (refs < delete_idx), delete_idx, refs),
        )
        remapped_refs = jnp.where(valid_ref, remapped_refs, -1)
        remapped_refs = jnp.where(child[:, 0:1] == 0, -1, remapped_refs)
        child = child.at[:, 1:-1].set(remapped_refs)
        return child

    return jax.lax.cond(has_deletable, do_delete, lambda _: tree, operand=None)

def prepend_operator(tree: Array, 
                     key: PRNGKey, 
                     variable_array: Array, 
                     args: Tuple) -> Array:
    """Adds an operator node before root node.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    max_arity = tree.shape[1] - 2
    sample_key, branch_key, old_branch_key = jr.split(key, 3)

    tree_size = jnp.sum(tree[:, 0] != 0).astype(jnp.int32)
    empty_nodes = jnp.sum(tree[:, 0] == 0).astype(jnp.int32)

    # We add a new root plus (new_slots - 1) fresh depth-1 branches => requires new_slots empty rows.
    feasible = slots[operator_indices] <= empty_nodes
    probs = operator_probabilities * feasible.astype(operator_probabilities.dtype)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-12)
    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=probs)
    new_slots = slots[new_operator].astype(jnp.int32)

    old_branch = jr.randint(old_branch_key, shape=(), minval=0, maxval=jnp.maximum(new_slots, 1))

    child_cols = jnp.arange(max_arity)
    use_branch = child_cols < new_slots
    is_old_branch = child_cols == old_branch

    branch_keys = jr.split(branch_key, max_arity)
    sampled_subtrees = jax.vmap(lambda k: sample_tree(k, 1, variable_array))(branch_keys)
    sampled_sizes = jnp.sum(sampled_subtrees[:, :, 0] != 0, axis=1).astype(jnp.int32)

    branch_sizes = jnp.where(use_branch, jnp.where(is_old_branch, tree_size, sampled_sizes), 0)
    cumulative_prev = jnp.cumsum(branch_sizes) - branch_sizes
    root_idx = max_nodes - 1
    branch_roots = jnp.where(use_branch, root_idx - 1 - cumulative_prev, -1)

    empty_row = -jnp.ones(tree.shape[1]).at[0].set(0.0).at[-1].set(0.0)
    child = jnp.tile(empty_row, (max_nodes, 1))

    def place_branch(i, carry):
        current_child = carry
        size = branch_sizes[i]
        b_root = branch_roots[i]
        source = jax.lax.select(is_old_branch[i], tree, sampled_subtrees[i])
        shift = b_root - (max_nodes - 1)
        rolled = jnp.roll(source, shift, axis=0)
        rolled = rolled.at[:, 1:-1].set(jnp.where(rolled[:, 1:-1] > -1, rolled[:, 1:-1] + shift, -1))
        mask = (tree_indices <= b_root) & (tree_indices > b_root - size)
        return jax.lax.select(use_branch[i], jnp.where(mask, rolled, current_child), current_child)

    child = jax.lax.fori_loop(0, max_arity, place_branch, child)

    child = child.at[root_idx, 0].set(new_operator)
    child = child.at[root_idx, 1:-1].set(branch_roots)

    return child

def insert_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """Inserts an operator node above a random node.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Mutated tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    max_arity = tree.shape[1] - 2
    select_key, sample_key, branch_key, old_branch_key = jr.split(key, 4)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p=(tree[:, 0] > 0))

    _, _, end_idx = jax.lax.while_loop(lambda c: c[1] > 0, find_end_idx, (tree, 1, mutate_idx))
    old_subtree_size = (mutate_idx - end_idx).astype(jnp.int32)

    empty_nodes = jnp.sum(tree[:, 0] == 0).astype(jnp.int32)
    feasible = slots[operator_indices] <= empty_nodes
    probs = operator_probabilities * feasible.astype(operator_probabilities.dtype)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-12)
    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=probs)
    new_slots = slots[new_operator].astype(jnp.int32)

    old_branch = jr.randint(old_branch_key, shape=(), minval=0, maxval=jnp.maximum(new_slots, 1))

    child_cols = jnp.arange(max_arity)
    use_branch = child_cols < new_slots
    is_old_branch = child_cols == old_branch

    # Extract old subtree rows (keeps old references, remapped by global shift below).
    empty_row = -jnp.ones(tree.shape[1]).at[0].set(0.0).at[-1].set(0.0)
    old_subtree = jnp.where((tree_indices <= mutate_idx) & (tree_indices > end_idx), tree, jnp.tile(empty_row, (max_nodes, 1)))

    branch_keys = jr.split(branch_key, max_arity)
    sampled_subtrees = jax.vmap(lambda k: sample_tree(k, 1, variable_array))(branch_keys)
    sampled_sizes = jnp.sum(sampled_subtrees[:, :, 0] != 0, axis=1).astype(jnp.int32)

    branch_sizes = jnp.where(use_branch, jnp.where(is_old_branch, old_subtree_size, sampled_sizes), 0)
    total_branch_size = jnp.sum(branch_sizes)
    cumulative_prev = jnp.cumsum(branch_sizes) - branch_sizes
    branch_roots = jnp.where(use_branch, mutate_idx - 1 - cumulative_prev, -1)

    # Preserve tree parts outside replaced subtree.
    remaining_size = end_idx - jnp.sum(tree[:, 0] == 0) + 1
    child = jnp.tile(empty_row, (max_nodes, 1))
    child = jnp.where(tree_indices >= mutate_idx, tree, child)

    shift_preserved = mutate_idx - end_idx - total_branch_size - 1
    rolled_tree = jnp.roll(tree, shift_preserved, axis=0)
    child = jnp.where(
        (tree_indices < mutate_idx - total_branch_size) &
        (tree_indices >= mutate_idx - total_branch_size - remaining_size),
        rolled_tree,
        child,
    )
    child = child.at[:, 1:-1].set(
        jnp.where(
            (child[:, 1:-1] <= end_idx) & (child[:, 1:-1] > -1),
            child[:, 1:-1] + shift_preserved,
            child[:, 1:-1],
        )
    )

    # Insert branches under new operator.
    def place_branch(i, carry):
        current_child = carry
        size = branch_sizes[i]
        b_root = branch_roots[i]
        source = jax.lax.select(is_old_branch[i], old_subtree, sampled_subtrees[i])

        # Old subtree rows are currently aligned to their original root at mutate_idx,
        # while sampled subtrees are aligned to max_nodes-1.
        shift_old = b_root - mutate_idx
        shift_sampled = b_root - (max_nodes - 1)
        shift = jax.lax.select(is_old_branch[i], shift_old, shift_sampled)

        rolled = jnp.roll(source, shift, axis=0)
        rolled = rolled.at[:, 1:-1].set(jnp.where(rolled[:, 1:-1] > -1, rolled[:, 1:-1] + shift, -1))
        mask = (tree_indices <= b_root) & (tree_indices > b_root - size)
        return jax.lax.select(use_branch[i], jnp.where(mask, rolled, current_child), current_child)

    child = jax.lax.fori_loop(0, max_arity, place_branch, child)

    child = child.at[mutate_idx, 0].set(new_operator)
    child = child.at[mutate_idx, 1:-1].set(branch_roots)

    return child

def replace_tree(tree: Array, 
                 key: PRNGKey, 
                 variable_array: Array, 
                 args: Tuple) -> Array:
    """Samples a new tree.

    Parameters
    ----------
    tree : Array
        Tree to be mutated.
    key : PRNGKey
        Random key.
    variable_array : Array
        The valid variables for this tree.
    args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Array
        Sampled tree.
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, tree_indices) = args
    return sample_tree(key, max_init_depth, variable_array)

def mutate_tree(tree: Array, 
                key: PRNGKey, 
                mutate_function: int, 
                variable_array: Array, 
                partial_mutate_functions: List[Callable]) -> Array:
    """Applies a mutation to a tree.

    Parameters
    ----------
    tree : Array
        The tree to be mutated.
    key : PRNGKey
        A random key.
    mutate_function : int
        Index of the mutation function to apply.
    variable_array : Array
        Array of valid variables for the tree.
    partial_mutate_functions : List[Callable]
        List of mutation functions with pre-defined arguments.

    Returns
    -------
    Array
        The mutated tree.
    """
    mutated_tree = jax.lax.switch(mutate_function, partial_mutate_functions, tree, key, variable_array)
    return mutated_tree

def get_mutations(tree: Array, 
                  key: PRNGKey,
                  operator_indices: Array) -> int:
    """Samples a mutation function to apply to the tree.

    Parameters
    ----------
    tree : Array
        The tree to be mutated.
    key : PRNGKey
        A random key.

    Returns
    -------
    int
        Index of the mutation function.
    """
    mutation_probs = jnp.ones(len(MUTATE_FUNCTIONS))
    mutation_probs = jax.lax.select(jnp.sum(tree[:, 0] == 0) < 8, jnp.array([0., 1., 1., 1., 0.]), mutation_probs)  # Tree is too big to add more nodes
    mutation_probs = jax.lax.select(jnp.sum(tree[:, 0] != 0) <= 3, jnp.array([1., 1., 1., 0., 1.]), mutation_probs)  # Tree does not have enough operators
    mutation_probs = jax.lax.select(jnp.sum(tree[:, 0] != 0) == 1, jnp.array([1., 1., 0., 0., 1.]), mutation_probs)  # Tree does not have operators
    mutation_probs = mutation_probs.at[2].set(mutation_probs[2] * (len(operator_indices)>1))

    return jr.choice(key, jnp.arange(len(MUTATE_FUNCTIONS)), p=mutation_probs)

# Define list with possible mutation functions
MUTATE_FUNCTIONS = [add_subtree, mutate_leaf, mutate_operator, delete_operator, insert_operator]

def initialize_mutation_functions(mutate_args: Tuple) -> Callable:
    """Initializes the mutation functions with static arguments.

    Parameters
    ----------
    mutate_args : Tuple
        Miscellaneous parameters required for mutation.

    Returns
    -------
    Callable
        A jittable mutation function.
    """
    operator_indices = mutate_args[4]
    partial_mutate_functions = [partial(f, args=mutate_args) for f in MUTATE_FUNCTIONS]  # Set args as static argument in mutation functions

    def mutate_trees(trees: Array, 
                     keys: PRNGKey, 
                     reproduction_probability: float, 
                     variable_array: Array,
                     ) -> Array:
        """Applies a mutation to a batch of trees.

        Parameters
        ----------
        trees : Array
            Batch of trees to be mutated.
        keys : PRNGKey
            Array of random keys.
        reproduction_probability : float
            Probability of a tree to be mutated.
        variable_array : Array
            Array of valid variables for the tree.

        Returns
        -------
        Array
            The mutated trees.
        """
        _, mutate_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (keys[0], jnp.zeros(trees.shape[0]), reproduction_probability))
        mutate_functions = jax.vmap(get_mutations, in_axes=[0,0,None])(trees, keys, operator_indices)

        mutated_trees = jax.vmap(mutate_tree, in_axes=[0,0,0,0,None])(trees, keys, mutate_functions, variable_array, partial_mutate_functions)

        #Only keep the new trees of the mutation indices
        return jnp.where(mutate_indices[:,None,None] * jnp.ones_like(trees), mutated_trees, trees)

    return mutate_trees