"""
Objective function optimization

In this example, Kozax is used to evolve a symbolic loss function to train a neural network.
With each candidate loss function, a neural network is trained on the task of binary classification of XOR data points.
"""

# Specify the cores to use for XLA
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Callable, Tuple
from jax import Array

from kozax.genetic_programming import GeneticProgramming

"""
We define a fitness function class that includes the network initialization, training loop and weight updates.
At every epoch, a new batch of data is sampled, and the fitness is computed as the accuracy of the trained network on a validation set.
"""

class FitnessFunction:
    """
    A class to define the fitness function for evaluating candidate loss functions.
    The fitness is computed as the accuracy of a neural network trained with the candidate loss function
    on a binary classification task (XOR data).

    Attributes:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layers in the neural network.
        output_dim (int): Dimension of the output.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        optim (optax.GradientTransformation): Optax optimizer instance.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, epochs: int, learning_rate: float):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.optim = optax.adam(learning_rate)
        self.epochs = epochs

    def __call__(self, candidate: str, data: Tuple[Array, Array, Array], tree_evaluator: Callable) -> Array:
        """
        Computes the fitness of a candidate loss function.

        Args:
            candidate: The candidate loss function (symbolic tree).
            data (tuple): A tuple containing the data keys, test keys, and network keys.
            tree_evaluator: A function to evaluate the symbolic tree.

        Returns:
            Array: The mean loss (1 - accuracy) on the validation set.
        """
        data_keys, test_keys, network_keys = data
        losses = jax.vmap(self.train, in_axes=[None, 0, 0, 0, None])(candidate, data_keys, test_keys, network_keys, tree_evaluator)
        return jnp.mean(losses)

    def get_data(self, key: jr.PRNGKey, n_samples: int = 50) -> Tuple[Array, Array]:
        """
        Generates XOR data.

        Args:
            key (jax.random.PRNGKey): Random key for data generation.
            n_samples (int): Number of samples to generate.

        Returns:
            tuple: A tuple containing the input data (x) and the target labels (y).
        """
        x = jr.uniform(key, shape=(n_samples, 2))
        y = jnp.logical_xor(x[:,0]>0.5, x[:,1]>0.5)

        return x, y[:,None]

    def loss_function(self, params: Tuple[Array, Array, Array, Array, Array, Array], x: Array, y: Array, candidate: str, tree_evaluator: Callable) -> Array:
        """
        Computes the loss with an evolved loss function for a given set of parameters and data.

        Args:
            params (tuple): The parameters of the neural network.
            x (Array): The input data.
            y (Array): The target labels.
            candidate: The candidate loss function (symbolic tree).
            tree_evaluator: A function to evaluate the symbolic tree.

        Returns:
            Array: The mean loss.
        """
        pred = self.neural_network(params, x)
        return jnp.mean(jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, jnp.concatenate([pred, y], axis=-1)))
    
    def train(self, candidate: str, data_key: jr.PRNGKey, test_key: jr.PRNGKey, network_key: jr.PRNGKey, tree_evaluator: Callable) -> Array:
        """
        Trains a neural network with a given candidate loss function.

        Args:
            candidate: The candidate loss function (symbolic tree).
            data_key (jax.random.PRNGKey): Random key for data generation during training.
            test_key (jax.random.PRNGKey): Random key for data generation during testing.
            network_key (jax.random.PRNGKey): Random key for initializing the network parameters.
            tree_evaluator: A function to evaluate the symbolic tree.

        Returns:
            Array: The validation loss (1 - accuracy).
        """
        params = self.init_network_params(network_key)

        optim_state = self.optim.init(params)

        def step(i: int, carry: Tuple[Tuple[Array, Array, Array, Array, Array, Array], optax._src.base.OptState, jr.PRNGKey]) -> Tuple[Tuple[Array, Array, Array, Array, Array, Array], optax._src.base.OptState, jr.PRNGKey]:
            params, optim_state, key = carry

            key, _key = jr.split(key)

            x_train, y_train = self.get_data(_key, n_samples=50)

            # Evaluate network parameters and compute gradients
            grads = jax.grad(self.loss_function)(params, x_train, y_train, candidate, tree_evaluator)
                
            # Update parameters
            updates, optim_state = self.optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)

            return (params, optim_state, key)

        (params, _, _) = jax.lax.fori_loop(0, self.epochs, step, (params, optim_state, data_key))

        # Evaluate parameters on test set
        x_test, y_test = self.get_data(test_key, n_samples=500)

        pred = self.neural_network(params, x_test)
        return 1 - jnp.mean(y_test==(pred>0.5)) # Return 1 - accuracy

    def neural_network(self, params: Tuple[Array, Array, Array, Array, Array, Array], x: Array) -> Array:
        """
        Defines the neural network architecture (forward pass).

        Args:
            params (tuple): The parameters of the neural network.
            x (Array): The input data.

        Returns:
            Array: The output of the neural network.
        """
        w1, b1, w2, b2, w3, b3 = params
        hidden = jnp.tanh(jnp.dot(x, w1) + b1)
        hidden = jnp.tanh(jnp.dot(hidden, w2) + b2)
        output = jnp.dot(hidden, w3) + b3
        return jax.nn.sigmoid(output)

    def init_network_params(self, key: jr.PRNGKey) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Initializes the parameters of the neural network.

        Args:
            key (jax.random.PRNGKey): Random key for parameter initialization.

        Returns:
            tuple: A tuple containing the initialized weights and biases.
        """
        key1, key2, key3 = jr.split(key, 3)
        w1 = jr.normal(key1, (self.input_dim, self.hidden_dim)) * jnp.sqrt(2.0 / self.input_dim)
        b1 = jnp.zeros(self.hidden_dim)
        w2 = jr.normal(key2, (self.hidden_dim, self.hidden_dim)) * jnp.sqrt(2.0 / self.hidden_dim)
        b2 = jnp.zeros(self.hidden_dim)
        w3 = jr.normal(key3, (self.hidden_dim, self.output_dim)) * jnp.sqrt(2.0 / self.hidden_dim)
        b3 = jnp.zeros(self.output_dim)
        return (w1, b1, w2, b2, w3, b3)

"""
To make sure the optimized loss function generalizes, a batch of neural networks are trained with different data and weight initialization.
For this purpose, a batch of keys for initialization, data sampling and validation data are generated.
"""

def generate_keys(key, batch_size=4):
    key1, key2, key3 = jr.split(key, 3)
    return jr.split(key1, batch_size), jr.split(key2, batch_size), jr.split(key3, batch_size)

"""
Here we define the hyperparameters and inputs to the genetic programming algorithm.
The inputs to the trees are the prediction and target value. 
"""

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    data_key, gp_key = jr.split(key)

    population_size = 100
    num_populations = 5
    num_generations = 15

    operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                    ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
                    ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                    ("log", lambda x: jnp.log(x + 1e-7), 1, 0.1),
                    ]

    variable_list = [["pred", "y"]]

    input_dim = 2
    hidden_dim = 16
    output_dim = 1

    fitness_function = FitnessFunction(input_dim, hidden_dim, output_dim, learning_rate=0.01, epochs=150)

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations = num_populations)

    data_keys, test_keys, network_keys = generate_keys(data_key)

    strategy.fit(gp_key, (data_keys, test_keys, network_keys), verbose=5)