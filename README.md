# Kozax: Genetic Programming in JAX
Kozax introduces a general framework for evolving computer programs with genetic programming in JAX. With JAX, the computer programs can be vectorized and evaluated on parallel on CPU and GPU. Furthermore, just-in-time compilation provides massive speedups for evolving offspring. Check out the [paper](https://arxiv.org/abs/2502.03047) introducing Kozax.

# Features
Kozax allows the user to:
- define custom operators
- define custom fitness functions
- use trees flexibly, ranging from symbolic regression to reinforcement learning
- evolve multiple trees simultaneously, even with different inputs
- numerically optimise constants in the computer programs

# How to use
You can install Kozax via pip with
```
pip install kozax
```

Below is a short demo showing how you can use kozax. First we generate data:
```python
import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(0)
key, data_key, init_key = jr.split(key, 3)
x = jr.uniform(data_key, shape=(30,), minval=-5, maxval = 5) #Inputs
y = -0.1*x**3 + 0.3*x**2 + 1.5*x #Targets
```

Now we have to define a fitness function. This allows for much freedom, because you can use the computer program anyway you want to during evaluation. The fitness function should have a `__call__` method that receives a candidate, the data and a function that is necessary to evaluate the tree.
```python
class FitnessFunction:
    def __call__(self, candidate, data, tree_evaluator):
        _X, _Y = data
        pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, _X)
        return jnp.mean(jnp.square(pred-_Y)) #Mean squared error

fitness_function = FitnessFunction()
```

Now we will use genetic programming to recover the equation from the data. This requires defining the hyperparameters, initializing the population and the general loop of evaluating and evolving the population.
```python
from kozax.genetic_programming import GeneticProgramming

#Define hyperparameters
population_size = 500
num_generations = 100

strategy = GeneticProgramming(num_generations, population_size, fitness_function)

#Sample initial population
population = strategy.initialize_population(init_key)

for g in range(num_generations):
    key, eval_key, sample_key = jr.split(key, 3)

    #Compute the fitness of the population
    fitness, population = strategy.evaluate_population(population, (x[:,None], y[:,None]), eval_key)

    if g < (num_generations-1):
        #Evolve a new population
        population = strategy.evolve(population, fitness, sample_key)

strategy.print_pareto_front()
```

There are additional [examples](https://github.com/sdevries0/kozax/tree/main/examples) on how to use kozax on more complex problems.


# Citation
If you make use of this code in your research paper, please cite:
```
@article{de2025kozax,
  title={Kozax: Flexible and Scalable Genetic Programming in JAX},
  author={de Vries, Sigur and Keemink, Sander W and van Gerven, Marcel AJ},
  journal={arXiv preprint arXiv:2502.03047},
  year={2025}
}
```
