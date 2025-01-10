# Kozax: Genetic Programming in JAX
Kozax introduces a general framework for evolving computer programs with genetic programming in JAX. With JAX, the computer programs can be vectorized and evaluated on parallel on CPU and GPU. Furthermore, Just-in-time compilation provides massive speedups for evolving offspring.

# Features
Kozax allows the user to:
- define custom operators
- define custom fitness functions
- use trees flexibly, ranging from symbolic regression to reinforcement learning
- evolve multiple trees simultaneously, even with different inputs
- numerically optimise constants in the computer programs

# How to use
Below is a short demo showing how you can use Kozax. First we generate data:
```python
import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(0)
key, data_key, init_key = jr.split(key, 3)
x = jr.uniform(data_key, shape=(30,), minval=-5, maxval = 5)
y = -0.1*x**3 + 0.3*x**2 + 1.5*x
```

Now we have to define a fitness function. This allows for much freedom, because you can use the computer program anyway you want to during evaluation. The fitness function should have a `__call__` method that receives a candidate, the data and a function that is necessary to evaluate the tree.
```python
class FitnessFunction:
    def __call__(self, candidate, data, tree_evaluator):
        _X, _Y = data
        pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, _X)
        return jnp.mean(jnp.square(pred-_Y))

fitness_function = FitnessFunction()
```

Now we will use genetic programming to recover the equation from the data. This requires defining the hyperparameters, initializing the population and the general loop of evaluating and evolving the population.
```python
from Kozax.genetic_programming import GeneticProgramming

#Define hyperparameters
population_size = 500
num_generations = 100

strategy = GeneticProgramming(num_generations, population_size, fitness_function)

population = strategy.initialize_population(init_key)

for g in range(num_generations):
    key, eval_key, sample_key = jr.split(key, 3)
    fitness, population = strategy.evaluate_population(population, (x[:,None], y[:,None]), eval_key)

    if g < (num_generations-1):
        population = strategy.evolve(population, fitness, sample_key)

best_fitnesses, best_solutions = strategy.get_statistics()
print(f"The best solution is {strategy.to_string(best_solutions[-1])} with a fitness of {best_fitnesses[-1]}")
```

There are additional [examples](https://github.com/sdevries0/Kozax/tree/main/examples) on how to use Kozax on more complex problems.


# Citation
If you make use of this code in your research paper, please cite:
```
@article{de2024discovering,
  title={Discovering Dynamic Symbolic Policies with Genetic Programming},
  author={de Vries, Sigur and Keemink, Sander and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2406.02765},
  year={2024}
}
```
