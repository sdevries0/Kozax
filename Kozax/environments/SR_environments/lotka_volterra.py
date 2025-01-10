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
import jax.random as jrandom
from kozax.environments.SR_environments.time_series_environment_base import EnvironmentBase

class LotkaVolterra(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        n_var = 2
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.init_mu = jnp.array([10, 10])
        self.init_sd = 2

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4
        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return jrandom.uniform(key, shape = (batch_size,2), minval=5, maxval=15)
    
    def sample_init_state2(self, ys, batch_size, key):
        return ys[jrandom.choice(key, jnp.arange(ys.shape[0]), shape=(batch_size,), replace=False)]
    
    def drift(self, t, state, args):
        return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False
    

class LotkaVolterraN(EnvironmentBase):
    def __init__(self, key, process_noise, obs_noise, n_dim=1, n_obs=2):
        n_var = 2*n_dim
        n_obs = n_dim * n_obs
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.init_mu = 10 * jnp.ones(n_var)
        self.init_sd = 2

        keys = jrandom.split(key, (2+n_var,))

        # self.rate = jrandom.uniform(keys[0], shape=n_var, minval=0.2, maxval=0.8)*jnp.repeat(jnp.array([[1.0,-1.0]]), n_dim,axis=0).ravel()
        prey_rates = jrandom.uniform(keys[0], shape=n_dim, minval=0.5, maxval=1.0)
        predator_rates = jrandom.uniform(keys[1], shape=n_dim, minval=-0.5, maxval=-0.2)
        self.rate = jnp.array([[prey_rates[i], predator_rates[i]] for i in range(n_dim)]).ravel()
        interaction = jnp.zeros((n_var, n_var))
        for i in range(n_var):
            a1_key, a2_key = jrandom.split(keys[2+i])
            a1 = jrandom.uniform(a1_key, minval=0.01, maxval=0.1) * (-1 + 2*((i%2) > 0))
            interaction = interaction.at[i, (i-1)%n_var].set(a1)
            a2 = jrandom.uniform(a2_key, minval=0.0, maxval=0.1-a1) * (-1 + 2*((i%2) > 0))
            interaction = interaction.at[i, (i+1)%n_var].set(a2)

        self.interaction = interaction

        print(self.rate)
        print(self.interaction)
        
        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return jrandom.uniform(key, shape = (batch_size,self.n_var), minval=5, maxval=15)
    
    def sample_init_state2(self, ys, batch_size, key):
        return ys[jrandom.choice(key, jnp.arange(ys.shape[0]), shape=(batch_size,), replace=False)]
    
    def drift(self, t, state, args):
        # return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])
        return state * (self.rate + self.interaction@state)

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False