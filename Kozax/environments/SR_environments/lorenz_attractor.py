import jax
import jax.numpy as jnp
import jax.random as jrandom
from Kozax.environments.SR_environments.time_series_environment_base import EnvironmentBase

class LorenzAttractor(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=3):
        n_var = 3
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.init_mu = jnp.array([1,1,1])
        self.init_sd = 1

        self.sigma = 10
        self.rho = 28
        self.beta = 8/3
        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return self.init_mu + self.init_sd*jrandom.normal(key, shape=(batch_size,3))
        # return jnp.ones((batch_size, 3))
    
    def drift(self, t, state, args):
        return jnp.array([self.sigma*(state[1]-state[0]), state[0]*(self.rho-state[2])-state[1], state[0]*state[1]-self.beta*state[2]])

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False
    
class Lorenz96(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_dim = 3, n_obs=3):
        n_var = n_dim
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.F = 8
        self.init_mu = jnp.ones(self.n_var)*self.F*0
        self.init_sd = 1
        

        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return self.init_mu + self.init_sd*jrandom.normal(key, shape=(batch_size,self.n_var))
        # return jnp.ones((batch_size, 3))
    
    def drift(self, t, state, args):
        f = lambda x_cur, x_next, x_prev1, x_prev2: (x_next - x_prev2) * x_prev1 - x_cur + self.F
        return jax.vmap(f)(state, jnp.roll(state, -1), jnp.roll(state, 1), jnp.roll(state, 2))

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False