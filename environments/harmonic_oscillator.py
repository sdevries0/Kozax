import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from .base_environment import EnvironmentBase


class HarmonicOscillator(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        self.n_dim = 1
        self.n_var = 2
        self.n_control_inputs = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var) * jnp.array([2.0, 1.0])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q, 0], [0, 0]])
        self.R = jnp.array([[self.r]])

    def sample_init_states(self, batch_size, ts, key, dynamic_target=False, A=1.0, p_jump=0.2, interval=10):
        init_key, target_key, jump_key = jrandom.split(key, 3)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var)) @ self.P0
        base_target = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=-3.0, maxval=3.0)

        if not dynamic_target:
            targets = jnp.repeat(base_target[:, None, :], len(ts), axis=1)
        else:
            jump_key1, jump_key2 = jrandom.split(jump_key, 2)
            jumps = jrandom.bernoulli(jump_key1, p_jump, shape=(batch_size, len(ts))) * A * (
                jrandom.choice(jump_key2, jnp.array([-1.0, 1.0]), shape=(batch_size, len(ts)))
            )
            mask = jnp.arange(len(ts)) % interval == 0
            targets = jax.vmap(lambda base, jump: base + jnp.cumsum(jump * mask))(base_target, jumps)
            targets = targets[:, :, None]
        return x0, targets

    def sample_params(self, batch_size, mode, ts, key):
        omega_key, zeta_key, args_key = jrandom.split(key, 3)
        if mode == "constant":
            omegas = jnp.ones((batch_size, ts.shape[0]))
            zetas = jnp.zeros((batch_size, ts.shape[0]))
        elif mode == "different":
            omegas = jrandom.uniform(omega_key, (batch_size,), minval=0.0, maxval=2.0)[:, None] * jnp.ones((batch_size, ts.shape[0]))
            zetas = jrandom.uniform(zeta_key, (batch_size,), minval=0.0, maxval=1.5)[:, None] * jnp.ones((batch_size, ts.shape[0]))
        elif mode == "changing":
            decay_factors = jrandom.uniform(args_key, (batch_size, 2), minval=0.98, maxval=1.02)
            init_omegas = jrandom.uniform(omega_key, (batch_size,), minval=0.5, maxval=1.5)
            init_zetas = jrandom.uniform(zeta_key, (batch_size,), minval=0.0, maxval=1.0)
            omegas = jax.vmap(lambda o, d, t: o * (d ** t), in_axes=[0, 0, None])(init_omegas, decay_factors[:, 0], ts)
            zetas = jax.vmap(lambda z, d, t: z * (d ** t), in_axes=[0, 0, None])(init_zetas, decay_factors[:, 1], ts)
        return omegas, zetas

    def initialize_parameters(self, params, ts):
        omega, zeta = params
        A = jax.vmap(lambda o, z: jnp.array([[0, 1], [-o, -z]]))(omega, zeta)
        self.A = diffrax.LinearInterpolation(ts, A)
        self.b = jnp.array([[0.0, 1.0]]).T
        self.G = jnp.array([[0, 0], [0, 1]])
        self.V = self.process_noise*self.G
        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        return self.A.evaluate(t) @ state + self.b @ args

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        target = jnp.squeeze(target)
        x_d = jnp.stack([target, jnp.zeros_like(target)], axis=1)
        u_d = jax.vmap(lambda t, xd: -jnp.linalg.pinv(self.b) @ self.A.evaluate(t) @ xd)(ts, x_d)
        costs = jax.vmap(
            lambda _state, _u, _xd, _ud: (_state - _xd) @ self.Q @ (_state - _xd) + (_u - _ud) @ self.R @ (_u - _ud)
        )(state, control, x_d, u_d)
        return jnp.mean(costs)

    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) + jnp.isnan(y)), -1.0, 1.0)
