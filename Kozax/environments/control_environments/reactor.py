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

import jax.numpy as jnp
import jax
import jax.random as jrandom
import diffrax

from kozax.environments.control_environments.control_environment_base import EnvironmentBase

class StirredTankReactor(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs = 3, n_targets = 1, max_control = jnp.array([300]), external_f=None):
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.n_var = 3
        self.n_control = 1
        self.n_dim = 1
        self.n_targets = n_targets
        self.init_lower_bounds = jnp.array([275, 350, 0.5])
        self.init_upper_bounds = jnp.array([300, 375, 1.0])
        self.max_control = max_control
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs)

        self.Q = jnp.array([[0,0,0],[0,0.01,0],[0,0,0]])
        self.r = jnp.array([[0.0001]])
        if external_f is None:
            self.external_f = lambda t: 0.0
        else:
            self.external_f = external_f

    def initialize_parameters(self, params, ts):
        Vol, Cp, dHr, UA, q, Tf, Tcf, Volc = params
        self.Ea  = 72750     # activation energy J/gmol
        self.R   = 8.314     # gas constant J/gmol/K
        self.k0  = 7.2e10    # Arrhenius rate constant 1/min
        self.Vol = Vol       # Volume [L]
        self.Cp  = Cp        # Heat capacity [J/g/K]
        self.dHr = dHr       # Enthalpy of reaction [J/mol]
        self.UA  = UA        # Heat transfer [J/min/K]
        self.q = q           # Flowrate [L/min]
        self.Cf = 1.0        # Inlet feed concentration [mol/L]
        self.Tf  = diffrax.LinearInterpolation(ts, Tf)        # Inlet feed temperature [K]
        self.Tcf = Tcf       # Coolant feed temperature [K]
        self.Volc = Volc       # Cooling jacket volume

        self.k = lambda T: self.k0*jnp.exp(-self.Ea/self.R/T)

        self.G = jnp.eye(self.n_var)*jnp.array([6, 6, 0.05])
        # self.V = self.process_noise*self.G
        self.process_noise_ts = diffrax.LinearInterpolation(ts, jnp.linspace(self.process_noise[0], self.process_noise[1], ts.shape[0]))

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([15,15,0.1])[:self.n_obs])

        self.max_control_ts = diffrax.LinearInterpolation(ts, jnp.hstack([mc*jnp.ones(int(ts.shape[0]//self.max_control.shape[0])) for mc in self.max_control]))
        self.external_influence = diffrax.LinearInterpolation(ts, jax.vmap(self.external_f)(ts))

    def sample_param_change(self, key, batch_size, ts, low, high):
        init_key, decay_key = jrandom.split(key)
        decay_factors = jrandom.uniform(decay_key, shape=(batch_size,), minval=1.01, maxval=1.02)
        print(decay_factors)
        init_values = jrandom.uniform(init_key, shape=(batch_size,), minval=low, maxval=high)
        values = jax.vmap(lambda v, d, t: v*(d**t), in_axes=[0, 0, None])(init_values, decay_factors, ts)
        return values

    def sample_params(self, batch_size, mode, ts, key):
        if mode=="Constant":
            Vol = 100*jnp.ones(batch_size)
            Cp = 239*jnp.ones(batch_size)
            dHr = -5.0e4*jnp.ones(batch_size)
            UA = 5.0e4*jnp.ones(batch_size)
            q = 100*jnp.ones(batch_size)
            Tf = 300*jnp.ones((batch_size, ts.shape[0]))
            Tcf = 300*jnp.ones(batch_size)
            Volc = 20.0*jnp.ones(batch_size)
        elif mode=="Different":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0],(batch_size,),minval=75,maxval=150)
            Cp = jrandom.uniform(keys[1],(batch_size,),minval=200,maxval=350)
            dHr = jrandom.uniform(keys[2],(batch_size,),minval=-55000,maxval=-45000)
            UA = jrandom.uniform(keys[3],(batch_size,),minval=25000,maxval=75000)
            q = jrandom.uniform(keys[4],(batch_size,),minval=75,maxval=125)
            Tf = jnp.repeat(jrandom.uniform(keys[5],(batch_size,),minval=300,maxval=350)[:,None], ts.shape[0], axis=1)
            Tcf = jrandom.uniform(keys[6],(batch_size,),minval=250,maxval=300)
            Volc = jrandom.uniform(keys[7],(batch_size,),minval=10,maxval=30)
        elif mode == "Changing":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0],(batch_size,),minval=75,maxval=150)
            Cp = jrandom.uniform(keys[1],(batch_size,),minval=200,maxval=350)
            dHr = jrandom.uniform(keys[2],(batch_size,),minval=-55000,maxval=-45000)
            UA = jrandom.uniform(keys[3],(batch_size,),minval=25000,maxval=75000)

            q = jrandom.uniform(keys[4],(batch_size,),minval=75,maxval=125)
            # Tf = jrandom.uniform(keys[5],(batch_size,),minval=300,maxval=350)
            Tf = self.sample_param_change(keys[5], batch_size, ts, 300, 350)
            Tcf = jrandom.uniform(keys[6],(batch_size,),minval=250,maxval=300)

            Volc = jrandom.uniform(keys[7],(batch_size,),minval=10,maxval=30)
        return (Vol, Cp, dHr, UA, q, Tf, Tcf, Volc)

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= self.init_lower_bounds, maxval= self.init_upper_bounds)
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=400, maxval=480)
        return x0, targets
    
    def f_obs(self, key, t_x):
        _, out = super().f_obs(key, t_x)
        # out = jnp.array([out[0], out[1], jnp.clip(out[2], 0, 1)])[:self.n_obs]
        return key, out
    
    def drift(self, t, state, args):
        Tc, T, c = state
        control = jnp.squeeze(args)
        # control = jnp.clip(control, 0, self.max_control_ts.evaluate(t))
        control = jnp.clip(control, 0, 300)
        # state = jnp.array([state[0], state[1], jnp.clip(state[2], 0, 1)])

        # print(Tc, T, c)

        dc = (self.q/self.Vol)*(self.Cf - c) - self.k(T)*c
        dT = (self.q/self.Vol)*(self.Tf.evaluate(t) - T) + (-self.dHr/self.Cp)*self.k(T)*c + (self.UA/self.Vol/self.Cp)*(Tc - T) + self.external_influence.evaluate(t)
        dTc = (control/self.Volc)*(self.Tcf - Tc) + (self.UA/self.Volc/self.Cp)*(T - Tc)

        # print(dc, dT, dTc)
        return jnp.array([dTc, dT, dc])

    def diffusion(self, t, state, args):
        return self.process_noise_ts.evaluate(t) * self.G

    def fitness_function(self, state, control, targets, ts):
        x_d = jax.vmap(lambda tar: jnp.array([0,tar,0]))(targets)
        costs = jax.vmap(lambda _state, _u, _x_d: (_state-_x_d).T @ self.Q @ (_state-_x_d) + (_u)@self.r@(_u))(state, control, x_d)
        return jnp.sum(costs)

    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) +jnp.isnan(y)), -1.0, 1.0)
    
    