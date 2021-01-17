import scipy
import numpy as np

from .optimizer import CubeOptimizer

import time

class Transcription(CubeOptimizer):
    def __init__(self,
                 dynamics,
                 sensor,
                 policy,
                 step_cost,
                 save_path,
                 state_dim=3,
                 action_dim=1,
                 cube_dim=4,
                 integration_timesteps=200):
        super(Transcription,
              self).__init__(dynamics, sensor, policy, step_cost, save_path,
                             state_dim, action_dim, cube_dim)
        self._T = integration_timesteps

    def dynamics_constraint_step(self, x, x_n, u):
        x_pred = self._dynamics(x, u)
        h = x_n - x_pred
        return h

    def policy_constraint_step(self, x, u, m):
        u_pred = self._policy(self._sensor(x, m))
        h = u - u_pred
        return h

    def dynamics_constraint_step_jac(self, x, x_n, u, t):
        Js = np.zeros((self._T, self._state_dim, self._state_dim))
        Ju = np.zeros((self._T, self._state_dim, self._action_dim))
        Jx = np.zeros((self._state_dim, self._cube_dim))

        I = np.eye(self._state_dim)
        x_pred = self._dynamics(x, u)

        Js[t] = I

        if t > 0:
            Js[t-1] = -I.dot(self._dynamics.Js(x, u))

        Ju[t] = -I.dot(self._dynamics.Ju(x, u))

        Js = np.reshape(np.transpose(Js, (1, 0, 2)), (self._state_dim, -1))
        Ju = np.reshape(np.transpose(Ju, (1, 0, 2)), (self._state_dim, -1))

        J_step = np.concatenate([Js, Ju, Jx], axis=1)

        return J_step

    def policy_constraint_step_jac(self, x, u, m, t):
        Js = np.zeros((self._T, self._action_dim, self._state_dim))
        Ju = np.zeros((self._T, self._action_dim, self._action_dim))

        o = self._sensor(x, m)
        Ju_ = self._policy.J(o)

        Ju[t] = np.eye(self._action_dim)
        if t > 0:
            Js[t-1] = -Ju_.dot(self._sensor.Js(x, m))

        Jx = -Ju_.dot(self._sensor.Jm(x, m))

        Js = np.reshape(np.transpose(Js, (1, 0, 2)), (self._action_dim, -1))
        Ju = np.reshape(np.transpose(Ju, (1, 0, 2)), (self._action_dim, -1))

        J_step = np.concatenate([Js, Ju, Jx], axis=1)
        return J_step

    def constraints(self, X, x0):
        xs = np.split(X[:self._state_dim * self._T], self._T)
        us = np.split(X[self._state_dim * self._T:self._T * (
            self._state_dim + self._action_dim)], self._T)
        m = X[-self._cube_dim:]

        h_x = np.zeros((self._T, self._state_dim))
        h_x[0] = self.dynamics_constraint_step(x0, xs[0], us[0])

        h_u = np.zeros((self._T, self._action_dim))
        h_u[0] = self.policy_constraint_step(x0, us[0], m)

        for t in range(self._T - 1):
            x = xs[t]
            x_n = xs[t + 1]
            u = us[t + 1]
            h_x[t + 1] = self.dynamics_constraint_step(x, x_n, u)
            h_u[t + 1] = self.policy_constraint_step(x, u, m)

        return np.concatenate([h_x.flatten(), h_u.flatten()])

    def constraints_jac(self, X, x0):
        xs = np.split(X[:self._state_dim * self._T], self._T)
        us = np.split(X[self._state_dim * self._T:self._T * (
            self._state_dim + self._action_dim)], self._T)
        m = X[-self._cube_dim:]

        J_x = np.zeros((self._T, self._state_dim, X.shape[0]))
        J_x[0] = self.dynamics_constraint_step_jac(x0, xs[0], us[0], 0)

        J_u = np.zeros((self._T, self._action_dim, X.shape[0]))
        J_u[0] = self.policy_constraint_step_jac(x0, us[0], m, 0)

        for t in range(self._T - 1):
            x = xs[t]
            x_n = xs[t + 1]
            u = us[t + 1]
            J_x[t + 1] = self.dynamics_constraint_step_jac(x, x_n, u, t+1)
            J_u[t + 1] = self.policy_constraint_step_jac(x, u, m, t+1)

        J_x = J_x.reshape((-1, X.shape[0]))
        J_u = J_u.reshape((-1, X.shape[0]))
        return np.concatenate([J_x, J_u], axis=0)

    def cost(self, X):
        xs = np.split(X[:self._state_dim * self._T], self._T)
        us = np.split(X[self._state_dim * self._T:self._T * (
            self._state_dim + self._action_dim)], self._T)

        ls = [self._step_cost.cost(_x, _u) for _x, _u in zip(xs, us)]
        c = np.sum(ls)
        return c

    def cost_jac(self, X):
        xs = np.split(X[:self._state_dim * self._T], self._T)
        us = np.split(X[self._state_dim * self._T:self._T * (
            self._state_dim + self._action_dim)], self._T)

        Jx = np.concatenate([self._step_cost.Jx(x, u).flatten() for x, u in zip(xs, us)])
        Ju = np.concatenate([self._step_cost.Ju(x, u).flatten() for x, u in zip(xs, us)])

        J = np.concatenate([
            Jx.flatten(),
            Ju.flatten(),
            np.zeros(self._cube_dim)
        ])
        return J

    def cube_constraint(self, X):
        x = X[-self._cube_dim:]
        h = np.array([-x[0] + 134, x[0] - 90, -x[1] + 144, x[1] - 114,  -x[2] + 2.5 , x[2] + 0.5])
        return h

    def cube_jac(self, X):
        xdim = self._cube_dim
        J = np.zeros((6, X.shape[-1]))
        J[0, -xdim] = -1.
        J[1, -xdim] = 1.
        J[2, -xdim+1] = -1.
        J[3, -xdim+1] = 1.
        J[4, -xdim+2] = -1.
        J[5, -xdim+2] = 1.
        return J


    def sample_trajectory(self, c, x0):
        xs = []
        us = []

        x = x0
        for i in range(self._T):
            u = self._policy(self._sensor(x, c))
            x = self._dynamics(x, u)
            xs.append(x)
            us.append(u)

        X = np.concatenate(xs)
        U = np.concatenate(us)
        XUM = np.concatenate([X, U, c])
        return XUM

    def run(self, c0, x0):
        x_init = self.sample_trajectory(c0, x0)

        constraints = {
            "type": "eq",
            "fun": lambda X: self.constraints(X, x0),
            "jac": lambda X: self.constraints_jac(X, x0)
        }

        cube_constraints = {
            "type": "ineq",
            "fun": self.cube_constraint,
            "jac": self.cube_jac
        }


        ret = scipy.optimize.minimize(
               self.cost,
               x_init,
               jac=self.cost_jac,
               callback=self.callback,
               constraints=[constraints, cube_constraints],
               options={"disp":6})

    def callback(self, x, *args):
        print(x[-self._cube_dim:], self.cost(x))

    def check_grad(self):
        def finite_diff(f, x0, eps=5e-5, fdim=self._T*self._state_dim + self._T*self._action_dim):
            gfd = np.zeros((fdim, x0.shape[0]))
            for i in range(x0.shape[0]):
                h = np.eye(x0.shape[0])[i] * eps

                f1 = f(x0 + h).flatten()
                f2 = f(x0 - h).flatten()
                f3 = f(x0 + 2*h).flatten()
                f4 = f(x0 - 2*h).flatten()
                gfd[:, i] = (-f3 + 8*f1 - 8*f2 + f4) / (12 * eps)

            return gfd

        def test_grad(f, gf, x0, eps=1e-5):
            t = time.time()
            gfv = gf(x0)
            print(time.time() - t)
            gfd = finite_diff(f, x0, eps)

            gfvb = gfv / np.linalg.norm(gfv)
            gfdb = gfd / np.linalg.norm(gfd)
            print(gfd, "Finite Difference")
            print(gfv, "Other")

            print(gfv - gfd)

        x0 = np.array([120, 129, np.pi])
        c0 = np.array([120, 134, 1., 0.])
        XUM = self.sample_trajectory(c0, x0)


        f = lambda x: self.constraints(x, x0)
        gf = lambda x: self.constraints_jac(x, x0)

        test_grad(f, gf, XUM)
