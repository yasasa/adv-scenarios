import scipy
import numpy as np
import ipopt
from ipopt import minimize_ipopt

from .optimizer import CubeOptimizer

import time

class Shooting(CubeOptimizer):
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
        super(Shooting,
              self).__init__(dynamics, sensor, policy, step_cost, save_path,
                             state_dim, action_dim, cube_dim)
        self._T = integration_timesteps

    def _step(self, c, x0, grad=False):
        x = x0

        total_cost = 0
        J = np.zeros((1, self._cube_dim))
        Js_M = np.zeros((self._state_dim, self._cube_dim))

        for i in range(self._T):
            o = self._sensor(x, c)
            u = self._policy(o)
            xn = self._dynamics(x, u)

            step_cost = self._step_cost.cost(xn, u)

            if grad:
                Jc = self._step_cost.Jx(xn, u)
                Jo_M = self._sensor.Jm(x, c)
                Jo_s = self._sensor.Js(x, c)

                Ju_o = self._policy.J(o)
                Ju_M = Ju_o.dot(Jo_M + Jo_s.dot(Js_M))

                Jsn_u = self._dynamics.Ju(x, u)
                Jsn_s = self._dynamics.Js(x, u)

                Jsn_M = Jsn_s.dot(Js_M) + Jsn_u.dot(Ju_M)
                J_ = Jc.dot(Jsn_M)

                J += J_

                Js_M = Jsn_M

            total_cost += step_cost
            x = xn

        if grad:
            return -total_cost, -J.flatten()

        return -total_cost

    def _finite_diff(self, f, x0, eps=1e-5, fdim=1):
        gfd = np.zeros((fdim, x0.shape[0]))
        for i in range(x0.shape[0]):
            h = np.eye(x0.shape[0])[i] * eps

            f1 = f(x0 + h).flatten()
            f2 = f(x0 - h).flatten()
            f3 = f(x0 + 2*h).flatten()
            f4 = f(x0 - 2*h).flatten()
            gfd[:, i] = (-f3 + 8*f1 - 8*f2 + f4) / (12 * eps)

        return gfd

    def cube_constraint(self, x):
        print(x)
        num_cubes = x.shape[0] // 2
        h = np.concatenate([[-x[2*i] + 134, x[2*i] - 95, -x[2*i+1] + 128, x[2*i + 1] - 120] for i in range(num_cubes)])
        return h

    def cube_jac(self, c):
        num_cubes = self._cube_dim // 2
        J = np.zeros((num_cubes*4, num_cubes*2))
        for i in range(num_cubes):
            J[4*i, 2*i] = -1.
            J[4*i+1, 2*i] = 1.
            J[4*i+2, 2*i+1] = -1.
            J[4*i+3, 2*i+1] = 1.

        return J


    def cost(self, c, x0):
        cost = self._step(c, x0, grad=False)
        return cost

    def cost_jac(self, c, x0):
        return self._step(c, x0, grad=True)[1]

    def save_traj(self, c, x0, filename):
        x = x0
        total_cost = 0
        xs = [x0]

        for i in range(self._T):
            o = self._sensor(x, c)
            u = self._policy(o)
            xn = self._dynamics(x, u)
            xs.append(xn)
            step_cost = self._step_cost.cost(xn, u)
            total_cost += step_cost
            x = xn
        np.save("{}.np".format(filename), np.stack(xs))
        print(total_cost)
        return xs



    def run(self, c0, x0):
        eps = 1e-4

        f = lambda c: self.cost(c, x0)
        j = lambda c: self.cost_jac(c, x0)
        ff = lambda c: self._step(c, x0, grad=True)
        fg = lambda c: scipy.optimize.approx_fprime(c, f, 1e-5)

        def objective(x):
            c = f(x)
            print(c, x)
            return c

        problem = type('', (), {})()
        problem.objective = objective
        problem.gradient = j
        problem.constraints = lambda x: np.array([])
        problem.jacobian = lambda x: np.array([])

        lb = [90, 120] * (self._cube_dim // 2)
        ub = [134, 127] * (self._cube_dim // 2)

        nlp = ipopt.problem(self._cube_dim,
                            0,
                            problem_obj = problem,
                            lb=lb,
                            ub=ub)
        nlp.addOption('tol', 1e-5)
        nlp.addOption('print_level', 6)
        nlp.solve(c0)

        #for i in range(50000):
        #    cost, g = ff(c)
        #    print(c, cost, g)
        #    c = c - g*0.0001

    def callback(self, x, cost, *args):
        print(x, cost(x))

    def check_grad(self):

        def test_grad(f, gf, x0, eps=1e-5):
            t = time.time()
            gfv = gf(x0)
            print(time.time() - t)
            gfd = self._finite_diff(f, x0, eps)

            print(gfd, "Finite Difference")
            print(gfv, "Other")

            print(gfv - gfd)

        x0 = np.array([120, 129, np.pi])
        c0 = np.array([120, 134, 1., 0.])


        f = lambda x: self.cost(x, x0)
        gf = lambda x: self.cost_jac(x, x0)

        test_grad(f, gf, c0)
