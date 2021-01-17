import torch

from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

def normalize_features(x, y):
    xmin = x.min(0)[0]
    xmax = x.max(0)[0]
    x = (x - xmin) / (xmax - xmin)
    y = (y - y.mean(dim=0)) / y.std(0)

    return x, y, xmax, xmin


class BOTorchWrapper:
    def __init__(self, objective, minimum_bound, maximum_bound, acq_restarts=10, acq_samples=512, acq_opt_itr=500):
        self.bounds = torch.stack([minimum_bound, maximum_bound])
        self.x_dim = minimum_bound.shape[0]
        self._f = objective
        self.acq_bounds = torch.stack([torch.zeros_like(self.bounds[0]), torch.ones_like(self.bounds[0])])
        self.acq_samples = acq_samples
        self.acq_opt_itr = acq_opt_itr
        self.acq_restarts = acq_restarts

    # Wrap to minimize instead of maximize
    def f(self, x):
        return -self._f(x)

    def _init_points(self, n):
        x = (self.bounds[1] - self.bounds[0]) * torch.rand(n, self.x_dim).type_as(self.bounds[0]) + self.bounds[0]
        y = torch.stack([self.f(_x) for _x in x])
        best_y, best_idx = y.max(0)
        best_x = x[best_idx]
        return x, y, best_x, best_y

    def _init_model(self, x, y, load=None):
        x, y, xmax, xmin = normalize_features(x, y)
        model = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        return model, mll, xmax, xmin

    def _find_and_eval(self, acq, xmax, xmin, restarts, q=1):
        x, _ = optimize_acqf(acq_function=acq,
                bounds=self.acq_bounds,
                num_restarts=restarts,
                q=q,
                raw_samples=self.acq_samples,
                options={"batch_limit": 5, "maxiter": self.acq_opt_itr})

        x_ = x * (xmax - xmin) + xmin
        y = self.f(x_.squeeze())
        return x_, y

    def _on_itr(self, k, x, y, xs, ys, best_y, best_x, cb, verbose):
        if verbose:
            print("[Iteration {:d}] Function Evaluation {:.4f}, Best So Far {:.4f}".format(k, y.item(), best_y.item()))
        if cb:
            cb(k, x, y, xs, ys, best_y, best_x)

    def run(self, init_points, iterations, callback=None, verbose=False):
        xs, ys, best_x, best_y = self._init_points(init_points)
        restarts = min(init_points, self.acq_restarts)
        for itr in range(iterations):
            model, mll, xmax, xmin = self._init_model(xs, ys)
            fit_gpytorch_model(mll)
            acq = ExpectedImprovement(model=model, best_f = ys.max())
            nx, ny = self._find_and_eval(acq, xmax, xmin, restarts)
            xs = torch.cat([xs, nx], dim=0)
            ys = torch.cat([ys, ny.unsqueeze(0)], dim=0)

            if best_y < ny:
                best_y = ny
                best_x = nx

            self._on_itr(itr, nx, ny, xs, ys, best_y, best_x, callback, verbose)

