import torch

from typing import Union, Iterable, Callable
#from botorch.models import SingleTaskGP
#
#from botorch.fit import fit_gpytorch_mll
#from gpytorch.mlls import ExactMarginalLogLikelihood
#
#from botorch.optim import optimize_acqf
#from botorch.acquisition import UpperConfidenceBound

ParamType = Union[Iterable, torch.Tensor]

def apply_op_to_params(op : Callable, param: ParamType):
    if not isinstance(param, torch.Tensor):
        for p in param:
            if type(p) is dict:
                op(p["params"])
            else:
                op(p)
    else:
        op(p)

class Optimizer:
    def __init__(self, params : ParamType):
        self.params = params

    def step(self, loss : torch.Tensor, *args, **kwargs):
        loss = self.step_(loss, *args, **kwargs)
        return loss, self.params.detach().clone()

    def step_(self, loss : torch.Tensor, *args, **kwargs):
        pass

    def reset():
        pass

class GradientOptimizer(Optimizer):
    def __init__(self, params : ParamType, clip_grad=0.):
        super().__init__(params)
        self.clip_grad = clip_grad

    def _zero_grad(self):
        pass

    def _step_opt(self):
        pass

    def step_(self, loss : torch.Tensor):
        assert(loss.requires_grad)
        self._zero_grad()
        loss.backward()
        if self.clip_grad > 0.:
            apply_op_to_params(lambda p: torch.nn.utils.clip_grad_norm_(p, self.clip_grad))
        self._step_opt()
        return loss

class PenaltyOptimizer(GradientOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step_(self, loss : torch.Tensor, penalty : torch.Tensor):
        return super().step_(loss + penalty)

class TorchOptim(PenaltyOptimizer):
    optim : torch.optim.Optimizer

    def __init__(self, params, clip_grad, *args, **kwargs):
        super().__init__(params, clip_grad)

    def _zero_grad(self):
        return self.optim.zero_grad()

    def _step_opt(self):
        return self.optim.step()


class RandomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, bounds, *args, **kwargs):
        self.bounds = bounds
        self.params = params

    def zero_grad(self):
        pass

    def load_state_dict(self):
        pass

    def step(self):
        for param, bound in zip(self.params, self.bounds):
            mu = (bound["max"] + bound["min"]) / 2.
            std = (bound["max"] - bound["min"]) / 2.
            param['params'].data = torch.randn_like(param['params'].data) * std + mu

    def add_param_group(self, *args, **kwargs):
        pass

    @property
    def state_dict(self):
        return {}

class BOOptimizer:
    def __init__(self, params, warmup_steps, param_min, param_max, *args, **kwargs):
        assert(len(params)==1)
        self.min = param_min
        self.max = param_max
        self.params = params[0]

        if self.min.dim() == 0:
            self.min = self.min.expand_as(self.params)

        if self.max.dim() == 0:
            self.max = self.max.expand_as(self.params)


        self.xs = []
        self.ys = []

        self.warmup =  warmup_steps

    def zero_grad(self):
        pass

    def load_state_dict(self):
        pass

    def step(self, eval_fn):
        if self.warmup > 0:
            q = torch.rand_like(self.params) * (self.max - self.min) + self.min
            self.xs.append(q)
            loss = eval_fn(q)
            self.ys.append(loss)
            self.warmup -= 1
        else:
            print(self.xs, self.ys)
            inputs = torch.stack(self.xs)
            outputs = -torch.stack(self.ys).view(-1, 1) # this maximizes
            print(outputs.shape)
            gp = SingleTaskGP(inputs, outputs)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            ucb = UpperConfidenceBound(gp, beta=0.1)
            bounds = torch.stack([self.min, self.max])
            q, v = optimize_acqf(ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

            self.xs.append(q)
            loss = eval_fn(q)
            self.ys.append(loss)


        self.params = q
        return q, loss


    def add_param_group(self, *args, **kwargs):
        pass

    @property
    def state_dict(self):
        return {}
        
class CEMOptimizer:
    def __init__(self, params, bounds, population=10, elite_frac=0.4, *args, **kwargs):
        self.params = params
        self.bounds = bounds
        
        self.mu = [(a["max"] + a["min"]) / 2 for a in self.bounds]
        self.std = [(a["max"] + a["min"]) / 2 for a in self.bounds] 

        self.param_count = len(params)
        self.population = population
        self.elite_frac = elite_frac
        self.n_elite = max(int(self.elite_frac * self.population), 1)

        self.shapes = [m.shape[-1] for m in self.mu]

    def zero_grad(self):
        pass

    def load_state_dict(self):
        pass
    
    def step(self, eval_fn):
        samples = [std * torch.randn(self.population, p["params"].shape[-1]).type_as(p["params"]) + mu for p, mu, std in zip(self.params, self.mu, self.std)]
        
        params = torch.cat(samples, dim=-1)
        losses = eval_fn(params)
        
        best_losses, elite_idxs = torch.topk(losses, k=self.n_elite, dim=0, largest=False)
        
        best = params[elite_idxs, ...]
        best_param_wise = best.split(self.shapes, dim=-1)
        
        self.mu = []
        self.std = []
        for i, param in enumerate(self.params):
            mu = best_param_wise[i].mean(dim=0)
            std = best_param_wise[i].std(dim=0)
            param["params"].data = best_param_wise[i][0]
            self.mu.append(mu)
            self.std.append(std)
        loss = losses[0] 
        return best[0], loss

    def add_param_group(self, *args, **kwargs):
        pass

    @property
    def state_dict(self):
        return {}

