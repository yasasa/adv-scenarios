import torch
import time

def G(f, xp, x, t1, t2, params):
    return xp - f(t2, t1, xp, x, params)

def vjp(f, x, v):
    x = tuple([_x.requires_grad_(True) for _x in x])
    return torch.autograd.functional.vjp(f, x, v)[-1]

def J(f, x):
    x = tuple([_x.requires_grad_(True) for _x in x])
    J = torch.autograd.functional.jacobian(f, x)
    return J

def grad(f, x):
    with torch.enable_grad():
        x.requires_grad_(True)
        y = f(x)
        return torch.autograd.grad(y, x)[0]

class DiscreteAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T, f, c, x0, p, explicit, record):
#         time0 = time.time()
        x = x0
        cost = torch.zeros(1).type_as(x0)
        xs = [x0]
        print(id(p))
        for t1, t2 in zip(T[:-1], T[1:]):
            time1 = time.time()
#             print('forward: t1:{}, t2:{}'.format(t1,t2))
            xn = f(t2, t1, x, x, p)
            cost += c(xn)
            xs.append(xn)
            x = xn

#             print('forward iter takes: {}'.format(time.time()-time1))
        
#         print(f(0.4, 0.6, x, x, p))
#         grads = torch.autograd.grad(f(0.4, 0.6, x, x, p), p)
#         print(grads)
        if record:
            _save(xs)

        ctx.f = f
        ctx.c = c
        ctx.explicit = explicit
        

        ctx.save_for_backward(p.detech(), T, torch.stack(xs))

#         print('forward takes: {}'.format(time.time()-time0))

        return cost

    @staticmethod
    def backward(ctx, c):
        
        norma = True

        print('backward')
        time0 = time.time()

        p, T, xs = ctx.saved_tensors

        if ctx.explicit:
            dGxn = torch.eye(xs[-1].shape[0]).type_as(xs[-1])
        else:
            G_ = lambda _xn: G(ctx.f, _xn, xs[-2], T[-2], T[-1], p)
            dGxn = J(G_, [xs[-1]])[0]

        dC = grad(ctx.c, xs[-1])

        l_N = torch.linalg.solve(dGxn, -dC.t())
        l = l_N

        dp = torch.zeros_like(p)

        N = xs.shape[0]
        for k in range(N-2, 0, -1):
            xn = xs[k+1]
            x = xs[k]
            px = xs[k-1]

            if ctx.explicit:
                Gxn = torch.eye(x.shape[0]).type_as(x)
            else:
                G_ = lambda _xn: G(ctx.f, _xn, px, T[k-1], T[k], p)
                Gxn = J(G_, [x])[0]


            G_ = lambda _x, _p: G(ctx.f, xn, _x, T[k], T[k+1], _p)
            G1, G2 = vjp(G_, [x, p], l)

            dC = grad(ctx.c, x)
            B = -dC.t() - G1
            l_n = torch.linalg.solve(Gxn.t(), B)

            dp += G2
            l = l_n

        dp += vjp(lambda _p: G(ctx.f, xs[1], xs[0], T[1], T[2], _p), [p], l)[0]
        dp = c * dp

        
        
        return None, None, None, None, dp, None, None

def discrete_adjoint(f, c, x, T, p, explicit=True, record=False):
    return DiscreteAdjoint.apply(T, f, c, x, p, explicit, record)

