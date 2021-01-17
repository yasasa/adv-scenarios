import torch
import time

def G(f, xp, x, t1, t2, params, i):
    return xp - f(i, t2, t1, xp, x, params)

def vjp(f, x, v):
    x = tuple([_x.requires_grad_(True) for _x in x])
    return torch.autograd.functional.vjp(f, x, v)[-1]

@torch.no_grad()
def fd_vjp(f, x, v, eps=1e-4):
    I = torch.eye(x.shape[0]).type_as(x)
    y0 = f(x)
    X = x + eps * I
    Y = torch.stack([f(_x) for _x in X])
    gradY = (Y - y0) / eps
    vj = (v * gradY).sum(-1)
    return vj

def J(f, x):
    x = tuple([_x.requires_grad_(True) for _x in x])
    J = torch.autograd.functional.jacobian(f, x)
    return J

def grad(f, x):
        
    with torch.enable_grad():
        x.requires_grad_(True)
        y = f(x)
       # y = y.expand(x.shape[0], -1)
        dy = torch.autograd.grad(y, x, grad_outputs=torch.ones(x.shape[0]).type_as(x))[0]
        return dy

class DiscreteAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T, f, c, x0, p, explicit, record, ret_traj):
#         time0 = time.time()
        x = x0
        cost = torch.zeros(x0.shape[0]).type_as(x0)
        xs = [x0]
        i = 0
        for t1, t2 in zip(T[:-1], T[1:]):
            time1 = time.time()
            xn = f(i, t2, t1, x, x, p)
            cost += c(xn)
            xs.append(xn.detach())
            x = xn
            i+=1
        ctx.f = f
        ctx.c = c
        ctx.explicit = explicit
        xs = torch.stack(xs, dim=1) # [B, T, xdim]
        ctx.save_for_backward(p.detach(), T, xs)

        if ret_traj:
            return cost, xs
        else:
            return cost
            

    @staticmethod
    def backward(ctx, c, *args): #Dummy args here because variable return in the forward.
        norma = True
        time0 = time.time()
        p, T, xs = ctx.saved_tensors
        if ctx.explicit:
            dGxn = torch.eye(xs.shape[-1]).type_as(xs).expand(xs.shape[0], -1, -1) #[B, xdim, xdim]
        else:
            G_ = lambda _xn: G(ctx.f, _xn, xs[-2], T[-2], T[-1], p, len(T) - 2)
            dGxn = J(G_, [xs[-1]])[0]

        dC = grad(ctx.c, xs[:, -1]) # [B, xdim]
        if ctx.explicit:
            l_N = -dC
        else:
            l_N = torch.linalg.solve(dGxn.mT, -dC) # [B, xdim]
        l = l_N

        dp = torch.zeros(*p.shape).type_as(p)

        N = xs.shape[1]
        for k in range(N-2, 0, -1):
            xn = xs[:, k+1]
            x = xs[:, k]
            px = xs[:, k-1]

            if ctx.explicit:
                Gxn = dGxn
            else:
                G_ = lambda _xn: G(ctx.f, _xn, px, T[k-1], T[k], p, k-1)
                Gxn = J(G_, [x])[0]


            G_ = lambda _x, _p: G(ctx.f, xn, _x, T[k], T[k+1], _p, k)
            G1, G2 = vjp(G_, [x, p], l)

            dC = grad(ctx.c, x)
            B = -dC - G1
            if ctx.explicit:
                l_n = B
            else:
                l_n = torch.linalg.solve(Gxn.mT, B)

            dp += G2 
            l = l_n
        
        dp += vjp(lambda _p: G(ctx.f, xs[:, 1], xs[:, 0], T[0], T[1], _p, 1), [p], l)[0]
        dp = c[:, None] * dp

        return None, None, None, None, dp, None, None, None

def discrete_adjoint(f, c, x, T, p, explicit=True, record=False, ret_traj=False):
    return DiscreteAdjoint.apply(T, f, c, x, p, explicit, record, ret_traj)

