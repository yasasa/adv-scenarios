import torch


class MPC:
    def __init__(self, dynamics, map, window, max_u, steps=5, dt=0.1):
        self.dynamics = dynamics
        self.map = map
        self.T = window
        self.opt_steps = steps
        self.max_u = max_u
        self.us = torch.zeros(self.T).type_as(self.map.points).requires_grad_(True)
        
    def simulate(self, x):
       # c = self.map.cost(x.view(1, -1)[:, :2], None).squeeze()
        c = 0
        for t in range(self.T):
            x = self.dynamics(x.clone(), self.us[t].unsqueeze(0))
          #  c += torch.abs(2.5 - torch.norm(x[:2], p=4, dim=-1)).squeeze()
           # print(c)
            c_ = self.map.cost(x.view(1, -1)[:, :2], None).squeeze()
            c += c_
       #     print(c_, x, self.us[t])
        
       # print("##")
        return c
        
    def __call__(self, x):
        return self.eval(x)
        
        
    def eval(self, x):
        opt = torch.optim.LBFGS([self.us], line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            c = self.simulate(x)
            c.backward()
            return c
            
        for _ in range(self.opt_steps):
            l = opt.step(closure)
            with torch.no_grad():
                self.us.data = self.us.clamp(min=-self.max_u, max=self.max_u)
            print(l)
            
        print("##")
        
        u = self.us[0]
        with torch.no_grad():
            self.us[:-1] = self.us.clone()[1:]
        return u
        
        
        