# models/fastprop/fastprop_csv_film.py
import torch
import torch.nn as nn
from types import SimpleNamespace

def _to_ns(d):
    return d if isinstance(d, SimpleNamespace) else SimpleNamespace(**d)

class ClampN(nn.Module):
    def __init__(self, n=3.0): super().__init__(); self.n = float(n)
    def forward(self, x): return torch.clamp(x, min=-self.n, max=self.n)

def _build_mlp(input_size: int, hidden_size: int, act_fun: str, num_layers: int):
    acts = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
            "relu6": nn.ReLU6, "leakyrelu": nn.LeakyReLU}
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(acts[act_fun]())
    return nn.Sequential(*layers) if layers else nn.Identity()

class _TempRBF(nn.Module):
    def __init__(self, centers: int = 16, t_min: float = 250.0, t_max: float = 350.0, proj_dim: int = 256):
        super().__init__()
        self.register_buffer("centers", torch.linspace(t_min, t_max, steps=centers).view(1, -1))
        sigma = (t_max - t_min) / max(1, centers - 1)
        self.register_buffer("gamma", torch.tensor(1.0 / (sigma**2), dtype=torch.float32))
        self.proj = nn.Sequential(nn.Linear(centers, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, T):  # T: (B,1)
        diff2 = (T - self.centers) ** 2
        rbf = torch.exp(-self.gamma * diff2)
        return self.proj(rbf)



class _FiLM(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.aff = nn.Linear(cond_dim, 2 * in_dim)
    def forward(self, h, cond):
        gamma, beta = self.aff(cond).chunk(2, dim=-1)
        return h * torch.exp(gamma) # + beta

class FastProp_FiLM(nn.Module):
    def __init__(self, fp_solu_dim: int, fp_solv_dim: int, hopt_params: dict, temp_embed_dim: int, output_size: int = 1):
        super().__init__()
        self.fp_solu_dim = int(fp_solu_dim)
        self.fp_solv_dim = int(fp_solv_dim)
    
        self.hidden_size = int(hopt_params["hidden_size"])
        self.num_layers  = int(hopt_params["num_layers"])
        self.activation  = str(hopt_params["activation_fxn"]).lower()
        self.input_act   = str(hopt_params["input_activation"]).lower()

        in_size = self.fp_solu_dim
        # input activation 
        if self.input_act == "clamp3":
            self.input_activation = ClampN(n=3.0)
        elif self.input_act == "sigmoid":
            self.input_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported input_activation: {self.input_act}")
        self.temp_embed_dim = in_size

        acts = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        # temperature embedding + FiLM
        self.temp_emb   = _TempRBF(centers=16, t_min=248.2, t_max=403.15, proj_dim=temp_embed_dim)
        self.film_layer = _FiLM(in_dim=in_size, cond_dim=temp_embed_dim)

        self.h1 = nn.Sequential(nn.Linear(in_size*2, self.hidden_size), acts[self.activation]())
        
        self.rest = _build_mlp(self.hidden_size, self.hidden_size, self.activation, max(0, self.num_layers - 1))

        # readout 
        self.out = nn.Linear(self.hidden_size if self.num_layers else in_size, output_size)

    def _stack_fp(self, objs):
        return torch.stack([_to_ns(o).fp for o in objs], dim=0)

    def forward(self, batch: dict):
        device = next(self.parameters()).device
        T   = batch["T (K)"].view(-1, 1).to(device)
        sol = self._stack_fp(batch["solute"]).to(device)
        sov = self._stack_fp(batch["solvent"]).to(device)
        tproj = self.temp_emb(T) 

        sol_t = self.film_layer(sol, tproj)
        sov_t = self.film_layer(sov, tproj)

        x0  = torch.cat([sol_t, sov_t], dim=-1)              
        x0  = self.input_activation(x0)

        h1 = self.h1(x0) 
        h = self.rest(h1) 

        y  = self.out(h)                                 # readout
        return y
