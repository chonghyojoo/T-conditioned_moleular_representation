# Origianl architecture can be found in GitHub:
# https://github.com/jacksonburns/fastprop
import torch
import torch.nn as nn
from types import SimpleNamespace

def _to_ns(d):
    return d if isinstance(d, SimpleNamespace) else SimpleNamespace(**d)

class ClampN(nn.Module):
    def __init__(self, n=3.0): super().__init__(); self.n = float(n)
    def forward(self, x): return torch.clamp(x, min=-self.n, max=self.n)

def _build_mlp(input_size: int, hidden_size: int, act_fun: str, num_layers: int):
    acts = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leakyrelu": nn.LeakyReLU,
    }
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(acts[act_fun]())
    return nn.Sequential(*layers) if layers else nn.Identity()

class FastProp(nn.Module):
    def __init__(self, fp_solu_dim: int, fp_solv_dim: int, hopt_params: dict, output_size: int = 1):
        super().__init__()
        self.fp_solu_dim = int(fp_solu_dim)
        self.fp_solv_dim = int(fp_solv_dim)
        self.hidden_size = int(hopt_params["hidden_size"])
        self.num_layers  = int(hopt_params["num_layers"])
        self.activation  = str(hopt_params["activation_fxn"]).lower()
        self.input_act   = str(hopt_params["input_activation"]).lower()
        in_size = self.fp_solu_dim + self.fp_solv_dim  + 1  

        # input activation
        if self.input_act == "clamp3":
            self.input_activation = ClampN(n=3.0)
        elif self.input_act == "sigmoid":
            self.input_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported input_activation: {self.input_act}")

        # body + head
        self.body = _build_mlp(in_size, self.hidden_size, self.activation, self.num_layers)
        self.out  = nn.Linear(self.hidden_size if self.num_layers else in_size, output_size)

    def _stack_fp(self, objs):
        return torch.stack([_to_ns(o).fp for o in objs], dim=0)

    def forward(self, batch: dict):
        device = next(self.parameters()).device
        T   = batch["T (K)"].view(-1, 1).to(device)
        sol = self._stack_fp(batch["solute"]).to(device)
        sov = self._stack_fp(batch["solvent"]).to(device)
        x   = torch.cat([sol, sov, T], dim=-1)
        x   = self.input_activation(x)
        x   = self.body(x)
        y   = self.out(x)
        return y
