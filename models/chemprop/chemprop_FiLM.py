
import torch
import torch.nn as nn
from models.MPN import MPN
from types import SimpleNamespace


def to_namespace(obj):
    # dict to SimpleNamespace
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

class SolvationModel(nn.Module):
    def __init__(self, args):
        super(SolvationModel, self).__init__()

        self.args = args
        self.atom_output_dim = args.hidden_size

        # D-MPNN encoder for solute and solvent
        self.solute_encoder = MPN(args)
        self.solvent_encoder = MPN(args)

        # Fully connected layers for solute and solvent
        self.fc_solute  = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fc_solvent = nn.Linear(args.hidden_size, args.ffn_hidden_size)

        d = args.ffn_hidden_size

        # Temperature to RBF embedding
        self.rbf_centers = nn.Parameter(torch.linspace(248.2, 403.15, steps=16), requires_grad=False) # Min T - Max T
        self.rbf_gamma   = nn.Parameter(torch.tensor(1.0/ (25.0**2)), requires_grad=False)
        self.temp_proj = nn.Sequential(
            nn.Linear(16 , d), nn.ReLU(), # 16 + 3
            nn.Linear(d, d)
        )

        # Feature-wise Linear Modulation
        self.film_affine = nn.Linear(d, 2*d)  # to [gamma||beta]

        # Final head
        self.final_proj1 = nn.Linear(d*2, d)
        self.final_act1  = nn.ReLU()
        self.final_out   = nn.Linear(d, args.output_size)

    # RBF for dimension expansion
    def _rbf_expand(self, T: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.view(1, -1)                 # (1,16)
        diff2   = (T.view(-1,1) - centers)**2                  # (B,16)
        rbf     = torch.exp(-self.rbf_gamma * diff2)           # (B,16)

        return rbf # torch.cat([rbf, aux], dim=-1)                   
    # Temperature embedding
    def _temp_embed(self, T: torch.Tensor) -> torch.Tensor:
        return self.temp_proj(self._rbf_expand(T))             # (B,d)

    # FiLM
    def _film(self, h: torch.Tensor, tproj: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_affine(tproj).chunk(2, dim=-1)
        gamma = torch.exp(gamma)  # suggestion from original paper
        return h * gamma + beta

    # encode one molecule to (d,) —  squeeze: (1,d) to (d,)
    def _encode_one(self, mol_dict, encoder, fc):
        ns = to_namespace(mol_dict)
        z, _ = encoder(ns)          # z: (1, hidden) or (hidden,)
        if z.dim() == 1:            # (hidden,)
            z = z.unsqueeze(0)      # (1, hidden)
        g = fc(z)                   # (1, d)
        g = g.squeeze(0)            # (d,)
        return g


    def forward(self, batch):
        device = next(self.parameters()).device
        T = batch['T (K)'].view(-1).to(device)                         # (B,)

        # encode solute to (B,d)
        g_sol_list = []
        for solute_dict in batch['solute']:
            g_sol_list.append(self._encode_one(solute_dict, self.solute_encoder, self.fc_solute))
        g_sol = torch.stack(g_sol_list, dim=0).to(device)              # (B, d)

        # encode each solvent to (B,d), NOT (B,1,d)
        g_s_list = []
        for solvent_dict in batch['solvent']:
            g_s_list.append(self._encode_one(solvent_dict, self.solvent_encoder, self.fc_solvent))
        g_solv = torch.stack(g_s_list, dim=0).to(device)  
        
        # Temperature embedding + FiLM
        tproj = self._temp_embed(T)                                    # (B,d)
        g_sol_c  = self._film(g_sol, tproj)                            # (B,d)
        g_s_c = self._film(g_solv, tproj)                            # [(B,d)]*n

        # Final head
        core = self.final_act1(self.final_proj1(torch.cat([g_s_c, g_sol_c], dim=-1)))  # (B,d)
        out  = self.final_out(core)           # (B,1)
        
        return out

    

# %%
