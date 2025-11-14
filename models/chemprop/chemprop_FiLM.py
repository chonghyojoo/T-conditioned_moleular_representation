import torch
import torch.nn as nn
from models.MPN import MPN
from types import SimpleNamespace


def to_namespace(obj):
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

class SolvationModel(nn.Module):
    def __init__(self, args):
        super(SolvationModel, self).__init__()

        self.args = args
        self.atom_output_dim = args.hidden_size
        
        self.solute_encoder = MPN(args)
        self.solvent_encoder = MPN(args)

        self.fc_solute  = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fc_solvent = nn.Linear(args.hidden_size, args.ffn_hidden_size)

        d = args.ffn_hidden_size

        # Temperature to RBF embedding
        self.rbf_centers = nn.Parameter(torch.linspace(248.2, 403.15, steps=16), requires_grad=False) 
        self.rbf_gamma   = nn.Parameter(torch.tensor(1.0/ (25.0**2)), requires_grad=False)
        self.temp_proj = nn.Sequential(
            nn.Linear(16 , d), nn.ReLU(), # 16 + 3
            nn.Linear(d, d)
        )

        self.film_affine = nn.Linear(d, 2*d) 

        self.final_proj1 = nn.Linear(d*2, d)
        self.final_act1  = nn.ReLU()
        self.final_out   = nn.Linear(d, args.output_size)

    def _rbf_expand(self, T: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.view(1, -1)              
        diff2   = (T.view(-1,1) - centers)**2               
        rbf     = torch.exp(-self.rbf_gamma * diff2)           

        return rbf              

    def _temp_embed(self, T: torch.Tensor) -> torch.Tensor:
        return self.temp_proj(self._rbf_expand(T))             


    def _film(self, h: torch.Tensor, tproj: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_affine(tproj).chunk(2, dim=-1)
        gamma = torch.exp(gamma)  
        return h * gamma + beta

    def _encode_one(self, mol_dict, encoder, fc):
        ns = to_namespace(mol_dict)
        z, _ = encoder(ns)          
        if z.dim() == 1:           
            z = z.unsqueeze(0)   
        g = fc(z)                
        g = g.squeeze(0)         
        return g


    def forward(self, batch):
        device = next(self.parameters()).device
        T = batch['T (K)'].view(-1).to(device)                      

        g_sol_list = []
        for solute_dict in batch['solute']:
            g_sol_list.append(self._encode_one(solute_dict, self.solute_encoder, self.fc_solute))
        g_sol = torch.stack(g_sol_list, dim=0).to(device)          

        g_s_list = []
        for solvent_dict in batch['solvent']:
            g_s_list.append(self._encode_one(solvent_dict, self.solvent_encoder, self.fc_solvent))
        g_solv = torch.stack(g_s_list, dim=0).to(device)  
        
        tproj = self._temp_embed(T)                                
        g_sol_c  = self._film(g_sol, tproj)                         
        g_s_c = self._film(g_solv, tproj)                  

        core = self.final_act1(self.final_proj1(torch.cat([g_s_c, g_sol_c], dim=-1)))  
        out  = self.final_out(core)         
        
        return out

    

# %%
