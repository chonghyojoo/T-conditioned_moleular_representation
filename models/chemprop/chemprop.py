#%%
import torch
import torch.nn as nn
from models.MPN import MPN
from types import SimpleNamespace

#%%
def to_namespace(obj):
    # dict to SimpleNamespace
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

class SolvationModel2(nn.Module):
    def __init__(self, args):
        super(SolvationModel2, self).__init__()

        self.args = args
        self.atom_output_dim = args.hidden_size

        # D-MPNN encoder for solute and solvent
        self.solute_encoder = MPN(args)
        self.solvent_encoder = MPN(args)

        # Fully connected layers for solute and solvent
        self.fc_solute  = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fc_solvent = nn.Linear(args.hidden_size, args.ffn_hidden_size)

        d = args.ffn_hidden_size

        # Final head
        self.final_proj1 = nn.Linear(d*2+1, d)
        self.final_act1  = nn.ReLU()
        self.final_out   = nn.Linear(d, args.output_size)


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
        T = batch['T (K)'].view(-1).to(device).view(-1, 1)                          # (B,1)
        T = (T-243.15)/(403.15-243.15)

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


        
        # Final head
        core = self.final_act1(self.final_proj1(torch.cat([g_solv, g_sol, T.unsqueeze(-1) if T.dim()==1 else T], dim=-1)))  # (B,d)
        out  = self.final_out(core)           # (B,1)
        
        return out

    

# %%
