#%%
import torch
from torch.utils.data import Dataset
from models.MolEncoder import MolEncoder  

class SolPropDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        # MolEncoder?
        solu_encs = MolEncoder(sample['solu_mol'])  
        solv_encs = MolEncoder(sample['solv_mol'])  
        encs = [solu_encs, solv_encs]

        def enc_to_dict(enc):
            return {
                'f_atoms': torch.tensor(enc.f_atoms, dtype=torch.float),
                'f_bonds': torch.tensor(enc.f_bonds, dtype=torch.float),
                'f_mols': torch.tensor(enc.f_mol, dtype=torch.float),
                'a2b': enc.a2b,
                'b2a': enc.b2a,
                'b2revb': enc.b2revb,
                'ascope': enc.ascope,
                'bscope': enc.bscope
            }

        dicts = [enc_to_dict(enc) for enc in encs]

        return {
            'solv_name':sample['solv_name'],
            'solu_name':sample['solu_name'],
            'solute': dicts[0],
            'solvent': dicts[1],
            'T (K)': torch.tensor(sample['T (K)'], dtype=torch.float),
            'target': torch.tensor(sample['target'], dtype=torch.float)
        }

# %%
