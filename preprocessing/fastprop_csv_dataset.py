from typing import List, Optional, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Fastprop_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        solute_cols: Optional[List[str]] = None,
        solvent_cols: Optional[List[str]] = None,
        temp_col: Optional[str] = None,
        target_col: Optional[str] = None,
        meta_cols: Optional[Dict[str, str]] = None,
        cols: Optional[List[str]] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[cols]

        # meta
        meta_cols = meta_cols or {}
        self.col_solu_name = meta_cols.get("solu_name", "solute_smiles") if "solute_smiles" in self.df.columns else None
        self.col_solv_name = meta_cols.get("solv_name", "solvent_smiles") if "solvent_smiles" in self.df.columns else None

        # feature columns
        if solute_cols is None:
            solute_cols = [c for c in self.df.columns if c.startswith("solute_")]
            solute_cols.remove('solute_smiles')
        if solvent_cols is None:
            solvent_cols = [c for c in self.df.columns if c.startswith("solvent_")]
            solvent_cols.remove('solvent_smiles')
        if len(solute_cols) == 0 or len(solvent_cols) == 0:
            raise ValueError("Not solute_*/ solvent_* columns")
        
        # remove prefix
        solute_feats  = [c.replace("solute_", "") for c in solute_cols]
        solvent_feats = [c.replace("solvent_", "") for c in solvent_cols]
        common_feats = sorted(set(solute_feats) & set(solvent_feats))

        if len(common_feats) == 0:
            raise ValueError("No common feature names between solute_ and solvent_ columns.")

        # rebuild columns keeping only common features (preserve prefix)
        solute_cols  = [f"solute_{f}" for f in common_feats if f"solute_{f}" in self.df.columns]
        solvent_cols = [f"solvent_{f}" for f in common_feats if f"solvent_{f}" in self.df.columns]

        # optional: reorder both identically by common_feats order
        solute_cols  = [f"solute_{f}" for f in common_feats]
        solvent_cols = [f"solvent_{f}" for f in common_feats]

        self.solute_cols = solute_cols
        self.solvent_cols = solvent_cols
        self.fp_solu_dim = len(self.solute_cols)  
        self.fp_solv_dim = len(self.solvent_cols) 

        # temperature
        if temp_col and temp_col in self.df.columns:
            self.temp_col = temp_col
        elif "T (K)" in self.df.columns:
            self.temp_col = "T (K)"
        elif "temperature" in self.df.columns:
            self.temp_col = "temperature"
        else:
            raise ValueError("Not 'T (K)' or 'temperature'")

        # target
        candidates = [target_col] if target_col else ["target", "logS", "y", "label", "value"]
        self.target_col = next((c for c in candidates if c in self.df.columns), None)
        if self.target_col is None:
            raise ValueError("Not target,logS,y,label or value")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sol_fp = torch.tensor(row[self.solute_cols].to_numpy(dtype="float32"))
        sov_fp = torch.tensor(row[self.solvent_cols].to_numpy(dtype="float32"))
        T_val  = torch.tensor(float(row[self.temp_col]), dtype=torch.float32)
        y_val  = torch.tensor([float(row[self.target_col])], dtype=torch.float32)

        solu_name = str(row[self.col_solu_name]) if self.col_solu_name else "NA"
        solv_name = str(row[self.col_solv_name]) if self.col_solv_name else "NA"

        return {
            "solu_name": solu_name,
            "solv_name": solv_name,
            "solute":  {"fp": sol_fp},     
            "solvent": {"fp": sov_fp},
            "T (K)":   T_val,
            "target":  y_val,
        }
