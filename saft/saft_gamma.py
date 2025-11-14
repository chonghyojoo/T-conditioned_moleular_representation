from rdkit import Chem
from collections import Counter
from sgtpy import mixture, saftgammamie, component
import math
from typing import Dict, Tuple, List
import numpy as np
from rdkit import Chem
from collections import Counter

# SGTPy DB
SGTPY_GROUPS = set([
    'CH3','CH2','CH','C',
    'aCH','aCCH2','aCCH','CH2=','CH=','cCH2',
    'COOH','CH3COCH3','COO','H2O','CH3OH','CH4','CO2',
    'OH','CH2OH','CHOH',
    'NH2','NH','N','cNH','cN',
    'C=','aCCH3','aCOH','cCH','cCHNH','cCHN',
    'H3O+','Li+','Na+','K+','Rb+','Mg2+','Ca2+','Sr2+','Ba2+','N+','OH-',
    'F-','Cl-','Br-','I-','COO-','HSO4-','SO42-','HNO3','NO3-','HCO3-',
    'aCCOaC','aCCOOH','aCNHaC','CH3CO','[CH3][OCH2]','[CH2][OCH2]',
    'cO','CH2O','cCOO'
])

SMARTS = [
    ("[CX3](=O)[OX2H1]", "COOH"),
    ("[CX3](=O)[OX2][#6]", "COO"),
    ("[CX3](=O)[O-]", "COO-"),

    ("c[CX3](=O)c", "aCCOaC"),                 
    ("O=C1c2ccccc2C(=O)c3ccccc13", "aCCOaC"),  
    ("[CX3](=O)[CH3]", "CH3CO"), 


    ("[CH2][OX2H]", "CH2OH"),
    ("[CH1][OX2H]", "CHOH"),
    ("c[OX2H]", "aCOH"),
    ("[CH3]-O-[CH2]", "[CH3][OCH2]"),
    ("[CH2]-O-[CH2]", "[CH2][OCH2]"),
    ("c-O-c", "cO"),

    ("[NX3;H2]-[#6]", "NH2"),
    ("[NX3;H1](-[#6])-[#6]", "NH"),
    ("[NX3](-[#6])(-[#6])-[#6]", "N"),
    ("c-[NX3;H2]", "cNH"),
    ("c-[NX3]", "cN"),

    ("[CH2]=[C]", "CH2="),
    ("[CH]=[C]", "CH="),
    ("[C]=[C]", "C="),

    ("[cH]", "aCH"),
    ("c-[CH3]", "aCCH3"),
    ("c-[CH2]", "aCCH2"),
    ("c-[CH1]", "aCCH"),
]


def _mark_used_selective(mol, used, match, label):
    if label == "aCCOaC":
        for i in match:
            atom = mol.GetAtomWithIdx(i)
            Z = atom.GetAtomicNum()
            if Z == 8:                
                used.add(i)
            elif Z == 6 and not atom.GetIsAromatic():  
                used.add(i)
        return

    selective = {"aCCH3", "aCCH2", "aCCH"}
    if label in selective:
        for i in match:
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
                used.add(i)
        return

    for i in match:
        used.add(i)


def map_smiles_to_groups(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    counts = Counter()
    used = set()

    for smarts, label in SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for hit in mol.GetSubstructMatches(patt, uniquify=True):
            if any(i in used for i in hit):
                continue
            counts[label] += 1
            _mark_used_selective(mol, used, hit, label)

    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in used or atom.GetAtomicNum() != 6:
            continue

        is_aromatic = atom.GetIsAromatic()
        h = atom.GetTotalNumHs()
        in_ring = ring_info.IsAtomInRingOfSize(idx, 5) or ring_info.IsAtomInRingOfSize(idx, 6)

        if is_aromatic:
            counts['aCH' if h >= 1 else 'cCH'] += 1
        elif in_ring and h == 2:
            counts['cCH2'] += 1
        elif h == 3:
            counts['CH3'] += 1
        elif h == 2:
            counts['CH2'] += 1
        elif h == 1:
            counts['CH'] += 1
        else:
            counts['C'] += 1
        used.add(idx)


    if len(counts) == 0:
        nC = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        counts['CH2'] = max(1, nC - 1)
        counts['CH3'] = 1

    return {k: int(v) for k, v in counts.items() if v > 0}

def to_sgtpy_groups_dict(smiles: str) -> dict:
    return map_smiles_to_groups(smiles)

def batch_map_smiles(smiles_list: List[str]) -> List[Tuple[str, Dict[str, int]]]:
    rows = []
    for s in smiles_list:
        cnt = map_smiles_to_groups(s)
        rows.append((s, dict(cnt)))
    return rows

# ================== Solid - Liquid ==================

R = 8.314462618  # J/mol/K

def gamma_infinite(eos, T: float, P: float, solute_idx: int = 1):
    nc = getattr(eos, "nc", 2)

    x_mix = np.zeros(nc, dtype=float)
    x_mix[solute_idx] = 1e-6
    x_mix[0] = 1.0 - x_mix[solute_idx]

    lnphi_mix, _ = eos.logfugef(x_mix, T, P, 'L')
    lnphi_i_mix = float(lnphi_mix[solute_idx])

    x_pure = np.zeros(nc, dtype=float)
    x_pure[solute_idx] = 1.0
    lnphi_pure, _ = eos.logfugef(x_pure, T, P, 'L')
    lnphi_i_pure = float(lnphi_pure[solute_idx])

    ln_gamma = lnphi_i_mix - lnphi_i_pure
    return math.exp(ln_gamma)

def ideal_mole_fraction(T: float, Tm: float, dHfus: float, dCp: float = 0.0) -> float: 
    # T    : K
    # dHfus: J/mol
    # dCp  : J/mol
    ln_xid = -(dHfus / R) * (1.0 / T - 1.0 / Tm)
    if abs(dCp) > 0.0:
        ln_xid -= (dCp / R) * (math.log(T / Tm) + (Tm - T) / T)
    return math.exp(ln_xid)

# ---------------------------
# SLE-based logS (log10 mol/mol)
# ---------------------------
def sle_logS10(
    solvent_smiles: str,
    solute_smiles: str,
    T: float,
    P: float,
    Tm_solute: float,        
    dHfus_solute: float,     
    dCp_solute: float = 0.0,
):

    comp_solvent, comp_solute = batch_map_smiles([solvent_smiles,solute_smiles])

    solve = component(GC = comp_solvent[1])
    solu  = component(GC = comp_solute[1])

    mix = mixture(solve, solu)
    mix.saftgammamie()
    eos = saftgammamie(mix)

    x_id  = ideal_mole_fraction(T, Tm_solute, dHfus_solute, dCp_solute)
    gamma = gamma_infinite(eos, T=T, P=P, solute_idx=1)  

    x = x_id / max(gamma, 1e-20)
    x = max(min(x, 1.0), 1e-30)
    logS10 = math.log10(x)

    return {"x_id": x_id, "gamma": gamma, "x": x, "logS10": logS10}