
#%%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
#%%
# function: SMILES to Mol_obje
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        if mol.GetNumBonds() == 0:
            mol = Chem.AddHs(mol) 
        #AllChem.EmbedMolecule(mol)
        AllChem.Compute2DCoords(mol)
    return mol

# function: mol fraction list
def get_mol_list(row):
    mols = []
    if pd.notnull(row['solute_smiles']):
        mols.append(row['solute_mol'])
    if pd.notnull(row['solvent_smiles']):
        mols.append(row['solvent_mol'])
    return mols

# Prepare datasets from GitHub:
# https://github.com/JacksonBurns/fastsolv/tree/main/paper

# %% data preprocessing
data_path = f'../../data/krasnov/solute/bigsoldb_chemprop_nonaq.csv' 

# data_path = '../../data/vermeire/solprop_chemprop_nonaq_ethanol.csv' 

# ================== path list for chemprop ==================
# BigSolDB      : "../data/krasnov/bigsoldb_chemprop_nonaq.csv"
# Leeds_acetone : "../data/boobier/leeds_acetone_chemprop.csv"
# Leeds_benzene : "../data/boobier/leeds_benzene_chemprop.csv"
# Leeds_ethanol : "../data/boobier/leeds_ethanol_chemprop.csv"
# SolProp       : "../data/vermeire/solprop_chemprop_nonaq.csv"

# ================== path list for fastprop ==================
# (fingerprints are already inlcuded)

# BigSolDB      : "../data/krasnov/bigsoldb_fastprop_nonaq.csv"
# Leeds_acetone : "../data/boobier/leeds_acetone_fastprop.csv"
# Leeds_benzene : "../data/boobier/leeds_benzene_fastprop.csv"
# Leeds_ethanol : "../data/boobier/leeds_ethanol_fastprop.csv"
# SolProp       : "../data/vermeire/solprop_fastprop_nonaq.csv"

df = pd.read_csv(data_path)
# Solute graph
df['solute_mol'] = df['solute_smiles'].apply(smiles_to_mol)
# Solvent graph
df['solvent_mol'] = df['solvent_smiles'].apply(smiles_to_mol)

#%%
processed_data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    sample = {
        'solv_name': row['solvent_smiles'],
        'solu_name': row['solute_smiles'],
        'solv_mol' : row['solvent_mol'],
        'solu_mol' : row['solute_mol'],
        'T (K)':row['temperature'],
        'target': [row['logS']]
    }
    processed_data.append(sample)

# %% save the processed data
with open(f'../../leeds_acetone_{mw}_up.pkl', 'wb') as f: # <- change the name of data file
    pickle.dump(processed_data, f)

print("Data preprocessing completed. Total samples:", len(processed_data))



