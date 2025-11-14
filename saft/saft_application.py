
from saft_gamma import sle_logS10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv('anthracene_acetonitrile.csv') 
solvent =  test_df['solvent_smiles'][0]
solute  =  test_df['solute_smiles'][0]

T_grid = test_df['T (K)'].values 

Tm   = 489.7          # K   <-- REPLACE with literature value (Melting point)
dHfus = 28.2e3        # J/mol <-- REPLACE with literature value (Enthalpy of fusion)

x_list = []
for T in T_grid:
    x = sle_logS10(
        solvent_smiles=solvent,
        solute_smiles=solute,
        T=T,
        P=1.01325e5,
        Tm_solute=Tm,
        dHfus_solute=dHfus,
        dCp_solute=0,
    )
    
    if isinstance(x, dict):
        x = x.get("logS10", np.nan)
    x_list.append(float(x))

# save
df = pd.DataFrame({"T_K": T_grid, "solubility": x_list})
df.to_csv("anthracene_acetonitrile_test.csv", index=False)

# plot
plt.figure()
plt.scatter(df["T_K"], df["solubility"], marker="o", label='SAFT-gamma Mie' )
plt.scatter(test_df["T (K)"], test_df["target"], label='Experiment', color="black", s=40, zorder=3)
plt.xlabel("Temperature (K)")
plt.ylabel("Solubility (Log S)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
