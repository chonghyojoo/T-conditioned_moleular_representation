import csv
from memory_profiler import profile
import numpy as np
import torch
# from features.calculated_features import morgan_fingerprint
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Lipinski

from torch import nn
from .MolEncoder import MolEncoder
from torch.utils.data.dataset import Dataset

import random
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

from .inp import TrainArgs

class MolencoderDatabase:
    def __init__(self):
        self.graph_data = dict()
        self.molencoding_data = dict()
        self.dgl_graph = dict()

    def set_graphencoding(self, smiles: str):
        if smiles is None:
            raise ValueError("Smiles is None")
        if smiles not in self.graph_data:
            if "InChI" in smiles:
                mol = Chem.MolFromInchi(smiles)
                inchi = Chem.MolToInchi(mol, options='/fixedH')
            else:
                mol = Chem.MolFromSmiles(smiles)
                inchi = Chem.MolToInchi(mol, options='/fixedH')
        self.graph_data[smiles] = mol

    def get_graphencoding(self, smiles: str):
        if smiles not in self.graph_data.keys():
            self.set_graphencoding(smiles)
        return self.graph_data[smiles]

    def set_molencoding(self, smiles: str, property: str = "solvation"):
        if smiles is None:
            raise ValueError("Molecule is None")
        mol = self.get_graphencoding(smiles)
        if smiles not in self.molencoding_data:
            self.molencoding_data[smiles] = MolEncoder(mol, property)

    def get_molencoding(self, smiles: str):
        if smiles not in self.molencoding_data:
            self.set_molencoding(smiles)
        return self.molencoding_data[smiles]



class DataPoint:
    """A datapoint contains a list of smiles, a list of targets and a list of features
    the smiles are converted to inchi (including a fixed H layer) and moles
    After operations it also contains a list of scaled targets, scaled features, scaled predictions and predictions"""

    def __init__(self, smiles, targets, features, molefracs, inp: TrainArgs, mol_encoders: MolencoderDatabase):
        self.smiles = smiles
        self.num_mols = len(smiles)
        self.prop = inp.property
        self.add_hydrogens_to_solvent = inp.add_hydrogens_to_solvent
        self.mol_encoders = []
        self.updated_mol_vecs = []
        self.updated_atom_vecs = []
        self.targets = targets
        self.features = features
        molefrac = 1.0
        if not molefracs:
            molefracs = [1.0]
        if inp.solute is True:
            self.solvents = inp.max_num_mols - 1
        else:
            self.solvents = inp.max_num_mols
        if len(molefracs) < inp.max_num_mols:
            for i in molefracs:
                molefrac -= i
            molefracs.append(molefrac)
        self.molefracs = molefracs
        self.scaled_targets = []
        self.scaled_features = []
        self.scaled_predictions = []
        self.predictions = []
        self.scaffold = []
        self.epistemic_uncertainty = []
        self.device = inp.device
        self.mol_encoding = mol_encoders
        self.mol = None

    def get_molfrac_tensor(self):
        return torch.tensor(self.molefracs, device=self.device)

    def get_mol(self):
        mol = [self.mol_encoding.get_graphencoding(smiles) for smiles in self.smiles]
        return mol

    def get_hydrogen_bond_acceptors(self):
        hba = [Chem.Lipinski.NumHAcceptors(mol) for mol in self.get_mol()]
        return hba

    def get_hydrogen_bond_donors(self):
        hbd = [Chem.Lipinski.NumHDonors(mol) for mol in self.get_mol()]
        return hbd

    def get_hydrogen_bond_adjency_matrix(self):
        acceptors = self.get_hydrogen_bond_acceptors()
        donors = self.get_hydrogen_bond_donors()
        adjency_matrix = []
        for acceptor in acceptors:
            row = []
            for donor in donors:
                if acceptor == 0 or donor == 0:
                    row.append(0)
                else:
                    row.append(np.min([acceptor, donor]))
            adjency_matrix.append(row)
        return torch.tensor(adjency_matrix, device=self.device)

    def get_scaffold(self):
        if not self.scaffold:
            if not self.mol:
                self.get_mol()
            self.scaffold = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                             if mol else None for mol in self.mol]
        return self.scaffold

    def get_inchi(self):
        inchi = [Chem.MolToInchi(mol, options='/fixedH') for mol in self.get_mol()]
        return inchi

    def get_mol_encoder(self):
        mol_encoders = [self.mol_encoding.get_molencoding(smiles) for smiles in self.smiles]
        return mol_encoders

    def get_mol_encoder_components(self):
        me = self.get_mol_encoder()
        fa = []
        fb = []
        fm = []
        a2b = []
        b2b = []
        b2revb = []
        for m in me:
            fa.append(m.get_components()[0])
            fb.append(m.get_components()[1])
            fm.append(m.get_components()[2])
            a2b.append(m.get_components()[3])
            b2b.append(m.get_components()[4])
            b2revb.append(m.get_components()[5])
        return fa, fb, fm, a2b, b2b, b2revb


class DatapointList(Dataset):
    """A DatapointList is simply a list of datapoints and allows for operations on the dataset"""
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_targets(self):
        return [d.targets for d in self.data]

    def get_scaled_targets(self):
        return [d.scaled_targets for d in self.data]

    def get_features(self):
        return [d.features for d in self.data]

    def get_scaled_features(self):
        return [d.scaled_features for d in self.data]

    def get_molefracs(self):
        return [d.molefracs for d in self.data]

    def set_scaled_targets(self, l):
        for i in range(0, len(l)):
            self.data[i].scaled_targets = l[i]

    def set_scaled_features(self, l):
        for i in range(0, len(l)):
            self.data[i].scaled_features = l[i]

    def get_mol_encoders(self):
        return [d.get_mol_encoder() for d in self.data]

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)


class DataTensor:
    def __init__(self, list_mol_encoders: MolEncoder, inp: TrainArgs, property: str = "solvation"):
        self.list_mol_encoders = list_mol_encoders
        self.prop = property
        self.cuda = True # inp.device # inp.cuda
        self.device = inp.device
        self.make_tensor()

    def make_tensor(self):
        fa_size, fb_size, fm_size = self.sizes()
        n_atoms = 1
        n_bonds = 1
        a_scope = []
        b_scope = []
        fa = [[0] * fa_size]
        fb = [[0] * (fa_size + fb_size)]
        fm = []
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for enc in self.list_mol_encoders:
            if enc:
                fa.extend(enc.f_atoms)
                fb.extend(enc.f_bonds)
                fm.append(enc.f_mol)
                for a in range(0, enc.n_atoms):
                    a2b.append([b + n_bonds for b in enc.a2b[a]])
                for b in range(enc.n_bonds):
                    b2a.append(n_atoms + enc.b2a[b])
                    b2revb.append(n_bonds + enc.b2revb[b])
                a_scope.append((n_atoms, enc.n_atoms))
                b_scope.append((n_bonds, enc.n_bonds))
                n_atoms += enc.n_atoms
                n_bonds += enc.n_bonds
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.f_atoms = torch.FloatTensor(fa)
        self.f_bonds = torch.FloatTensor(fb)
        self.f_mols = torch.FloatTensor(fm)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.ascope = a_scope
        self.bscope = b_scope
        if self.cuda:
            self.f_atoms = self.f_atoms.to(self.device)
            self.f_bonds = self.f_bonds.to(self.device)
            self.f_mols = self.f_mols.to(self.device)
            self.a2b = self.a2b.to(self.device)
            self.b2a = self.b2a.to(self.device)
            self.b2revb = self.b2revb.to(self.device)

    def sizes(self):
        dummy_smiles = "CC"
        dummy_mol = Chem.MolFromSmiles(dummy_smiles)
        dummy_encoder = MolEncoder(dummy_mol, property=self.prop)
        return dummy_encoder.get_sizes()

    def set_current_a_hiddens(self, a_hiddens: torch.FloatTensor):
        for i, (a_start, a_size) in enumerate(self.ascope):
            if a_size == 0:
                self.list_mol_encoders[i].set_updated_a_messages(
                    nn.Parameter(torch.zeros(a_hiddens.size()), requires_grad=False))
            else:
                cur_hiddens = a_hiddens.narrow(0, a_start, a_size)
                self.list_mol_encoders[i].set_current_a_hiddens(cur_hiddens)

    def get_current_a_hiddens(self):
        return [i.get_current_a_hiddens() for i in self.list_mol_encoders]


# @profile
def read_data(inp: TrainArgs, encoding='utf-8', file=None):
    """
    Reading in the data, assume input, features and next targets. The header should contain 'mol', 'feature', 'frac' or
    'target' as a keyword and headers without these keywords are not allowed. Input are either smiles or inchi, but
    should be the same if multiple molecules are read
    """
    if file is None:
        file = inp.input_file

    f = open(file, 'r', encoding=encoding)
    reader = csv.reader(f, delimiter=inp.delimiter)

    header = next(reader)
    all_data = list()
    solutes_count = []
    solvents_count = []
    targets_count = []
    features_count = []
    molefracs_count = []
    for i in header:
        for j in inp.solute_headers:
            if j == i:
                solutes_count.append(header.index(i))
        for j in inp.solvent_headers:
            if j == i:
                solvents_count.append(header.index(i))
        for j in inp.target_headers:
            if j == i:
                targets_count.append(header.index(i))
        for j in inp.features_headers:
            if j == i:
                features_count.append(header.index(i))
        for j in inp.molefrac_headers:
            if j == i:
                molefracs_count.append(header.index(i))
    if len(solutes_count) > 1:
        print(solutes_count)
        raise NotImplementedError("The SolProp package is not yet able to handle multiple solutes")
    if len(solutes_count) == 0 and inp.solute is True:
        raise NameError("The solute header is not found.")

    molencoderdatabase = MolencoderDatabase()
    data_line = 0
    x = 0
    readerlist = list()
    for line in reader:
        readerlist.append(line)
    # readerlist = readerlist[0:inp.max_molecules]
    # random.shuffle(readerlist)
    print("Length of readerlist: ", len(readerlist))
    for line in readerlist:
        if data_line == x:
            print("Reached " + str(x) + " molecules from the dataset")
            x += 50000
        if len(all_data) == inp.max_molecules:
            break
        data_line += 1
        smiles = list()
        features = list()
        targets = list()
        molefracs = list()
        for count in solutes_count:
            smiles.append(line[count]) if line[count] != '' else smiles.append(None)
        for count in solvents_count:
            if line[count] != '':
                smiles.append(line[count])
        for count in targets_count:
            if len(line[count]) > 0:
                targets.append(float(line[count]))
            else:
                targets.append(np.NaN)
        for count in features_count:
            features.append(float(line[count])) \
                if line[count] else features.append(None)
        for count in molefracs_count:
            if line[count] != '':
                molefracs.append(float(line[count]))
        if np.NaN in targets and len(targets) == 1:
            print("Error in reading the data, this line is skipped.", data_line, line)
            continue
        try:
            for i in smiles:
                if 'InChI' in i:
                    mol = Chem.MolFromInchi(i)
                    Chem.MolToInchi(mol, options='/fixedH')
                else:
                    mol = Chem.MolFromSmiles(i)
                    Chem.MolToInchi(mol, options='/fixedH')
            all_data.append(DataPoint(smiles, targets, features, molefracs, inp, molencoderdatabase))
        except:
            print("Error in data point: ", line)
            continue
    f.close()
    return all_data
