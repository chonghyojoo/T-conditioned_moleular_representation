import torch
import torch.nn as nn
from rdkit import Chem

from .data import DataTensor
from .MolEncoder import MolEncoder
# from memory_profiler import profile

class MPN(nn.Module):
    """
    This is a class that contains a message passing neural network for encoding a molecule.
    """
    def __init__(self, args):
        super(MPN, self).__init__()
        self.depth = args.depth
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(p=args.dropout)
        self.activation = get_activation_function(args.activation)
        self.bias = args.bias
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(1, self.hidden_size), requires_grad=False
        )
        self.cuda = args.cuda
        self.atom_messages = args.atomMessage
        self.prop = args.property
        self.aggregation = args.aggregation
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.fa_size, self.fb_size, self.fm_size = self.sizes()

        if not self.atom_messages:
            self.W_i = nn.Linear(self.fb_size + self.fa_size, self.hidden_size, bias=self.bias)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.hidden_size + self.fa_size, self.hidden_size, bias=self.bias)
        else:
            self.W_i = nn.Linear(self.fa_size, self.hidden_size, bias=self.bias)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.hidden_size + self.fb_size, self.hidden_size, bias=self.bias)

    def sizes(self):
        dummy_smiles = "CC"
        dummy_mol = Chem.MolFromSmiles(dummy_smiles)
        dummy_encoder = MolEncoder(dummy_mol, property=self.prop)
        return dummy_encoder.get_sizes()

    def forward(self, data: DataTensor):
        """
        Encodes a batch of molecular graphs.
        :param data: Parameter containing the data on which the model needs to be run.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        f_atoms = data.f_atoms.to(self.device)
        f_bonds = data.f_bonds.to(self.device)
        f_mol = data.f_mols.to(self.device)
        a2b = pad_list2d(data.a2b).to(self.device)
        b2a = torch.tensor(data.b2a, dtype=torch.long, device=self.device)
        b2revb = torch.tensor(data.b2revb, dtype=torch.long, device=self.device)
        ascope = data.ascope
        bscope = data.bscope
        # debug

        # if self.atom_messages:
        #    a2a = mol_graph.get_a2a()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms.to(f_atoms.device)) # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds.to(f_atoms.device))  # num_bonds x hidden_size

        message = self.activation(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.atom_messages:
                print("atom messages not supported")
                # nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                # nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                # nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                # message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(
                    message, a2b
                )  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.activation(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x

        # a2x = a2a if self.atom_messages else a2b

        a2x = a2b
        nei_a_message = index_select_ND(
            message, a2x
        )  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden

        a_input = torch.cat(
            [f_atoms, a_message], dim=1
        )  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.activation(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden

        # ReadoutFalse
        mol_vecs = []
        atoms_vecs = []
        selected_f_mol = []

        for i, (a_start, a_size) in enumerate(ascope):
            # print(f"[{i}] a_start: {a_start}, a_size: {a_size}")
            if a_size == 0:
                print(f"Skipping molecule {i} due to zero atom size")
                mol_vecs.append(self.cached_zero_vector.to(f_atoms.device))
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                atoms_vecs.append(cur_hiddens)
                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(dim=0)
                else:
                    raise ValueError(
                        f"aggregation function {self.aggregation} not defined"
                    )
                mol_vecs.append(mol_vec)
        # print("cached_zero_vector shape:", self.cached_zero_vector.shape)
        # print("self.hidden_size:", self.hidden_size)
        for a_start, a_size in ascope:
            selected_f_mol.append(f_mol[a_start].unsqueeze(0))
        selected_f_mol = torch.cat(selected_f_mol, dim=0).to(f_atoms.device)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        # print("mol_vecs shape:", mol_vecs.shape)
        # print("f_mol shape before:", selected_f_mol.shape)
        if selected_f_mol.dim() == 1:
            selected_f_mol = selected_f_mol.unsqueeze(0)
        # print("f_mol shape after:", selected_f_mol.shape)

        # mol_vecs = torch.cat([mol_vecs, selected_f_mol], dim=1)  # (num_molecules, hidden_size)
        # print("mol_vecs shape:", mol_vecs.shape) 

        return mol_vecs, atoms_vecs


def get_activation_function(activation) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif activation == "PReLU":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "ELU":
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
        indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
        features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = (
        index_size + suffix_dim
    )  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    # target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = torch.index_select(source, 0, index.view(-1))
    target = target.view(
        final_size
    )  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    return target
    # return source.index_select(dim=0, index=index.view(-1)).view(final_size)

def pad_list2d(list2d, pad_value=0):
    """
    Pads a list of lists to make a 2D torch tensor.

    Args:
        list2d (List[List[int]]): ragged nested list
        pad_value (int): value to pad with

    Returns:
        torch.LongTensor: (len(list2d), max_len)
    """
    max_len = max((len(sub) for sub in list2d), default=1)
    padded = [sub + [pad_value] * (max_len - len(sub)) for sub in list2d]
    return torch.tensor(padded, dtype=torch.long)
