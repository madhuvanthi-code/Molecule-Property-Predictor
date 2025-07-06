from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import numpy as np

def mol_to_graph_data_obj(mol):
    # atom features = atom number one-hot encoded (up to atomic number 10)
    atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
    x = torch.zeros((mol.GetNumAtoms(), 11))
    for i, z in enumerate(atom_types):
        if z <= 10:
            x[i, z] = 1.0

    # edge index
    edge_index = [[], []]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)