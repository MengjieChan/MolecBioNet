import numpy as np
from rdkit.Chem import MolFromSmiles,MolToSmiles, BRICS, rdmolops, AllChem
from .features import atom_features, bond_features, encode_bond_15
import rdkit.Chem as Chem
from jarvis.core.specie import get_node_attributes
from collections import defaultdict
from itertools import combinations
import torch
import dgl

def get_mol(smiles, addH=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) 
    if addH == True:
        mol = Chem.AddHs(mol)
    # Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    return mol


def node_features(mol, use_bond_feature = True, encoder_atom = True):
    atom_feature = {}
    atom_neighbor = defaultdict(list)
    bond_feature_dict = {}

    for atom in mol.GetAtoms():
        if encoder_atom:
            atom_feat = np.array(get_node_attributes(atom.GetSymbol(), atom_features="cgcnn"))
        else:
            atom_feat = atom_features(atom, explicit_H = False)
        atom_feature[atom.GetIdx()] = atom_feat
    
    atom_total_features = []

    if use_bond_feature:
        for bond in mol.GetBonds():
            bond_id = bond.GetIdx()
            bond_feat = encode_bond_15(bond)
            bond_feature_dict[bond_id] = bond_feat

            atom_neighbor[bond.GetBeginAtomIdx()].append(bond_id)
            atom_neighbor[bond.GetEndAtomIdx()].append(bond_id)

        for atom_idx in sorted(atom_feature.keys()):
            atom_feat = atom_feature[atom_idx]
            bond_feat = []

            for bond_id in atom_neighbor[atom_idx]:
                bond_feat.append(bond_feature_dict[bond_id])

            if bond_feat:
                avg_bond_feat = np.mean(bond_feat, axis=0)
            else:
                avg_bond_feat = np.zeros(15)
            atom_feat = np.concatenate([atom_feat, avg_bond_feat])
            

            atom_total_features.append(list(atom_feat))
    else:
        for atom_idx in sorted(atom_feature.keys()):
            atom_feat = atom_feature[atom_idx]
            atom_total_features.append(list(atom_feat))

    atom_total_features_tensor = torch.tensor(atom_total_features, dtype=torch.float32)
    
    return atom_total_features_tensor


def smiles_to_graph(smiles):
    mol = get_mol(smiles, addH=False)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)

    src_list = []
    dst_list = []
    edge_features = []

    num_atoms = mol.GetNumAtoms()

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        src_list.append(begin_atom)
        src_list.append(end_atom)
        dst_list.append(end_atom)
        dst_list.append(begin_atom)
        # bond_feat = bond_features(bond)
        bond_feat = list(encode_bond_15(bond))
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    node_feats = node_features(mol, use_bond_feature = True, encoder_atom = True)
    bond_feats = torch.tensor(edge_features, dtype=torch.float32)
    atom_feats_size = node_feats.size(1)

    g = dgl.DGLGraph()
    g.add_nodes(num_atoms)
    g.add_edges(src_list, dst_list)
    g.ndata['feat'] = node_feats
    # g.edata['feat'] = bond_feats

    return g, atom_feats_size


def break_bonds(mol, bond_indices):
    emol = Chem.EditableMol(mol)
    breaked_bonds = []
    # Remove bonds based on indices
    for indices, _ in bond_indices:
        ia,ib = indices
        emol.RemoveBond(ia,ib)
        breaked_bonds.append(indices)
    
    # Get the fragmented molecule
    fragmented_mol = emol.GetMol()
    
    # Generate separate fragments
    atom_indices = Chem.GetMolFrags(fragmented_mol, sanitizeFrags=True)
    fragment_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    fragments = [MolToSmiles(x, False) for x in fragment_mols]

    return fragments, atom_indices, breaked_bonds



def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return torch.tensor(list(fingerprint), dtype=torch.float32)


def frag_graph(smiles):
    # print(smiles)
    mol = get_mol(smiles, addH=False)
    # bond_indices = list(BRICS.FindBRICSBonds(mol))
    # fragments, atom_list, _ = break_bonds(mol, bond_indices)
    fragmented_mol = Chem.FragmentOnBRICSBonds(mol)
    atom_list = Chem.GetMolFrags(fragmented_mol, sanitizeFrags=True)
    fragment_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    fragments = [MolToSmiles(x, False) for x in fragment_mols]

    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)


    fragment_bonds = set()
    for frag1, frag2 in combinations(range(len(atom_list)), 2):
        for atom1 in atom_list[frag1]:
            for atom2 in atom_list[frag2]:
                if atom1 < mol.GetNumAtoms() and atom2 < mol.GetNumAtoms():
                    if adjacency_matrix[atom1, atom2] == 1 or atom1 == atom2:
                        fragment_bonds.add((frag1, frag2))
                        fragment_bonds.add((frag2, frag1))
    

    if fragment_bonds:
        src, dst = zip(*fragment_bonds)
        graph = dgl.graph((src, dst), num_nodes=len(fragments))
    else:
        graph = dgl.graph(([], []), num_nodes=len(fragments))

    fragment_features = []
    atom_matrix = np.zeros((len(fragments), mol.GetNumAtoms()), dtype=int)

    for i, frag_smiles in enumerate(fragments):
        feature = get_morgan_fingerprint(frag_smiles)
        fragment_features.append(feature)

        for atom in atom_list[i]:
            if atom < mol.GetNumAtoms():
                atom_matrix[i, atom] = 1

    fragment_features = torch.stack(fragment_features)

    graph.ndata['feat'] = fragment_features
    graph.ndata['atom_indices'] = torch.tensor(atom_matrix, dtype=torch.float32) 

    if graph.number_of_edges() > 0:
        graph.edata['type'] = torch.ones(graph.number_of_edges(), dtype=torch.int32)
    else:
        graph.edata['type'] = torch.empty((0,), dtype=torch.int32)

    return graph, fragment_features.size(1)


def pre_dgl_graphs(id2smiles):
    
    molecular_graphs = {}
    frag_graphs = {}

    for id, smiles in id2smiles.items():
        molecular_graphs[id], atom_feats_size = smiles_to_graph(smiles)
        frag_graphs[id], frag_feats_size = frag_graph(smiles)

    return molecular_graphs, frag_graphs, atom_feats_size, frag_feats_size


def merge_graphs(graph1, graph2):

    num_nodes_g1 = graph1.number_of_nodes()
    num_nodes_g2 = graph2.number_of_nodes()

    src_g1, dst_g1 = graph1.edges()
    src_g2, dst_g2 = graph2.edges()

    src_edges = torch.cat([src_g1, src_g2 + num_nodes_g1])
    dst_edges = torch.cat([dst_g1, dst_g2 + num_nodes_g1])

    src_cross = torch.arange(num_nodes_g1).unsqueeze(1).repeat(1, num_nodes_g2).view(-1)
    dst_cross = torch.arange(num_nodes_g2).unsqueeze(0).repeat(num_nodes_g1, 1).view(-1) + num_nodes_g1

    merged_graph = dgl.heterograph({
        ('drug', 'inter', 'drug'): (src_edges, dst_edges),
        ('drug', 'cross', 'drug'): (torch.cat([src_cross, dst_cross]), torch.cat([dst_cross, src_cross])),
    })

    merged_features = torch.cat([graph1.ndata['feat'], graph2.ndata['feat']], dim=0)
    merged_graph.ndata['feat'] = merged_features

    return merged_graph