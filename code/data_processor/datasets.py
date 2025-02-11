from torch.utils.data import Dataset
import os
import lmdb
import numpy as np
import pandas as pd
import json
import dgl
import torch
import networkx as nx
from utils.data_utils import process_files_ddi
from utils.graph_utils import deserialize, get_neighbors, ssp_multigraph_to_dgl
import scipy.sparse as ssp
from data_processor.mol_graph import pre_dgl_graphs, merge_graphs


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""
    def __init__(self, db_path, db_name, raw_data_paths= None, add_traspose_rels=None, dataset='', ssp_graph = None, molecular_graphs = None, frag_graphs = None, id2entity= None, id2relation= None, rel= None,  global_graph = None, BKG_file_name=''):
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db = self.main_env.open_db(db_name.encode())
        self.db_path = db_path
        self.db_name = db_name
        main_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        BKG_file = os.path.join(main_dir, '..', 'data', dataset, f"{BKG_file_name}.txt")
        self.entity_type = np.loadtxt(os.path.join(main_dir, '..', 'data', dataset, 'entity.txt'))

        if not ssp_graph:
            
            ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel, num_drugs = process_files_ddi(raw_data_paths, BKG_file)
            
            self.num_rels = rel
            print('number of relations:%d'%(self.num_rels))
            self.num_drugs = num_drugs
            print('number of drugs:%d'%(self.num_drugs))
            
            # Add transpose matrices to handle both directions of relations.
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            #add self loops
            ssp_graph.append(ssp.identity(ssp_graph[0].shape[0]))

            # the effective number of relations after adding symmetric adjacency matrices and/or self connections
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = ssp_multigraph_to_dgl(ssp_graph)
            self.ssp_graph = ssp_graph
            
        else:
            self.num_rels = rel
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = global_graph
            self.ssp_graph = ssp_graph

        if not molecular_graphs:
            entity2smiles = pd.read_csv(os.path.join(main_dir, '..', 'data', dataset, 'Drug_Information.txt'), sep = '\t', header = None, names=['entity', 'smiles'])
            entity2smiles = dict(zip(entity2smiles['entity'], entity2smiles['smiles']))
            id2smiles = {id: entity2smiles[id2entity[id]] for id in id2entity if id2entity[id] in entity2smiles}
            molecular_graphs, frag_graphs, atom_feats_size, frag_feats_size = pre_dgl_graphs(id2smiles)
            self.molecular_graphs = molecular_graphs
            self.frag_graphs = frag_graphs
            self.atom_feats_size = atom_feats_size
            self.frag_feats_size = frag_feats_size
        else:
            self.molecular_graphs = molecular_graphs
            self.frag_graphs = frag_graphs


        self.id2entity = id2entity
        self.id2relation = id2relation
        self.num_entity = ssp_graph[0].shape[0]
        with self.main_env.begin(db=self.db) as txn:
            self.num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(write=False) as txn:
            max_n_label_sub = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            max_n_label_obj = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')
            self.max_n_label = [max_n_label_sub, max_n_label_obj]
            print(self.max_n_label)
        self.subgraph_feature_num = max_n_label_sub + 1 + max_n_label_obj + 1 + np.max(self.entity_type).astype(int) + 1
        self.num_node = self.global_graph.num_nodes()
        
    

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes, drug_pair, r_label, g_label, n_labels = deserialize(txn.get(str_id)).values()
            directed_subgraph = self._prepare_subgraphs(nodes, n_labels)
            comb_graph = self._prepare_mol_graph(drug_pair)
            return directed_subgraph, comb_graph, drug_pair, r_label, g_label
           

    def __len__(self):
        return self.num_graphs
    

    def _prepare_mol_graph(self, drug_pair):
        
        frag_graph1 = self.frag_graphs[drug_pair[0]]
        frag_graph2 = self.frag_graphs[drug_pair[1]]
        comb_graph = merge_graphs(frag_graph1, frag_graph2)
        return comb_graph


    def _prepare_subgraphs(self, nodes, n_labels):
        subgraph = self.global_graph.subgraph(nodes)
        src, dst = subgraph.edges()
        subgraph.edata['type'] = self.global_graph.edata['type'][self.global_graph.subgraph(nodes).edata[dgl.EID]]
        
        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph.ndata['ntype'] = torch.LongTensor(self.entity_type[nodes])
        # dgl.save_graphs('subgraph.bin', subgraph)

        subgraph = self._prepare_features(subgraph, n_labels)
        _,_,edges_btw_roots = subgraph.edge_ids(0, 1,return_uv=True)
        subgraph.remove_edges(edges_btw_roots)
        subgraph.edata['id'] = torch.LongTensor(np.arange(subgraph.num_edges()))
        return subgraph


    def _prepare_features(self, subgraph, n_labels):
        # One hot encode the node label feature and the node type feature
        n_nodes = subgraph.number_of_nodes()

        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        node_types = subgraph.ndata['ntype'].numpy()
        num_classes = np.max(self.entity_type).astype(int) + 1
        type_feats = np.zeros((n_nodes, num_classes))
        type_feats[np.arange(n_nodes), node_types.astype(int)] = 1
        n_feats = np.concatenate((label_feats, type_feats), axis=1) if type_feats is not None else label_feats
        subgraph.ndata['n_feat'] = torch.FloatTensor(n_feats)
        #print(subgraph.ndata['n_feat'])
        #print(subgraph.ndata['n_feat'].shape)
        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels]) 
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph

