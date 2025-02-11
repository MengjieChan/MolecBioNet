import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
    

class GraphSAGE(nn.Module):
    def __init__(self, params):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(params.gcn_dropout)
        self.activation = F.relu
        self.aggregator_type = params.gcn_aggregator_type
        self.emb_dim = params.emb_dim
        self.subgraph_feature_num = params.subgraph_feature_num


        self.pre_embed = nn.Embedding(params.num_nodes, self.emb_dim-self.subgraph_feature_num)
        self.rel_weight = nn.Parameter(torch.Tensor(params.aug_num_rels, self.emb_dim-self.subgraph_feature_num), requires_grad = True)
        nn.init.kaiming_uniform_(self.pre_embed.weight.data, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.rel_weight, nonlinearity='relu')


        self.layer1 = SAGEConv(self.emb_dim-self.subgraph_feature_num, self.emb_dim-self.subgraph_feature_num, self.aggregator_type)
        self.layer2 = SAGEConv(self.emb_dim-self.subgraph_feature_num, self.emb_dim-self.subgraph_feature_num, self.aggregator_type)

        self.bn_kg1 = nn.BatchNorm1d(self.emb_dim-self.subgraph_feature_num)
        self.bn_kg2 = nn.BatchNorm1d(self.emb_dim-self.subgraph_feature_num)



    def forward(self, g):
        h = self.pre_embed(g.ndata['idx'])
        edge_weight = self.rel_weight[g.edata['type']]

        h = self.bn_kg1(self.layer1(g, h, edge_weight=edge_weight))
        h = self.activation(h)
        h = self.dropout(h)
        h = self.bn_kg2(self.layer2(g, h, edge_weight=edge_weight))
        h = self.activation(h)

        return h