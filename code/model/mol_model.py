import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

class MolecularUpdateModule(nn.Module):
    def __init__(self, atom_dim, hidden_dim, emb_dim):
        super(MolecularUpdateModule, self).__init__()

        self.conv1 = dglnn.GraphConv(atom_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = dglnn.GraphConv(hidden_dim, emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)

        self.pool = dglnn.SumPooling()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, molecular_graphs):
        graphs = molecular_graphs
        features = graphs.ndata['feat']

        h = self.conv1(graphs, features)
        h = self.bn1(h.view(h.shape[0], -1))
        h = F.relu(h)

        h = self.conv2(graphs, h)
        h = self.bn2(h.view(h.shape[0], -1))

        graphs.ndata['h'] = h

        graph_embedding = self.pool(graphs, h).squeeze()

        return graph_embedding