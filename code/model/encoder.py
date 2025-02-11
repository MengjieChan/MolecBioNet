import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import dgl
import dgl.sparse as dglsp
import dgl.nn as dglnn
from .loss import CenterLoss


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)
        h = self.gelu(h)
        
        h2 = h
        h = self.FFN2(self.gelu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)
    


class BKG_encoder(nn.Module):
    def __init__(self, kg_emb_dim, num_layers, num_heads, num_drugs, device):
        super(BKG_encoder, self).__init__()

        self.kg_emb_dim = kg_emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.num_drugs = num_drugs
        self.device = device

        self.layers = nn.ModuleList(
            [GTLayer(self.kg_emb_dim, self.num_heads) for _ in range(self.num_layers)]
        )


        self.W_h = nn.Linear(self.kg_emb_dim, self.kg_emb_dim, bias = False)
        self.W_base = nn.Linear(self.kg_emb_dim * 2, self.kg_emb_dim, bias = False)
        nn.init.kaiming_uniform_(self.W_h.weight.data)
        nn.init.kaiming_uniform_(self.W_base.weight.data)

        # center loss
        self.criterion_cent = CenterLoss(num_classes = self.num_drugs, feat_dim = self.kg_emb_dim, device = self.device)
    
    def forward(self, g, drug_pairs):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = g.ndata["h"]

        for layer in self.layers:
            h = layer(A, h)

        g.ndata["h"] = h

        # Extract head and tail node features
        head_ids = (g.ndata["id"] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata["id"] == 2).nonzero().squeeze(1)

        base_vector = torch.cat([g.ndata['h'] [head_ids], g.ndata['h'] [tail_ids]], dim=1)
        g.ndata['base'] = dgl.broadcast_nodes(g, base_vector)
        transformed_h = self.W_h(g.ndata['h'])    # [total_nodes, emb_dim]
        transformed_base = self.W_base(g.ndata['base'])  # [total_nodes, emb_dim]

        # Non-linear attention score computation
        g.ndata['attn'] = (transformed_h * transformed_base).sum(dim=1, keepdim=True).squeeze(1) 
        g.ndata["weighted_h"] = h * dgl.softmax_nodes(g, "attn").unsqueeze(1)

        # Graph embedding output
        g_out = dgl.readout_nodes(g, "weighted_h", op="mean").squeeze()

        # Center loss computation
        center_loss = self.criterion_cent(
            torch.cat([h[head_ids], h[tail_ids]], dim=0),
            torch.cat([drug_pairs[:, 0], drug_pairs[:, 1]], dim=0),
        )

        return g, g_out, center_loss
    




class MG_encoder(nn.Module):
    def __init__(self, frag_dim, hidden_dim1, hidden_dim2, emb_dim, num_heads):
        super(MG_encoder, self).__init__()

        self.frag_dim = frag_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.emb_dim = emb_dim

        self.num_heads = num_heads

        self.layer1 = dglnn.HeteroGraphConv({
            'inter': dglnn.GraphConv(self.frag_dim, self.hidden_dim1 * self.num_heads),
            'cross': dglnn.GATConv(self.frag_dim, self.hidden_dim1, num_heads=self.num_heads, activation=None)},
            aggregate = self._my_agg_func)
        
        self.layer2 = dglnn.HeteroGraphConv({
            'inter': dglnn.GraphConv(self.hidden_dim1 * self.num_heads, self.hidden_dim2 * self.num_heads),
            'cross': dglnn.GATConv(self.hidden_dim1 * self.num_heads, self.hidden_dim2, num_heads=self.num_heads, activation=None)},
            aggregate = self._my_agg_func)
        
        self.layer3_inter = dglnn.GraphConv(self.hidden_dim2 * self.num_heads, self.emb_dim, allow_zero_in_degree=True)
        self.layer3_cross = dglnn.GATConv(self.hidden_dim2 * self.num_heads, self.emb_dim, num_heads=1, activation=None)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim1 * self.num_heads)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2 * self.num_heads)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)


    def _my_agg_func(self, outputs, dsttype):
    # tensors: is a list of tensors to aggregate
    # dsttype: string name of the destination node type for which the
    #          aggregation is performed
        tensor = []
        for data in outputs:
            tensor.append(data.view(data.shape[0], -1))
        stacked = torch.stack(tensor, dim=0)
        return torch.sum(stacked, dim=0)
    
    
    def forward(self, frag_graphs):
        
        h = frag_graphs.ndata['feat']
        
        # GATConv
        h = self.layer1(frag_graphs, {'drug': h})
        h = F.leaky_relu(h['drug'], negative_slope=0.2)
        h = self.bn1(h.view(h.shape[0], -1))

        h = self.layer2(frag_graphs, {'drug': h})
        h = F.leaky_relu(h['drug'], negative_slope=0.2)
        h = self.bn2(h.view(h.shape[0], -1))

        # GATConv
        h1 = self.layer3_inter(frag_graphs['inter'],  h)
        h2, attention_weights = self.layer3_cross(frag_graphs['cross'], (h, h), get_attention=True)
        h = (h1+h2.squeeze()).squeeze()
        h = self.bn3(h)

        frag_graphs.ndata['h'] = h

        src_nodes, _ = frag_graphs['cross'].edges()

        node_out_attention_sum = scatter_add(attention_weights.squeeze(), src_nodes, dim=0)
        frag_graphs.ndata['weighted_h'] = h * node_out_attention_sum.unsqueeze(1) # shape: [num_nodes, feature_dim]
                
        frag_embeddings = dgl.readout_nodes(frag_graphs, 'weighted_h', op='mean').squeeze()

        return frag_embeddings, node_out_attention_sum