import torch
import torch.nn as nn
from .mol_model import MolecularUpdateModule
from .GraphSAGE import GraphSAGE
from .encoder import BKG_encoder, MG_encoder
from .loss import MutualInformationLoss

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, features, latent_vector):
        x = torch.cat([features, latent_vector], dim=-1)  # Combine features and latent vector
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class Classifier_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.global_graph = params.global_graph
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.num_drugs = params.num_drugs

        self.kg_emb_dim = 256
        self.kg_num_layers = 1
        self.kg_num_heads = 4

        self.atom_dim = params.atom_feats_size
        self.mol_hidden_dim = 512

        self.frag_dim = params.frag_feats_size
        self.hidden_dim1 = 128
        self.hidden_dim2 = 128
        self.num_heads = 5
        self.n_rel = params.num_rels

        # Define main modules
        self.embedding_model = GraphSAGE(params)
        self.BKG_encoder = BKG_encoder(
            self.kg_emb_dim, self.kg_num_layers, self.kg_num_heads, self.num_drugs, self.device
        )
        self.MolCov = MolecularUpdateModule(self.atom_dim, self.mol_hidden_dim, self.emb_dim)
        self.MG_encoder = MG_encoder(self.frag_dim, self.hidden_dim1, self.hidden_dim2, self.emb_dim, self.num_heads)

        self.W_d = nn.Linear(self.emb_dim + self.kg_emb_dim, self.emb_dim + self.kg_emb_dim, bias=False)
        nn.init.kaiming_uniform_(self.W_d.weight.data)

        self.decoder1 = MLPDecoder(
            input_dim=self.kg_emb_dim, latent_dim=self.kg_emb_dim, hidden_dim=128, output_dim=self.kg_emb_dim
        )
        self.decoder2 = MLPDecoder(
            input_dim=self.emb_dim, latent_dim=self.emb_dim, hidden_dim=128, output_dim=self.emb_dim
        )

        self.W_final = nn.Linear(self.emb_dim * 5 + self.kg_emb_dim *3, self.n_rel)
        nn.init.kaiming_uniform_(self.W_final.weight.data)

        self.mutual_info_loss = MutualInformationLoss(self.emb_dim)

    def forward(self, g, molecular_graphs, frag_graphs, drug_pairs):
        # KG embeddings
        global_graph_node = self.embedding_model(self.global_graph)
        g.ndata["h"] = torch.cat([global_graph_node[g.ndata["idx"]], g.ndata["n_feat"]], dim=1)
        g, kg_embedding, centerloss = self.BKG_encoder(g, drug_pairs)

        # Molecular and fragment embeddings
        molecular_embedding = self.MolCov(molecular_graphs)
        frag_embeddings, frag_attention = self.MG_encoder(frag_graphs)

        # Compute drug pair-specific embeddings
        head_ids = (g.ndata["id"] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata["id"] == 2).nonzero().squeeze(1)

        drugA = self.W_d(
            torch.cat((molecular_embedding[drug_pairs[:, 0]], g.ndata["h"][head_ids]), dim=-1)
        )
        drugB = self.W_d(
            torch.cat((molecular_embedding[drug_pairs[:, 1]], g.ndata["h"][tail_ids]), dim=-1)
        )

        # Mutual information and decoders
        loss_mi, z_kg, z_mol = self.mutual_info_loss(kg_embedding, frag_embeddings)
        mi_kg = self.decoder1(kg_embedding, z_kg)
        mi_mol = self.decoder2(frag_embeddings, z_mol)

        # Final prediction with complex W_final
        final_features = torch.cat((frag_embeddings, kg_embedding, mi_mol, mi_kg, drugA, drugB), dim=-1)
        scores = self.W_final(final_features)

        return scores, centerloss, loss_mi
