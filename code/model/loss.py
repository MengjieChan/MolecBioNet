import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    

class MutualInformationLoss(nn.Module):
    def __init__(self, emb_dim):
        super(MutualInformationLoss, self).__init__()

        # Embedding projection layers
        self.fc_kg = nn.Linear(emb_dim, emb_dim)
        self.fc_mol = nn.Linear(emb_dim, emb_dim)

        self.tanh = nn.Tanh()


    def forward(self, z_kg, z_mol):
        """Calculate the Mutual Information Loss"""

        z_kg = self.tanh(self.fc_kg(z_kg))
        z_mol = self.tanh(self.fc_mol(z_mol))

        # Compute bidirectional KL divergence
        bi_di_kld = F.kl_div(F.log_softmax(z_kg, dim=1), F.softmax(z_mol, dim=1), reduction='batchmean') + F.kl_div(F.log_softmax(z_mol, dim=1), F.softmax(z_kg, dim=1), reduction='batchmean')

        # Compute conditional entropy H(z_mol | z_kg) and H(z_kg | z_mol)
        ce_kg_mol = F.mse_loss(z_kg, z_mol.detach())
        ce_mol_kg = F.mse_loss(z_mol, z_kg.detach())

        # Mutual information loss (minimize mutual information)
        # MI(z_mol, z_kg) = H(z_mol | z_kg) + H(z_kg | z_mol) - (KL(z_mol || z_kg) + KL(z_kg || z_mol))
        mutual_information_loss = ce_kg_mol + ce_mol_kg - bi_di_kld

        return mutual_information_loss.mean(), z_kg, z_mol