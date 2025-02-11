import os
import numpy as np
import torch
import random
from sklearn import metrics
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score
from tqdm import tqdm
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_to_device_dgl
GLOBAL_SEED = 42
GLOBAL_WORKER_ID = None


def init_fn(worker_id): 
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    seed = GLOBAL_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class Evaluator_multiclass():
    """
    Drugbank
    """
    def __init__(self, params, classifier, data,is_test=False):
        self.params = params
        self.graph_classifier = classifier
        self.data = data
        self.molecular_graphs = move_to_device_dgl(self.data.molecular_graphs, self.params.device)
        self.global_graph = data.global_graph
        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.is_test = is_test
        self.eval_times = 0
        self.current_epoch = 0
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        self.eval_times += 1
        scores = []
        labels = []
        probas_list = []
        total_loss = 0
        self.current_epoch += 1
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn,worker_init_fn=init_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in tqdm(enumerate(dataloader)):
                data, mol_graph_data, drug_pairs, r_labels, polarity = self.move_batch_to_device(batch, self.params.device)

                score_logits, centerloss, contras_loss = self.graph_classifier(data, self.molecular_graphs, mol_graph_data, drug_pairs)
                probas = torch.softmax(score_logits, dim=1).cpu().numpy()

                loss = self.criterion(score_logits, r_labels) + self.params.alpha * centerloss + self.params.beta * contras_loss

                label_ids = r_labels.to('cpu').numpy()
                labels += label_ids.flatten().tolist()
                scores += torch.argmax(score_logits, dim=1).cpu().flatten().tolist()
                probas_list.extend(probas)
                total_loss += loss.item()
                
        acc = metrics.accuracy_score(labels, scores)
        f1_score = metrics.f1_score(labels, scores, average='macro')

        labels_onehot = np.eye(self.params.num_rels)[labels]
        pr_auc = average_precision_score(labels_onehot, probas_list, average="macro")

        f1 = metrics.f1_score(labels, scores, average=None)
        kappa = metrics.cohen_kappa_score(labels, scores)
        return {'loss': total_loss/b_idx, 'acc': acc, 'f1_score': f1_score, 'pr_auc': pr_auc, 'k':kappa}, {'f1': f1}