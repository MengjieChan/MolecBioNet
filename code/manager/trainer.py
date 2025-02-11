import torch
import os
import numpy as np
import time
import logging
import random
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import json
from sklearn import metrics
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_to_device_dgl
import csv


GLOBAL_SEED = 42
GLOBAL_WORKER_ID = None

class Trainer(object):
    def __init__(self, params, model, train_data,valid_evaluator,test_evaluator):

        self.params = params
        self.graph_classifier = model

        self.train_data = train_data
        self.molecular_graphs = move_to_device_dgl(self.train_data.molecular_graphs, self.params.device)
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator

        self.batch_size = params.batch_size
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.updates_counter = 0
        self.early_stop = 0
        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = Adam(self.graph_classifier.parameters(), lr=params.lr, weight_decay=params.weight_decay_rate)
        self.scheduler = ExponentialLR(self.optimizer, params.lr_decay_rate)

        self.criterion = nn.CrossEntropyLoss()

        self.move_batch_to_device = move_batch_to_device_dgl
        self.reset_training_state()
        self.test_result = {}
        self.val_result = {}
        

    def train_batch(self):
        total_loss = 0
        all_labels = []
        all_scores = []


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

        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, worker_init_fn=init_fn)

        self.graph_classifier.train()
        bar = tqdm(enumerate(train_dataloader))

        for b_idx, batch in bar:
           
            subgraph_data, mol_graph_data, drug_pairs, relation_labels, polarity = self.move_batch_to_device(batch, self.params.device)
            
            scores, centerloss, contras_loss = self.graph_classifier(subgraph_data, self.molecular_graphs, mol_graph_data, drug_pairs)
            
            loss = self.criterion(scores, relation_labels) + self.params.alpha*centerloss + self.params.beta*contras_loss
            
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1

            bar.set_description('batch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
            with torch.no_grad():
                total_loss += loss.item()
                label_ids = relation_labels.to('cpu').numpy()
                all_labels += label_ids.flatten().tolist()
                all_scores += torch.argmax(scores, dim=1).cpu().flatten().tolist() 

            # valid and test
            if self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result, save_dev_data = self.valid_evaluator.eval()

                logging.info('Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'s')
                
                tic = time.time()
                test_result, save_test_data = self.test_evaluator.eval()
                logging.info('Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'s')


                if result['f1_score'] >= self.best_metric:
                    self.save_classifier()
                    self.early_stop = 0
                    self.best_metric = result['f1_score']
                    self.test_best_metric = test_result['f1_score']
                    self.not_improved_count = 1
                    self.val_result = result
                    self.test_result = test_result
                    logging.info('Test Performance Per Class:' + str(save_test_data) + 'in ' + str(time.time() - tic)+'s')

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count >= self.params.early_stop_epoch:
                        self.early_stop = 1
                        break
                self.last_metric = result['f1_score']
                self.scheduler.step()

        acc = metrics.accuracy_score(all_labels, all_scores)
        auc = metrics.f1_score(all_labels, all_scores, average='macro')
        auc_pr = metrics.f1_score(all_labels, all_scores, average='micro')
        f1 = metrics.f1_score(all_labels, all_scores, average=None)
        
        return total_loss/b_idx, acc, auc, auc_pr, f1


    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, f'best_graph_classifier_{self.params.iFold}.pth'))
        logging.info('Better models found w.r.t accuracy. Saved it!')
        

    def reset_training_state(self):
        self.best_metric = 0
        self.test_best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 1

    def train(self):
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            self.epoch = epoch
            time_start = time.time()
            loss, acc, auc, auc_pr, f1 = self.train_batch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training acc: {acc}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric} in {time_elapsed}')
            

        with open('experiments/%s/result.csv'%(self.params.experiment_name), mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.params.result_fieldnames)
            
            writer.writerow({
                'Fold': self.params.iFold,
                'Eval Accuracy': self.val_result['acc'],
                'Eval F1 Score': self.val_result['f1_score'],
                'Eval PR AUC': self.val_result['pr_auc'],
                'Eval Kappa': self.val_result['k'],
                
                'Test Accuracy': self.test_result['acc'],
                'Test F1 Score': self.test_result['f1_score'],
                'Test PR AUC': self.test_result['pr_auc'],
                'Test Kappa': self.test_result['k']
            })
