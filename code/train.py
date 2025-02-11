import os
import argparse
import random
import torch
import numpy as np
import logging
from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning
from manager.trainer import Trainer
from manager.evaluator import Evaluator_multiclass
from model.Classifier_model import Classifier_model
from utils.initialization_utils import initialize_experiment, initialize_model
from data_processor.datasets import SubgraphDataset
from data_processor.subgraph_extraction import generate_subgraph_datasets
import csv
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def process_dataset(params):
    params.db_path = os.path.join(params.main_dir, '..', 'data', params.dataset, params.iFold, f'digraph_hop_{params.hop}_{params.BKG_file_name}')

    print("Database path:", params.db_path)

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
    logging.info("The subgraph data generation phase is Finished!")

    train_data = SubgraphDataset(db_path = params.db_path,
                                db_name = 'train_subgraph',
                                raw_data_paths = params.file_paths,
                                add_traspose_rels = params.add_traspose_rels,
                                dataset = params.dataset,
                                BKG_file_name = params.BKG_file_name)

    test_data = SubgraphDataset(db_path = params.db_path,
                                db_name = 'test_subgraph',
                                dataset = params.dataset,
                                ssp_graph = train_data.ssp_graph,
                                molecular_graphs = train_data.molecular_graphs,
                                frag_graphs = train_data.frag_graphs,
                                id2entity = train_data.id2entity,
                                id2relation = train_data.id2relation,
                                rel = train_data.num_rels,
                                global_graph = train_data.global_graph,
                                BKG_file_name = params.BKG_file_name)

   
    valid_data = SubgraphDataset(db_path = params.db_path,
                                db_name = 'valid_subgraph',
                                dataset = params.dataset,
                                ssp_graph = train_data.ssp_graph,
                                molecular_graphs = train_data.molecular_graphs,
                                frag_graphs = train_data.frag_graphs,
                                id2entity = train_data.id2entity,
                                id2relation = train_data.id2relation,
                                rel = train_data.num_rels,
                                global_graph = train_data.global_graph,
                                BKG_file_name = params.BKG_file_name)

    params.num_rels = train_data.num_rels  # only relations in dataset
    params.global_graph = train_data.global_graph.to(params.device)
    params.aug_num_rels = train_data.aug_num_rels  # including relations in BKG and self loop
    params.num_nodes = train_data.num_node
    params.subgraph_feature_num = train_data.subgraph_feature_num
    params.atom_feats_size = train_data.atom_feats_size
    params.frag_feats_size = train_data.frag_feats_size
    params.num_drugs = train_data.num_drugs
    print('params.subgraph_feature_num:', params.subgraph_feature_num)
    print('params.num_nodes:', params.num_nodes)
    print('params.atom_feats_size:', params.atom_feats_size)
    print('params.frag_feats_size:', params.frag_feats_size)


    logging.info(f"Device: {params.device}")
    logging.info(f"# All nodes: {params.num_nodes}")
    logging.info(f"# Subgraph features: {params.subgraph_feature_num}")
    logging.info(f"# Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    return train_data, valid_data, test_data

  
def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    params.result_file_path = "results.csv"
    params.result_fieldnames = ['Fold', 'Eval Accuracy', 'Eval F1 Score', 'Eval PR AUC', 'Eval Kappa',
                'Test Accuracy', 'Test F1 Score', 'Test PR AUC', 'Test Kappa']
    with open('/root/MolecBioNet/code/experiments/%s/result.csv'%(params.experiment_name), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=params.result_fieldnames)
        writer.writeheader()

    for iFold in range(params.Folds):
        set_seed(params.seed)
        params.iFold = 'iFold_' + str(iFold+1)
        logging.info(f"iFold: {params.iFold}")
        params.file_paths = {
            'train': os.path.join(params.main_dir, '..', 'data', params.dataset, params.iFold, f"{params.train_file}.txt"),
            'valid': os.path.join(params.main_dir, '..', 'data', params.dataset, params.iFold, f"{params.valid_file}.txt"),
            'test': os.path.join(params.main_dir, '..', 'data', params.dataset, params.iFold, f"{params.test_file}.txt")
        }
        train_data, valid_data, test_data = process_dataset(params)

        classifier = initialize_model(params, Classifier_model)
        
        valid_evaluator = Evaluator_multiclass(params, classifier, valid_data)
        test_evaluator = Evaluator_multiclass(params, classifier, test_data,is_test=True)
        print(classifier)
        logging.info(f"Model: {classifier}")
        

        trainer = Trainer(params, classifier, train_data, valid_evaluator, test_evaluator)
        logging.info('start training...')
        trainer.train()
    

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="model params")
    """
    default params are best on drugbank
    best params on BioSNAP: --dataset=BioSNAP  --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2
    """

    # global
    parser.add_argument('--seed', type=int, default=42, help="seeds for random initial")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument("--experiment_name", "-e", type=str, default="default", help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument('--Folds', type=int, default=5)
    
    # dataset
    parser.add_argument('--dataset', "-d", type=str, default='Ryu') # Ryu DrugBank
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test", help="Name of file containing validation triplets")
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file')

    # extract subgraphs 
    parser.add_argument("--max_links", type=int, default=250000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=50, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--enclosing_subgraph', '-en', type=bool, default=False, help='whether to only consider enclosing subgraph')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')

    # trainer
    parser.add_argument("--eval_every_iter", type=int, default=225, help="Interval of iterations to evaluate the model") # 225 489
    parser.add_argument("--save_every_epoch", type=int, default=10, help="Interval of epochs to save a checkpoint of the model")
    parser.add_argument("--early_stop_epoch", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of the optimizer")
    parser.add_argument("--lr_decay_rate", type=float, default=0.93, help="adjust the learning rate via epochs")
    parser.add_argument("--weight_decay_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=80,help="numer of epochs")
    parser.add_argument("--num_workers", type=int, default=16,help="Number of dataloading processes")
    
    # GraphSAGE params
    parser.add_argument("--emb_dim", "-dim", type=int, default=256,help="Entity embedding size")
    parser.add_argument('--gcn_aggregator_type', type=str, choices=['mean', 'gcn', 'pool'], default='mean')
    parser.add_argument("--gcn_dropout", type=float, default=0.2,help="node_dropout rate in GCN layers")

    # loss params
    parser.add_argument('--alpha', type=float, default=2, help="Hyperparameter controlling the contribution of center loss")
    parser.add_argument('--beta', type=float, default=10, help="Hyperparameter controlling the contribution of mutual information loss")

    params = parser.parse_args()

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    
    initialize_experiment(params, __file__)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    main(params)


