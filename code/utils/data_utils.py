import numpy as np
from scipy.sparse import csc_matrix

def process_files_ddi(files, BKG_file,keeptrainone = False):
    entity2id = {}
    relation2id = {}

    triplets = {} # key:'train', 'valid', 'test'; value: list[[drug,drug,relation],[drug,drug,relation],...]
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = [] # list
        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
            if  triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
            # data: list, [drug A, drug B, DDI type]
        triplets[file_type] = np.array(data)
    
    num_drugs =  len(entity2id)
        
    triplet_kg = np.loadtxt(BKG_file)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        # same id within train/valid/test and BKG_file does not mean same relation
        if rel+r not in relation2id:
            relation2id[rel+r] = rel+r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    max_entity_id = max(np.max(kg_triple[:, 0]), np.max(kg_triple[:, 1])) + 1
    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed from the train data and BKG data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(max_entity_id, max_entity_id)))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(max_entity_id, max_entity_id)))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, num_drugs