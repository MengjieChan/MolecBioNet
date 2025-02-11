# Towards Interpretable Drug-Drug Interaction Prediction: A Graph-Based Approach with Molecular and Network-Level Explanations


## Overview

This repository contains Python codes and datasets necessary to run the MolecBioNet model. MolecBioNet is a novel graph-based framework that integrates molecular and biomedical knowledge for robust and interpretable DDI prediction. By modeling drug pairs as unified entities, MolecBioNet captures both macro-level biological interactions and micro-level molecular influences, offering a comprehensive perspective on DDIs. The framework extracts local subgraphs from biomedical knowledge graphs and constructs hierarchical interaction graphs from molecular representations, leveraging classical graph neural network methods to learn multi-scale representations of drug pairs. To enhance accuracy and interpretability, MolecBioNet introduces two domain-specific pooling strategies: context-aware subgraph pooling (CASPool), which emphasizes biologically relevant entities, and attention-guided influence pooling (AGIPool), which prioritizes influential molecular substructures. Please take a look at our paper for more details on the method.

<p align="center">

<img src="https://github.com/Redamancy-CX330/MolecBioNet/blob/main/Overall%20Framework.png" align="center">

</p>


## Install the Environment

### OS Requirements

The package development version is tested on _Linux_ (Ubuntu 20.04) operating systems with CUDA 12.1.

### Python Dependencies

MRHGNN is tested under ``Python == 3.10.14``. 

We provide a txt file containing the necessary packages for MolecBioNet. All the required basic packages can be installed using the following command:

```
pip install -r requirements.txt
```


## Usage

To train and evaluate the model, you could run the following command.

- Ryu's Dataset

```bash
python train.py --dataset 'Ryu' --batch_size 512 --eval_every_iter 225 --alpha 2 --beta 10 --lr 1e-3 --weight_decay_rate 1e-5 --num_epochs 80
```

- DrugBank Dataset

```bash
python train.py --dataset 'DrugBank'  --batch_size 512 --eval_every_iter 489 --alpha 2 --beta 10 --lr 1e-3 --weight_decay_rate 1e-5 --num_epochs 80
```

## Citation

Please kindly cite this paper if you find it useful for your research. Thanks!
