# reaction_yield_pretrained_gnn
Source code for the paper: Improving Chemical Reaction Yield Prediction Using Pre-Trained Graph Neural Networks

## Data
- The datasets used in the paper
  - Pre-Training Dataset (10M mols collected from Pubchem) - https://arxiv.org/pdf/2010.09885.pdf
  - Chemical Reaction Yield Benchmark Datasets - https://github.com/rxn4chemistry/rxn_yields/

## Components
- **data/get_pretraining_data.py** - pre-training dataset preprocessing functions
- **data/get_reaction_yield_data.py** - chemical reaction yield benchmark dataset preprocessing functions
- **src/dataset.py** - data structure & functions
- **src/model.py** - model architecture & training / inference functions
- **src/pretrain.py** - pre-training functions
- **src/finetune.py** - fine-tuning functions
- **src/preprocess_util.py** - util functions for data preprocessing
- **src/util.py** - util functions for model training / inference
- **main_pretrain.py** - script for pre-training
- **main_finetune.py** - script for fine-tuning
- **run.sh** - run code example

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**
- **Mordred**
