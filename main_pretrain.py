import argparse
import os
import random
import numpy as np
import torch
from rdkit import Chem, rdBase

from data.get_pretraining_data import preprocess, get_mordred
from src.pretrain import pretrain

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--pretrain_dpath", type=str, default="./data/pretraining/")
    arg_parser.add_argument(
        "--pretrain_graph_save_path", type=str, default="./data/pretraining/"
    )
    arg_parser.add_argument(
        "--pretrain_mordred_save_path", type=str, default="./data/pretraining/"
    )

    arg_parser.add_argument("--pca_dim", type=int)
    arg_parser.add_argument("--seed", type=int, default=27407)

    args = arg_parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    # We use the demo dataset (1k mols) for convenience in github repo.
    # The full dataset (10M mols collected from Pubchem) can be downloaded from
    # https://arxiv.org/pdf/2010.09885.pdf
    molsuppl = Chem.SmilesMolSupplier(
        args.pretrain_dpath + "pubchem-1k.txt", delimiter=","
    )

    if not os.path.exists(args.pretrain_graph_save_path + "pubchem_graph.npz"):
        preprocess(molsuppl, args.pretrain_graph_save_path)

    if not os.path.exists(args.pretrain_mordred_save_path + "pubchem_mordred.npz"):
        get_mordred(molsuppl, args.pretrain_mordred_save_path)

    if not os.path.exists("./model/pretrained/"):
        os.makedirs("./model/pretrained/")

    pretrain(args)
