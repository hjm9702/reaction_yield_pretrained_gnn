import argparse
import os
import random
import numpy as np
import torch
from rdkit import rdBase

from data.get_reaction_yield_data import get_graph_data
from src.finetune import finetune

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--yield_dpath", type=str, default="./data/reaction_yield_prediction/"
    )
    arg_parser.add_argument(
        "--yield_graph_save_path", type=str, default="./data/reaction_yield_prediction/"
    )

    # data_id -> #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
    # split_id -> #data_id 1 & 2: 0-9, data_id 3: 1-4
    # train_size -> data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]

    arg_parser.add_argument("--data_id", type=int)
    arg_parser.add_argument("--split_id", type=int)
    arg_parser.add_argument("--train_size_id", type=int)
    arg_parser.add_argument("--seed", type=int, default=27407)

    args = arg_parser.parse_args()

    if args.data_id in [1, 2]:
        seed = 27407 + args.split_id

    elif args.data_id == 3:
        seed = args.seed

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    if args.data_id in [1, 2]:
        if not os.path.exists(
            args.yield_graph_save_path
            + "dataset_%d_%d.npz" % (args.data_id, args.split_id)
        ):
            load_dict = np.load(
                args.yield_dpath
                + "split/data%d_split_%d.npz" % (args.data_id, args.split_id),
                allow_pickle=True,
            )

            rsmi_list = load_dict["data_df"][:, 0]
            yld_list = load_dict["data_df"][:, 1]
            filename = args.yield_graph_save_path + "dataset_%d_%d.npz" % (
                args.data_id,
                args.split_id,
            )

            get_graph_data(rsmi_list, yld_list, filename)

    elif args.data_id == 3:
        if not os.path.exists(
            args.yield_graph_save_path + "test_%d.npz" % args.split_id
        ):
            load_dict = np.load(
                args.yield_graph_save_path + "split/Test%d_split.npz" % args.split_id,
                allow_pickle=True,
            )

            rsmi_list = load_dict["data_df"][:, 0]
            yld_list = load_dict["data_df"][:, 1]
            filename = args.yield_graph_save_path + "test_%d.npz" % args.split_id

            get_graph_data(rsmi_list, yld_list, filename)

    if not os.path.exists("./model/finetuned/"):
        os.makedirs("./model/finetuned/")

    finetune(args)
