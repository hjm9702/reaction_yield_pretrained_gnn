import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from dgl.convert import graph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.model import GIN, linear_head
from src.train_util import collate_graphs_pretraining


def pretrain(args):
    pretraining_dataset = Pretraining_Dataset(
        args.pretrain_graph_save_path, args.pretrain_mordred_save_path, args.pca_dim
    )

    train_loader = DataLoader(
        dataset=pretraining_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_graphs_pretraining,
        drop_last=True,
    )

    node_dim = pretraining_dataset.node_attr.shape[1]
    edge_dim = pretraining_dataset.edge_attr.shape[1]
    mordred_dim = pretraining_dataset.mordred.shape[1]

    g_encoder = GIN(node_dim, edge_dim).cuda()
    m_predictor = linear_head(in_feats=1024, out_feats=mordred_dim).cuda()

    pc_eigenvalue = pretraining_dataset.pc_eigenvalue

    pretrain_moldescpred(g_encoder, m_predictor, train_loader, pc_eigenvalue, args.seed)


def pretrain_moldescpred(
    g_encoder,
    m_predictor,
    trn_loader,
    pc_eigenvalue,
    seed,
    cuda=torch.device("cuda:0"),
):
    max_epochs = 10

    pretrained_model_path = "./model/pretrained/" + "%d_pretrained_gnn.pt" % (seed)

    optimizer = Adam(
        list(g_encoder.parameters()) + list(m_predictor.parameters()),
        lr=5e-4,
        weight_decay=1e-5,
    )

    pc_eigenvalue = torch.from_numpy(pc_eigenvalue).to(cuda)

    def weighted_mse_loss(input, target, weight):
        return (weight * ((input - target) ** 2)).mean()

    l_start_time = time.time()

    trn_size = trn_loader.dataset.__len__()
    batch_size = trn_loader.batch_size

    for epoch in range(max_epochs):
        g_encoder.train()
        m_predictor.train()

        start_time = time.time()

        trn_loss_list = []

        for batchidx, batchdata in tqdm(
            enumerate(trn_loader), total=trn_size // batch_size, leave=False
        ):
            inputs, n_nodes, mordred = batchdata

            inputs = inputs.to(cuda)

            mordred = mordred.to(cuda)

            g_rep = g_encoder(inputs)
            m_pred = m_predictor(g_rep)

            loss = weighted_mse_loss(m_pred, mordred, pc_eigenvalue)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            trn_loss_list.append(train_loss)

        printed_train_loss = np.mean(trn_loss_list)
        print(
            "---epoch %d, lr %f, train_loss %.6f, time_per_epoch %f"
            % (
                epoch,
                optimizer.param_groups[-1]["lr"],
                printed_train_loss,
                (time.time() - start_time) / 60,
            )
        )

    torch.save(g_encoder.state_dict(), pretrained_model_path)

    print("pretraining terminated!")
    print("learning time (min):", (time.time() - l_start_time) / 60)


class Pretraining_Dataset:
    def __init__(self, graph_save_path, mordred_save_path, pc_num):
        self.graph_save_path = graph_save_path
        self.mordred_save_path = mordred_save_path
        self.pc_num = pc_num
        self.load()

    def load(self):
        print("mordred_pretrain.py started, pc num: ", self.pc_num)

        [mordred] = np.load(
            self.mordred_save_path + "pubchem_mordred.npz", allow_pickle=True
        )

        # Eliminating descriptors with more than 10 missing values
        missing_col_idx = np.arange(mordred.shape[1])[np.sum(np.isnan(mordred), 0) > 10]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), missing_col_idx)]

        assert np.sum(np.isnan(mordred) > 10) == 0

        # Eliminating descriptors with all zero values
        zero_std_col_idx = np.where(np.nanstd(mordred, axis=0) == 0)[0]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), zero_std_col_idx)]

        # Eliminating descriptors with inf
        inf_col_idx = np.where(np.sum(mordred == np.inf, axis=0) > 0)[0]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), inf_col_idx)]

        # Remove mols with missing values
        non_missing_mols_idx = np.where(np.sum(np.isnan(mordred), 1) == 0)[0]
        mordred = mordred[non_missing_mols_idx]

        # Standardizing descriptors to have a mean of zero and std of one
        scaler = StandardScaler()
        mordred = scaler.fit_transform(mordred)

        # Applying PCA to reduce the dimensionality of descriptors
        pca = PCA(n_components=self.pc_num)
        mordred = pca.fit_transform(mordred)
        self.pc_eigenvalue = pca.explained_variance_
        print("eigenvalue:", self.pc_eigenvalue)

        # Clipping each dimension to -10*std ~ 10*std
        mordred = np.clip(mordred, -np.std(mordred, 0) * 10, np.std(mordred, 0) * 10)

        # Re-standardizing descriptors
        scaler = StandardScaler()
        mordred = scaler.fit_transform(mordred)

        print("mordred processed finished!")

        [mol_dict] = np.load(
            self.graph_save_path + "pubchem_graph.npz",
            allow_pickle=True,
        )

        self.n_node = mol_dict["n_node"][non_missing_mols_idx]
        self.n_edge = mol_dict["n_edge"][non_missing_mols_idx]

        n_csum_tmp = np.concatenate([[0], np.cumsum(mol_dict["n_node"])])
        e_csum_tmp = np.concatenate([[0], np.cumsum(mol_dict["n_edge"])])

        self.node_attr = np.vstack(
            [
                mol_dict["node_attr"][n_csum_tmp[idx] : n_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )

        self.edge_attr = np.vstack(
            [
                mol_dict["edge_attr"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )
        self.src = np.hstack(
            [
                mol_dict["src"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )
        self.dst = np.hstack(
            [
                mol_dict["dst"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )

        self.mordred = mordred

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])

    def __getitem__(self, idx):
        g = graph(
            (
                self.src[self.e_csum[idx] : self.e_csum[idx + 1]],
                self.dst[self.e_csum[idx] : self.e_csum[idx + 1]],
            ),
            num_nodes=self.n_node[idx],
        )
        g.ndata["attr"] = torch.from_numpy(
            self.node_attr[self.n_csum[idx] : self.n_csum[idx + 1]]
        ).float()
        g.edata["edge_attr"] = torch.from_numpy(
            self.edge_attr[self.e_csum[idx] : self.e_csum[idx + 1]]
        ).float()

        n_node = self.n_node[idx].astype(int)
        mordred = self.mordred[idx].astype(float)

        return g, n_node, mordred

    def __len__(self):
        return self.n_node.shape[0]
