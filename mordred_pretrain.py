import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dgl.convert import graph
from dgl.data.utils import split_dataset
import dgl

from model import MPNN, PAGTN, GIN

from tqdm import tqdm
import os
import time
import random
import argparse




def pretrain(args):
    p_dpath = args.p_dpath
    p_dname = args.p_dname
    p_seed = args.p_seed
    backbone = args.backbone
    
    os.environ["PYTHONHASHSEED"] = str(p_seed)
    random.seed(p_seed)
    np.random.seed(p_seed)
    torch.manual_seed(p_seed)
    torch.backends.cudnn.benchmark = False
    
    
        

    pretrained_model_paths = ['./pretrained_model/mordred/%d_%s_%s_5_encoder.pt'%(p_seed, p_dname, backbone), 
                            './pretrained_model/mordred/%d_%s_%s_10_encoder.pt'%(p_seed, p_dname, backbone)]
                            # './pretrained_model/mordred/%d_%s_%s_20_encoder.pt'%(p_seed, p_dname, backbone), 
                            # './pretrained_model/mordred/%d_%s_%s_50_encoder.pt'%(p_seed, p_dname, backbone), 
                            # './pretrained_model/mordred/%d_%s_%s_100_encoder.pt'%(p_seed, p_dname, backbone)]
    


    pc_idx = 70    


    pretraining_dataset = Pretraining_Dataset(p_dpath, p_dname, pc_idx)
    train_loader = DataLoader(dataset = pretraining_dataset, batch_size = 32, shuffle = True, collate_fn = collate_graphs_pretraining, drop_last = True)

    node_dim = pretraining_dataset.node_attr.shape[1]
    edge_dim = pretraining_dataset.edge_attr.shape[1]
    mordred_dim = pretraining_dataset.mordred.shape[1]


    if backbone == 'gin':
        g_encoder = GIN(node_dim, edge_dim, 'pretrain').cuda()
        m_predictor = linear_head(in_feats=300, out_feats = mordred_dim).cuda()
        
    elif backbone == 'mpnn':
        g_encoder = MPNN(node_dim, edge_dim, 'pretrain').cuda()
        m_predictor = linear_head(in_feats=4*64, out_feats = mordred_dim).cuda()

    elif backbone == 'gtn':
        g_encoder = PAGTN(node_dim, edge_dim).cuda()
        m_predictor = linear_head(in_feats=(node_dim + 64)*2, out_feats = mordred_dim).cuda()
        

    pc_eigenvalue = pretraining_dataset.pc_eigenvalue

    
    pretraining_pretext(g_encoder, m_predictor, train_loader, pretrained_model_paths, pc_eigenvalue)






def pretraining_pretext(g_encoder, m_predictor, trn_loader, model_paths, pc_eigenvalue, cuda = torch.device('cuda:0')):

    max_epochs = 10
    optimizer = Adam(list(g_encoder.parameters())+list(m_predictor.parameters()), lr=1e-3, weight_decay = 1e-5)

    #loss_fn = nn.MSELoss()
    pc_eigenvalue = pc_eigenvalue**0.5
    pc_eigenvalue /= np.linalg.norm(pc_eigenvalue, 2)
    pc_eigenvalue = torch.from_numpy(pc_eigenvalue).to(cuda)
    def weighted_mse_loss(input, target, weight):
        return (weight * (input-target)**2).mean()

    l_start_time = time.time()

    trn_size = trn_loader.dataset.__len__()
    batch_size = trn_loader.batch_size
    #trn_log = np.zeros(max_epochs)

    val_log = np.zeros(max_epochs)
    

    for epoch in range(max_epochs):

        g_encoder.train()
        m_predictor.train()

        start_time = time.time()

        trn_loss_list=[]

        for batchidx, batchdata in tqdm(enumerate(trn_loader), total = trn_size // batch_size, leave=False):
        
            inputs, n_nodes, mordred = batchdata

            inputs = inputs.to(cuda)
            
            mordred = mordred.to(cuda)

            g_rep = g_encoder(inputs)
            m_pred = m_predictor(g_rep)

            loss = weighted_mse_loss(m_pred, mordred, pc_eigenvalue)
            #loss = loss_fn(m_pred, mordred).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            trn_loss_list.append(train_loss)
        
        printed_train_loss = np.mean(trn_loss_list)
        print('---epoch %d, lr %f train_loss %.3f'%(epoch, optimizer.param_groups[-1]['lr'], printed_train_loss))


        if epoch ==4:
            torch.save(g_encoder.state_dict(), model_paths[0])
            

    torch.save(g_encoder.state_dict(), model_paths[1])
    
    print('pretraining terminated!')
    print('learning time (min):', (time.time()-l_start_time)/60)



class Pretraining_Dataset():

    def __init__(self, dpath, dname, pc_idx):
        self.dpath = dpath
        self.dname = dname
        self.pc_idx = pc_idx
        self.load()

    def load(self):

        if self.dname in ['chembl', 'zinc', 'chembl+zinc']:
        
            if self.dname == 'chembl':
                mordred = pd.read_csv('%s/%s_mordred/%s_mordred.csv'%(self.dpath, self.dname, self.dname), header=None)

            elif self.dname == 'zinc':
                mordred = pd.concat([pd.read_csv('%s/%s_mordred/%s_mordred_%d.csv'%(self.dpath, self.dname, self.dname, idx), header=None) for idx in range(4)], axis=0)

            elif self.dname == 'chembl+zinc':
                chembl_idx_wo_zinc = np.load('./pretrained_data/chembl/idx_wo_zinc.npz', allow_pickle=True)['data'][0]
                mordred_chembl = pd.read_csv('%s/chembl_mordred/chembl_mordred.csv'%self.dpath, header=None).iloc[chembl_idx_wo_zinc]
                mordred_zinc = pd.concat([pd.read_csv('%s/zinc_mordred/zinc_mordred_%d.csv'%(self.dpath, idx), header=None) for idx in range(4)], axis=0)
                mordred = pd.concat([mordred_chembl, mordred_zinc], axis=0)

                del mordred_chembl
                del mordred_zinc

            print('before # missing values: %d'%(mordred.isna().sum().sum()))
            mordred.dropna(axis=1, inplace=True)
            print('after # missing values: %d'%(mordred.isna().sum().sum()))

            # scaler = StandardScaler()
            # mordred = scaler.fit_transform(mordred)

            mordred = mordred.loc[:, (mordred.std()!=0)]
            mordred = (mordred-mordred.mean())/mordred.std()

            #assert np.isnan(mordred).sum() == 0
            assert mordred.isna().sum().sum() == 0

            if self.dname == 'chembl':
                if self.pc_idx == 60: pc_num = 11
                elif self.pc_idx == 70: pc_num = 25
                elif self.pc_idx == 80: pc_num = 50
                elif self.pc_idx == 90: pc_num = 93
                elif self.pc_idx == 50: pc_num = 5
            
            elif self.dname == 'zinc':
                if self.pc_idx == 60: pc_num = 10
                elif self.pc_idx == 70: pc_num = 22
                elif self.pc_idx == 80: pc_num = 44
                elif self.pc_idx == 90: pc_num = 85

            elif self.dname == 'chembl+zinc':
                if self.pc_idx == 60: pc_num = 13
                elif self.pc_idx == 70: pc_num = 28
                elif self.pc_idx == 80: pc_num = 55
                elif self.pc_idx == 90: pc_num = 99

            pca = PCA(n_components = pc_num)
            mordred = pca.fit_transform(mordred)
            print('pca explained variance:', pca.explained_variance_ratio_.sum())

            self.pc_eigenvalue = pca.explained_variance_

        elif self.dname == 'pubchem':
            pass
            
            # mordred = np.vstack([np.load('%s/%s/pubchem_mordred_%d.npz'%(self.dpath, self.dname, idx), allow_pickle=True)[0] for idx in [0,1,2,8,9]])

            # print('# missing values:', np.isnan(mordred).sum())
            # assert np.isnan(mordred).sum() == 0
            # pca = np.load('%s/pubchem_1M/pubchem_mordred_pc.npz'%(self.dpath), allow_pickle=True)[0]

            # if self.pc_idx == 60: pc_num = 22
            # elif self.pc_idx == 70: pc_num = 43
            # elif self.pc_idx == 80: pc_num = 76

            # mordred = mordred[:, :pc_num]
            # self.pc_eigenvalue = pca.explained_variance_[:pc_num]



        print('mordred load finished!!')
        
        if self.dname == 'chembl':
            [mol_dict_1] = np.load('./pretrained_data/%s/%s_graph_0.npz'%(self.dname, self.dname), allow_pickle=True)
            [mol_dict_2] = np.load('./pretrained_data/%s/%s_graph_1.npz'%(self.dname, self.dname), allow_pickle=True)

            mol_dict = {key: np.concatenate([mol_dict_1[key],mol_dict_2[key]], 0) for key in mol_dict_1.keys()}

            
        elif self.dname == 'zinc':
            [mol_dict_1] = np.load('./pretrained_data/%s/%s_graph_0.npz'%(self.dname, self.dname), allow_pickle=True)
            [mol_dict_2] = np.load('./pretrained_data/%s/%s_graph_1.npz'%(self.dname, self.dname), allow_pickle=True)

            mol_dict = {key: np.concatenate([mol_dict_1[key],mol_dict_2[key]], 0) for key in mol_dict_1.keys()}

        elif self.dname == 'chembl+zinc':
            [mol_dict_1] = np.load('./pretrained_data/chembl/chembl_graph_0.npz', allow_pickle=True)
            [mol_dict_2] = np.load('./pretrained_data/chembl/chembl_graph_1.npz', allow_pickle=True)
            mol_dict_chembl = {key: np.concatenate([mol_dict_1[key],mol_dict_2[key]], 0) for key in mol_dict_1.keys()}
            for key in mol_dict_chembl.keys():
                mol_dict_chembl[key] = mol_dict_chembl[key][chembl_idx_wo_zinc]
            
            
            [mol_dict_3] = np.load('./pretrained_data/zinc/zinc_graph_0.npz', allow_pickle=True)
            [mol_dict_4] = np.load('./pretrained_data/zinc/zinc_graph_1.npz', allow_pickle=True)

            mol_dict = {key: np.concatenate([mol_dict_chembl[key], mol_dict_3[key], mol_dict_4[key]], 0) for key in mol_dict_1.keys()}
        
        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
        #self.smi = mol_dict['smi']
        self.mordred = mordred
        

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])


    

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx].astype(int)
        mordred = self.mordred[idx].astype(float)
        
        return g, n_node, mordred

    def __len__(self):
        return self.n_node.shape[0]


def collate_graphs_pretraining(batch):

    gs, n_nodes, mordreds = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    mordreds = torch.FloatTensor(np.vstack(mordreds))
    
    return gs, n_nodes, mordreds


class linear_head(nn.Module):
    
    def __init__(self, in_feats, out_feats):
        super(linear_head, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        
        
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats)
        )
    
    def forward(self, x):
        return self.mlp(x)






if __name__ == '__main__':

    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--p_dpath', type = str, default='/mnt/hdd/molclr/mordred')
    arg_parser.add_argument('--p_dname', type = str)
    arg_parser.add_argument('--p_seed', type = int)
    arg_parser.add_argument('--backbone', type = str)

    args = arg_parser.parse_args()

    pretrain(args)
