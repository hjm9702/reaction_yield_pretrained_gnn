import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.nn.pytorch import NNConv, Set2Set, GINEConv
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.functional import edge_softmax
import dgl.function as fn

from util import MC_dropout
#from gnn_models.gin_molclr import GINet
#from gnn_models.gcn_molclr import GCN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from gin_predictor import GINPredictor

class PAGTN(nn.Module):

    def __init__(self, node_in_feats, edge_feats,
                 node_feats = 64,
                 depth=3,
                 n_heads=5,
                 activation=nn.ReLU(),
                 readout_feats=1024
                 ):
        
        super(PAGTN, self).__init__()

        
        
        
        self.gnn = PAGTNGNN(node_in_feats, node_feats, edge_feats,
                              depth, n_heads, activation)

        
        self.readout = Set2Set(input_dim = node_in_feats + node_feats,
                               n_iters = 3,
                               n_layers = 1)
        #self.readout = MLPNodeReadout(node_in_feats + node_feats)    
        
        
        # self.readout_g = MLPNodeReadout(node_feats + node_in_feats, pred_hid_feats)
                        
        self.sparsify = nn.Sequential(
            nn.Linear((node_in_feats + node_feats)*2, readout_feats), nn.PReLU()
        )


    def forward(self, g):
        
        def embed(g):
            
            node_feats_orig = g.ndata['attr']
            edge_feats = g.edata['edge_attr']
            
            
            node_feats_embedding = self.gnn(g, node_feats_orig, edge_feats)
            
            
            return node_feats_orig, node_feats_embedding
            
        node_feats_orig, node_feats = embed(g)
        
        graph_embedding = self.readout(g, torch.cat([node_feats_orig, node_feats], dim=1))
        
        out = self.sparsify(graph_embedding)
        
        
        return out
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
                
                
            own_state[name].copy_(param)
            print(f'variable {name} loaded!')



class PAGTNLayer(nn.Module):
    """
    Single PAGTN layer from `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    edge_feats : int
        Size for the input edge features.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_feats,
                 activation=nn.ReLU()):
        super(PAGTNLayer, self).__init__()
        self.attn_src = nn.Linear(node_in_feats, node_in_feats)
        self.attn_dst = nn.Linear(node_in_feats, node_in_feats)
        self.attn_edg = nn.Linear(edge_feats, node_in_feats)
        
        self.attn_dot = nn.Linear(node_in_feats, 1)
       
        
        self.msg_src = nn.Linear(node_in_feats, node_out_feats)
        self.msg_dst = nn.Linear(node_in_feats, node_out_feats)
        self.msg_edg = nn.Linear(edge_feats, node_out_feats)
        self.wgt_n = nn.Linear(node_in_feats, node_out_feats)
        self.act = activation
        

    

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats) or (V, n_head, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats) or (V, n_head, node_out_feats)
            Updated node features.
        """

        g = g.local_var()
        # In the paper node_src, node_dst, edge feats are concatenated
        # and multiplied with the matrix. We have optimized this step
        # by having three separate matrix multiplication.
        
        g.ndata['src'] = self.attn_src(node_feats)
        g.ndata['dst'] = self.attn_dst(node_feats)
        edg_atn = self.attn_edg(edge_feats).unsqueeze(-2)

        g.apply_edges(fn.u_add_v('src', 'dst', 'e'))
        atn_scores = self.act(g.edata.pop('e') + edg_atn)

        atn_scores = self.attn_dot(atn_scores)
        atn_scores = edge_softmax(g, atn_scores)



        g.ndata['src'] = self.msg_src(node_feats)
        g.ndata['dst'] = self.msg_dst(node_feats)
        g.apply_edges(fn.copy_src('dst', 'e'))
        atn_inp = g.edata.pop('e') + self.msg_edg(edge_feats).unsqueeze(-2)
        
        g.edata['msg'] = atn_scores * atn_inp
        g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'feat'))
        out = g.ndata.pop('feat') + self.wgt_n(node_feats)
        
        return out


class PAGTNGNN(nn.Module):
    """Multilayer PAGTN model for updating node representations.
    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    node_hid_feats : int
        Size for the hidden node features.
    edge_feats : int
        Size for the input edge features.
    depth : int
        Number of PAGTN layers to be applied.
    nheads : int
        Number of attention heads.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """

    def __init__(self,
                 node_in_feats,
                 node_hid_feats,
                 edge_feats,
                 depth,
                 nheads,
                 activation=nn.ReLU()):
        super(PAGTNGNN, self).__init__()
        self.depth = depth
        self.nheads = nheads
        self.node_hid_feats = node_hid_feats

        self.atom_inp = nn.Linear(node_in_feats, node_hid_feats * nheads)
                
        self.model = nn.ModuleList([PAGTNLayer(node_hid_feats, node_hid_feats,
                                               edge_feats,
                                               activation)
                                    for _ in range(depth)])
        
        self.act = activation

        self.layer_norm = nn.LayerNorm(node_hid_feats)
        

    def forward(self, g, node_feats, edge_feats, n_nodes=None):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node features.
        """
        g = g.local_var()
                
        atom_input = self.atom_inp(node_feats).view(-1, self.nheads, self.node_hid_feats)
        atom_input = self.act(atom_input)

        atom_h = atom_input
        for i in range(self.depth):
            # attn_h = self.model[i](g, atom_h, edge_feats)
            #layer norm
            # attn_h = self.layer_norm(attn_h)
            #graph norm
            # attn_h = attn_h / torch.sqrt(torch.repeat_interleave(n_nodes, n_nodes).unsqueeze(-1).unsqueeze(-1))
            # atom_h = torch.nn.functional.relu(attn_h) + atom_input

            atom_h = self.model[i](g, atom_h, edge_feats)

        atom_h = atom_h.mean(1)

        #h_0 = atom_input.mean(1)
        
        return atom_h


class MLPNodeReadout(nn.Module):
    
    def __init__(self, node_feats, graph_feats=300):
        super(MLPNodeReadout, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(node_feats, graph_feats), nn.ReLU(),
            #nn.Linear(graph_feats, graph_feats), nn.ReLU(),
            nn.Linear(graph_feats, graph_feats)
        )

    def forward(self, g, node_feats):

        node_feats = self.project(node_feats)
       
        with g.local_scope():
            g.ndata['h'] = node_feats
            graph_feats = dgl.sum_nodes(g, 'h')

        return graph_feats




class GIN(nn.Module):
    
    def __init__(self, node_in_feats, edge_in_feats, mode,
                node_hid_feats = 300,
                depth = 3,
                readout_feats = 1024):

        super(GIN, self).__init__()

        self.depth = depth
        self.mode = mode


        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats), nn.ReLU()
        )


        self.gnn_layers = nn.ModuleList([GINEConv(
                    apply_func = nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats)
                    )
                ) for _ in range(self.depth)])
        

        # self.readout = Set2Set(input_dim = node_hid_feats*2,
        #                        n_iters = 3,
        #                        n_layers = 1)

        #self.readout = MLPNodeReadout(node_hid_feats + node_hid_feats)
        
        self.readout = AvgPooling()
        
        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        #self.layer_norm = nn.LayerNorm(node_hid_feats)
        
        self.dropout = nn.Dropout(0.1)
        #self.bn = nn.BatchNorm1d(node_hid_feats)
        

    def forward(self, g):
        
        node_feats_orig = g.ndata['attr']
        edge_feats_orig = g.edata['edge_attr']

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        edge_feats = self.project_edge_feats(edge_feats_orig)


        # node_aggr = [node_feats]
        for i in range(self.depth):
            node_feats = self.gnn_layers[i](g,node_feats, edge_feats)
            #node_feats = self.bn(node_feats)
            if i < self.depth-1:
                node_feats = nn.ReLU()(node_feats)

            node_feats = self.dropout(node_feats)
            #layer norm
            #node_feats = self.layer_norm(node_feats)
            #graph norm
            #node_feats = node_feats / torch.sqrt(torch.repeat_interleave(g.number_of_nodes(), g.number_of_nodes()).unsqueeze(1))
            #node_feats = torch.nn.functional.relu(node_feats) + node_feats_init

        # node_aggr.append(node_feats)
        # node_aggr = torch.cat(node_aggr, 1)

        #readout = self.readout(g, torch.cat([node_feats_init, node_feats], dim=1))
        readout = self.readout(g, node_feats)
        #out = self.sparsify(readout)
        
        if self.mode == 'finetune':
            readout = self.sparsify(readout)

        return readout
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            if name.startswith('sparsify'):
                print('pass sparsifier')
                continue

            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
                
                
            own_state[name].copy_(param)
            print(f'variable {name} loaded!')


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, mode, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.mode = mode

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        
        if self.mode == 'finetune':
            readout = self.sparsify(readout)
        #graph_feats = self.sparsify(readout)
        
        return readout


    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            if name.startswith('sparsify'):
                print('pass sparsifier')
                continue

            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
                
                
            own_state[name].copy_(param)
            print(f'variable {name} loaded!')



class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, pretrain_mode = 'naive_gin', pretraining_epoch = 0,
                 readout_feats = 1024,
                 predict_hidden_feats = 512, prob_dropout = 0.1):
        
        super(reactionMPNN, self).__init__()


        if pretrain_mode == 'naive_gin':
            self.mpnn = GIN(node_in_feats, edge_in_feats, 'finetune')
            #self.mpnn = GINPredictor(node_in_feats=node_in_feats,edge_in_feats=edge_in_feats)

        elif pretrain_mode == 'mordred_chembl_gin':
            self.mpnn = GIN(node_in_feats, edge_in_feats, 'finetune')
            state_dict =  torch.load('./pretrained_model/mordred/27407_chembl_gin_%d_encoder.pt'%pretraining_epoch, map_location = 'cuda:0')
            self.mpnn.load_my_state_dict(state_dict)
            print('Successfully loaded pretrained gin mordred-chembl!!!')
        
        elif pretrain_mode == 'mordred_zinc_gin':
            self.mpnn = GIN(node_in_feats, edge_in_feats, 'finetune')
            state_dict =  torch.load('./pretrained_model/mordred/27407_zinc_gin_%d_encoder.pt'%pretraining_epoch, map_location = 'cuda:0')
            self.mpnn.load_my_state_dict(state_dict)
            print('Successfully loaded pretrained gin mordred-zinc!!!')

        elif pretrain_mode == 'mordred_chembl+zinc_gin':
            self.mpnn = GIN(node_in_feats, edge_in_feats, 'finetune')
            state_dict =  torch.load('./pretrained_model/mordred/27407_chembl+zinc_gin_%d_encoder.pt'%pretraining_epoch, map_location = 'cuda:0')
            self.mpnn.load_my_state_dict(state_dict)
            print('Successfully loaded pretrained gin mordred-chembl+zinc!!!')



        # elif pretrain_mode == 'mordred_chembl_mpnn':
        #     self.mpnn = MPNN(node_in_feats, edge_in_feats, 'finetune')
        #     state_dict =  torch.load('./pretrained_model/mordred/27407_chembl_mpnn_%d_encoder.pt'%pretraining_epoch, map_location = 'cuda:0')
        #     self.mpnn.load_my_state_dict(state_dict)
        #     print('Successfully loaded pretrained mpnn mordred-chembl!!!')
        
        # elif pretrain_mode == 'mordred_zinc_mpnn':
        #     self.mpnn = MPNN(node_in_feats, edge_in_feats, 'finetune')
        #     state_dict =  torch.load('./pretrained_model/mordred/27407_zinc_mpnn_5_encoder.pt', map_location = 'cuda:0')
        #     self.mpnn.load_my_state_dict(state_dict)
        #     print('Successfully loaded pretrained mpnn mordred-zinc!!!')
        
        # elif pretrain_mode == 'mordred_chembl_gtn':
        #     self.mpnn = PAGTN(node_in_feats, edge_in_feats, 'finetune')
        #     state_dict =  torch.load('./pretrained_model/mordred/27407_chembl_gtn_5_encoder.pt', map_location = 'cuda:0')
        #     self.mpnn.load_my_state_dict(state_dict)
        #     print('Successfully loaded pretrained gtn mordred-chembl!!!')
        
        # elif pretrain_mode == 'mordred_zinc_gtn':
        #     self.mpnn = PAGTN(node_in_feats, edge_in_feats, 'finetune')
        #     state_dict =  torch.load('./pretrained_model/mordred/27407_zinc_gtn_5_encoder.pt', map_location = 'cuda:0')
        #     self.mpnn.load_my_state_dict(state_dict)
        #     print('Successfully loaded pretrained gtn mordred-zinc!!!')
        
        

        # elif pretrain_mode == 'molclr_gin':
        #     self.mpnn = GINet()
        #     state_dict =  torch.load('./pretrained_model/gin_molclr_pretrained.pth', map_location = 'cuda:0')
        #     self.mpnn.load_my_state_dict(state_dict)
        #     print('Successfully loaded pretrained gin molclr!!!')

        self.predict = nn.Sequential(
            nn.Linear(2 * readout_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )
    
    def forward(self, rmols, pmols):
        
        

        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)

        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)

        return out[:,0], out[:,1]
        

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, val_monitor_epoch = 10, n_forward_pass = 5, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    
    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
        
        # rmol_max_cnt = train_loader.dataset.dataset.dataset.rmol_max_cnt
        # pmol_max_cnt = train_loader.dataset.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.MSELoss(reduction = 'none')
    
   
    
    n_epochs = 500
    optimizer = Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [400, 450], gamma = 0.1, verbose = False)
    
    # n_epochs = 500
    # optimizer = Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-5)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=80, min_lr=1e-7, verbose=True)

    #val_log = np.zeros(n_epochs)
    val_log = []

    writer = SummaryWriter(log_dir = os.path.join('./log_chembl/', model_path))
    writer.flush()


    best_epoch = 0
    for epoch in range(n_epochs):
        
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []

        for batchidx, batchdata in enumerate(train_loader):

            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]
           
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)
            
            pred, logvar = net(inputs_rmol, inputs_pmol)
            

            loss = loss_fn(pred, labels)
            loss = (1 - 0.1) * loss.mean() + 0.1 * ( loss * torch.exp(-logvar) + logvar ).mean()
            #loss = loss_fn(pred, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        

        if (epoch+1) % 10 ==0:
            print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
                %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, np.mean(train_loss_list), (time.time()-start_time)/60))
                
        lr_scheduler.step()

        # validation with test set
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            net.eval()
            MC_dropout(net)
            
            val_y = val_loader.dataset.dataset.yld[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
            result = [mean_absolute_error(val_y, val_y_pred),
                      mean_squared_error(val_y, val_y_pred) ** 0.5,
                      r2_score(val_y, val_y_pred)]
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))
            val_log.append(result[1])

            writer.add_scalar("RMSE/test", result[1], epoch)

            if np.min(val_log) == val_log[-1]:
                 torch.save(net.state_dict(), model_path)
                 best_epoch = epoch+1


        ####### validation with validation set
        # val_y = val_loader.dataset.dataset.dataset.yld[val_loader.dataset.dataset.indices[val_loader.dataset.indices]]
        # val_y_pred, _, _ = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)


        # assert len(val_y_pred) == len(val_y)
        
        # #val_loss = loss_fn(val_y_pred, val_y)
        # #val_loss = mean_squared_error(val_y_pred, val_y, multioutput='raw_values')
        # #val_loss = (1 - 0.1) * val_loss.mean() + 0.1 * ( val_loss * (1/val_y_var) + np.log(val_y_var) ).mean()
        # val_loss = mean_squared_error(val_y_pred, val_y)

        # val_log[epoch] = val_loss
        # if (epoch + 1) % val_monitor_epoch == 0:
        #     print('--- validation at epoch %d, processed %d, current val loss %.3f best val loss %.3f' %(epoch, len(val_y), val_loss, np.min(val_log[:epoch + 1])))      

        # lr_scheduler.step(val_loss)
            
        # #print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))

        # if np.argmin(val_log[:epoch + 1]) == epoch:
        #     torch.save(net.state_dict(), model_path) 
        
        # elif np.argmin(val_log[:epoch + 1]) <= epoch - 100:
        #     break


    #net.load_state_dict(torch.load(model_path))

    print('training terminated at epoch %d' %epoch)

    return net, best_epoch
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    batch_size = test_loader.batch_size
    
    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
        # rmol_max_cnt = test_loader.dataset.dataset.dataset.rmol_max_cnt
        # pmol_max_cnt = test_loader.dataset.dataset.dataset.pmol_max_cnt
    except:
        
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
             
    net.eval()
    MC_dropout(net)
    
    test_y_mean = []
    test_y_var = []
    
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]

            mean_list = []
            var_list = []
            
            for _ in range(n_forward_pass):
                mean, logvar = net(inputs_rmol, inputs_pmol)
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)
    
    return test_y_pred, test_y_epistemic, test_y_aleatoric



