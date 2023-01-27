import numpy as np
import sys, csv, os, random
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from model import training, inference

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import reactionMPNN

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import argparse


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--data_id', type = int)
arg_parser.add_argument('--split_id', type = int)
arg_parser.add_argument('--train_size_id', type=int)
arg_parser.add_argument('--pretrain_mode', type = str)
arg_parser.add_argument('--r_seed', type=int, default = 27407)
arg_parser.add_argument('--pretraining_epoch', type=int, default=0)


args = arg_parser.parse_args()

data_id = args.data_id
split_id = args.split_id
train_size_id = args.train_size_id
pretrain_mode = args.pretrain_mode
r_seed = args.r_seed

pretraining_epoch = args.pretraining_epoch


if data_id == 1:
    train_size_list = [2767, 1977, 1186, 791, 395, 197, 98]
    train_size = train_size_list[train_size_id]

elif data_id ==2:
    train_size_list = [4032, 2880, 1728, 1152, 576, 288, 144]
    train_size = train_size_list[train_size_id]


elif data_id ==3:
    train_size_list = [3057, 3055, 3058, 3055]
    train_size = train_size_list[train_size_id-1]


# data_id = 1 #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
# split_id = 0 #data_id 1 & 2: 0-9, data_id 3: 1-4 
# train_size = 2767 #data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]
# pretrain_mode = 'molclr_gin'

if data_id in [1,2]:
    r_seed = 27407+ split_id

if data_id ==3:
    r_seed = r_seed

os.environ["PYTHONHASHSEED"] = str(r_seed)
random.seed(r_seed)
np.random.seed(r_seed)
torch.manual_seed(r_seed)
torch.backends.cudnn.benchmark = False




batch_size = 128
use_saved = False
model_path = './model/model_%d_%d_%d_%d_%s.pt' %(data_id, split_id, train_size, r_seed, pretrain_mode)
if not os.path.exists('./model/'): os.makedirs('./model/')
        
data = GraphDataset(data_id, split_id, pretrain_mode)
frac_split = (train_size + 1e-5)/len(data)
train_set, test_set = split_dataset(data, [frac_split, 1 - frac_split], shuffle=False, random_state=r_seed)

#train_set, val_set = split_dataset(trainval_set, [0.70, 0.30], shuffle=True, random_state = r_seed)


train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
#val_loader = DataLoader(dataset=val_set, batch_size=int(np.min([batch_size, len(val_set)])), shuffle=False, collate_fn=collate_reaction_graphs)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

print('-- CONFIGURATIONS')
print('--- data_type:', data_id, split_id)
print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
print('--- max no. reactants:', data.rmol_max_cnt)
print('--- max no. products:', data.pmol_max_cnt)
print('--- use_saved:', use_saved)
print('--- model_path:', model_path)


# training 
train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
#train_y = train_loader.dataset.dataset.dataset.yld[train_loader.dataset.dataset.indices[train_loader.dataset.indices]]

assert len(train_y) == len(train_set)
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

node_dim = data.rmol_node_attr[0].shape[1]
edge_dim = data.rmol_edge_attr[0].shape[1]
net = reactionMPNN(node_dim, edge_dim, pretrain_mode, pretraining_epoch).cuda()

if use_saved == False:
    print('-- TRAINING')
    #net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)
    net, best_epoch = training(net, train_loader, test_loader, train_y_mean, train_y_std, model_path)
    #torch.save(net.state_dict(), model_path)
else:
    pass
    #print('-- LOAD SAVED MODEL')
    #net.load_state_dict(torch.load(model_path))


# inference

test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]

test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std)
test_y_pred = np.clip(test_y_pred, 0, 100)

result = [mean_absolute_error(test_y, test_y_pred),
          mean_squared_error(test_y, test_y_pred) ** 0.5,
          r2_score(test_y, test_y_pred),
          stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
          
print('-- RESULT')
print('--- test size: %d' %(len(test_y)))
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))


result_name = pretrain_mode+'_'+str(pretraining_epoch)


# save result to csv
if not os.path.exists('./result_%s'%result_name):
    os.mkdir('./result_%s'%result_name)

if data_id == 1:

    if not os.path.isfile(f'result_%s/result_1.csv'%result_name):
        with open(f'result_%s/result_1.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'train_size_id', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman'])

    with open(f'result_%s/result_1.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.train_size_id, args.pretrain_mode, result[0], result[1], result[2], result[3]])

elif data_id == 2:
    if not os.path.isfile(f'result_%s/result_2.csv'%result_name):
        with open(f'result_%s/result_2.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'train_size_id', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman'])

    with open(f'result_%s/result_2.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.train_size_id, args.pretrain_mode, result[0], result[1], result[2], result[3]])


elif data_id == 3:
    if not os.path.isfile(f'result_%s/result_3.csv'%result_name):
        with open(f'result_%s/result_3.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'r_seed', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman'])

    with open(f'result_%s/result_3.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.r_seed, args.pretrain_mode, result[0], result[1], result[2], result[3]])


### best monitoring result

net.load_state_dict(torch.load(model_path))



test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]

test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std)
test_y_pred = np.clip(test_y_pred, 0, 100)

result = [mean_absolute_error(test_y, test_y_pred),
          mean_squared_error(test_y, test_y_pred) ** 0.5,
          r2_score(test_y, test_y_pred),
          stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
          
print('-- RESULT')
print('--- test size: %d' %(len(test_y)))
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))




# save result to csv
if not os.path.exists('./result_%s_best'%result_name):
    os.mkdir('./result_%s_best'%result_name)

if data_id == 1:

    if not os.path.isfile(f'result_%s_best/result_1.csv'%result_name):
        with open(f'result_%s_best/result_1.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'train_size_id', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman', 'best_epoch'])

    with open(f'result_%s_best/result_1.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.train_size_id, args.pretrain_mode, result[0], result[1], result[2], result[3], best_epoch])

elif data_id == 2:
    if not os.path.isfile(f'result_%s_best/result_2.csv'%result_name):
        with open(f'result_%s_best/result_2.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'train_size_id', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman', 'best_epoch'])

    with open(f'result_%s_best/result_2.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.train_size_id, args.pretrain_mode, result[0], result[1], result[2], result[3], best_epoch])


elif data_id == 3:
    if not os.path.isfile(f'result_%s_best/result_3.csv'%result_name):
        with open(f'result_%s_best/result_3.csv'%result_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['data_id', 'split_id', 'r_seed', 'pretrain_mode', 'mae', 'rmse', 'r2', 'spearman', 'best_epoch'])

    with open(f'result_%s_best/result_3.csv'%result_name, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.data_id, args.split_id, args.r_seed, args.pretrain_mode, result[0], result[1], result[2], result[3], best_epoch])