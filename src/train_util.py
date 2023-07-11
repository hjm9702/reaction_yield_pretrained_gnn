import numpy as np
import torch
import dgl


def collate_graphs_pretraining(batch):
    gs, n_nodes, mordreds = map(list, zip(*batch))

    gs = dgl.batch(gs)

    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    mordreds = torch.FloatTensor(np.vstack(mordreds))

    return gs, n_nodes, mordreds


def collate_reaction_graphs(batch):
    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(batchdata[-1])

    return *gs, labels


def MC_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    pass
