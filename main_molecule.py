# -*- coding: utf-8 -*-

from fire import Fire
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch_geometric.datasets import QM9, MoleculeNet, MD17, QM7b
from torch_geometric.loader import DataLoader

from moleculenet.models import EGNN

device = torch.device("cuda")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("moleculenet/output.log"),
        logging.StreamHandler()
    ]
)


def to_onehot(x):
    x_oh = []
    for i in range(x.shape[1]):
        xi = x[:, i]
        xi_unique = xi.unique()
        if len(xi_unique) == 1:
            continue
        for i, n in enumerate(xi_unique):
            xi[xi == n] = i
        x_oh.append(F.one_hot(xi))
    x_oh = torch.cat(x_oh, -1)
    return x_oh


def preprocess(data):
    data.x = to_onehot(data.x).float()
    data.edge_attr = to_onehot(data.edge_attr).float()
    return data


def main(lr=3e-6,
         weight_decay=1e-6,
         epochs=2000,
         test_interval=100,
         ):
    dataset = MoleculeNet('moleculenet', 'ESOL')
    dataset._data.x = to_onehot(dataset.x).float()
    dataset._data.edge_attr = to_onehot(dataset.edge_attr).float()

    train_dataset = dataset[:800]
    val_dataset = dataset[800:]

    model = EGNN(in_node_nf=dataset.x.shape[1], in_edge_nf=dataset.edge_attr.shape[1], out_node_nf=1,
                 hidden_nf=64, device=device, n_layers=4, attention=False)

    train_loader = DataLoader(train_dataset, batch_size=50)
    val_loader = DataLoader(val_dataset, batch_size=50)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.MSELoss()

    best_loss = 100.
    for e in range(epochs):
        train_loss = train_epoch(model, loss_fn, train_loader, optimizer, lr_scheduler, e, train=True)

        if e % test_interval == 0:
            val_loss = train_epoch(model, loss_fn, val_loader, optimizer, lr_scheduler, e, train=False)
            best_loss = min(val_loss, best_loss)
            logging.info(f'Ep {e} train: {train_loss:.4f}\tval: {val_loss:.4f}\tbest: {best_loss:.4f}')


def train_epoch(model, loss_fn, loader, optimizer, lr_scheduler, e, train):
    loss_list = []
    for i, data in enumerate(loader):
        data = data.to(device)
        pos = torch.zeros((len(data.x), 3), device=data.x.device, dtype=data.x.dtype)
        pred = model(data.x, pos, [data.edge_index[0], data.edge_index[1]], data.edge_attr, data.ptr)[0]

        loss = loss_fn(pred, data.y[:, 0])
        loss_list.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()

    if train:
        lr_scheduler.step()

    return np.mean(loss_list)


if __name__ == '__main__':
    Fire(main)

