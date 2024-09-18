import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score
from data import construct_A_from_edge_index
from config import Config
from model import GraphVAE

def evaluate(model, val_iterator):
    all_preds = []
    all_y = []
    for idx, data in enumerate(val_iterator):
        X = data.x.float()
        edge_index = data.edge_index.int()
        Y = data.y.to(model.device)
        X = Variable(X).to(model.device)
        edge_index = Variable(edge_index).to(model.device)
        A = construct_A_from_edge_index(edge_index, model.num_nodes)
        X_new, A_new, Y_new, logvar_Y, mu_Y, logvar_S, mu_S, edge_index, l = model(X,edge_index,Y)
        predicted = (Y_new > 0.5).float().argmax(dim=1).unsqueeze(1)
        all_preds.extend(predicted.numpy())
        all_y.extend(np.array([0 if i[0] else 1 for i in Y.numpy()]))
    score = accuracy_score(all_y, all_preds)
    return score

def train(config: Config, train_iterator, val_iterator):
    model = GraphVAE(num_nodes=config.num_nodes,
                     num_feats=config.num_feats,
                     latent_dim_S=config.latent_dim_S,
                     latent_dim_Y=config.latent_dim_Y,
                     gcn_hidden_dim=config.gcn_hidden_dim,
                     num_labels=config.num_labels,
                     device=config.device,
                     pool=config.pool)

    optimizer = optim.Adam(list(model.parameters()), lr=config.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_milestones, gamma=config.learning_rate)
    model.train()
    train_losses = []
    losses = []
    start_time = time.time()
    for epoch in range(config.train_epoch):
        for id, data in enumerate(train_iterator):
            model.zero_grad()
            X = data.x.float()
            edge_index = data.edge_index.int()
            Y = data.y.to(config.device)
            X = Variable(X).to(config.device)
            edge_index = Variable(edge_index).to(config.device)
            X_new, A_new, Y_new, logvar_Y, mu_Y, logvar_S, mu_S, edge_index, l = model(X,edge_index,Y)
            loss = model.loss_function(X_new, Y_new, X,Y, logvar_Y, mu_Y, logvar_S, mu_S, edge_index, l)
            losses.append(loss.data.cpu().numpy())
            print('Epoch: ', epoch, ', Iter: ', id, ', Loss: ', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (id) % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                losses = []
                val_accuracy = evaluate(model, val_iterator)
                print(
                    "Epoch: [{}/{}],  iter: {}, average training loss: {:.5f}, val accuracy: {:.4f}, training time = {:.4f}".format(
                        epoch, config.train_epoch, id, avg_train_loss, val_accuracy, time.time() - start_time))
