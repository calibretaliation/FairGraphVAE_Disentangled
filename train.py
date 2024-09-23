import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score
from data import construct_A_from_edge_index
from config import Config
from model import GraphVAE, F_1_U_S, F_2_Y, hgr_correlation


def evaluate(model, val_iterator, config):
    all_preds = []
    all_y = []
    for idx, data in enumerate(val_iterator):
        X = data.x.float()
        edge_index = data.edge_index.int()
        Y = data.y.to(config.device)
        X = Variable(X).to(config.device)
        edge_index = Variable(edge_index).to(config.device)
        loss, Y_new, S_hat, u_S = model(X,edge_index,Y)
        predicted = (Y_new > 0.5).float().argmax(dim=1).unsqueeze(1)
        all_preds.extend(predicted.numpy())
        all_y.extend(np.array([0 if i[0] else 1 for i in Y.numpy()]))
    score = accuracy_score(all_y, all_preds)
    return score

def train(config: Config, train_iterator, val_iterator):
    model = GraphVAE(config)
    f1_u_S = F_1_U_S(config)
    f2_Y = F_2_Y(config)
    optimizer_min = optim.Adam(list(model.parameters()), lr=config.learning_rate)
    scheduler_min = MultiStepLR(optimizer_min, milestones=config.LR_milestones, gamma=config.learning_rate)
    optimizer_max = optim.Adam(list(f1_u_S.parameters()) + list(f2_Y.parameters()), lr=config.learning_rate)
    scheduler_max = MultiStepLR(optimizer_max, milestones=config.LR_milestones, gamma=config.learning_rate)
    model.train()
    f1_u_S.train()
    f2_Y.train()
    train_losses = []
    losses = []
    losses_max = []
    start_time = time.time()
    for epoch in range(config.train_epoch):
        for id, data in enumerate(train_iterator):
            optimizer_min.zero_grad()
            X = data.x.float()
            edge_index = data.edge_index.int()
            Y = data.y.float().to(config.device)
            X = Variable(X).to(config.device)
            edge_index = Variable(edge_index).to(config.device)

            # Min phase
            loss, Y_new, S_hat, u_S = model(X, edge_index, Y)
            losses.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer_min.step()
            scheduler_min.step()

            # Max phase
            optimizer_max.zero_grad()
            _, _, _, u_S = model(X, edge_index, Y)
            f1_us = f1_u_S(u_S)
            f2_y = f2_Y(Y)
            hgr_loss = hgr_correlation(f1_us, f2_y)
            losses_max.append(hgr_loss.data.cpu().numpy())
            hgr_loss.backward()
            optimizer_max.step()
            scheduler_max.step()

            print('Epoch: ', epoch, ', Iter: ', id, ', Loss: ', loss)

            if (id) % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                losses = []
                avg_train_loss_max = np.mean(losses_max)
                train_losses.append(avg_train_loss_max)
                losses_max = []
                val_accuracy = evaluate(model, val_iterator, config)
                print(
                    "Epoch: [{}/{}],  iter: {}, avg loss min phase: {:.5f}, avg loss max phase: {:.5f}, val accuracy: {:.4f}, training time = {:.4f}".format(
                        epoch, config.train_epoch, id, avg_train_loss, avg_train_loss_max, val_accuracy, time.time() - start_time))
