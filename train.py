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

def evaluate(model, val_iterator, config):
    all_preds = []
    all_y = []
    for idx, data in enumerate(val_iterator):
        X = data.x.float()
        edge_index = data.edge_index.int()
        Y = data.y.to(config.device)
        X = Variable(X).to(config.device)
        edge_index = Variable(edge_index).to(config.device)
        loss, Y_new, S_hat = model(X,edge_index,Y)
        predicted = (Y_new > 0.5).float().argmax(dim=1).unsqueeze(1)
        all_preds.extend(predicted.numpy())
        all_y.extend(np.array([0 if i[0] else 1 for i in Y.numpy()]))
    score = accuracy_score(all_y, all_preds)
    return score

def train(config: Config, train_iterator, val_iterator):
    model = GraphVAE(config)

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
            loss, Y_new, S_hat =  model(X,edge_index,Y)
            losses.append(loss.data.cpu().numpy())
            print('Epoch: ', epoch, ', Iter: ', id, ', Loss: ', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (id) % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                losses = []
                val_accuracy = evaluate(model, val_iterator, config)
                print(
                    "Epoch: [{}/{}],  iter: {}, average training loss: {:.5f}, val accuracy: {:.4f}, training time = {:.4f}".format(
                        epoch, config.train_epoch, id, avg_train_loss, val_accuracy, time.time() - start_time))
