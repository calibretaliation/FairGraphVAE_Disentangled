import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score

from config import Config
from model import GraphVAE, loss_function

def evaluate(model, val_iterator):
    all_preds = []
    all_y = []
    for idx, (X,A,Y) in enumerate(val_iterator):
        x = X.double().to(model.device)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
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
        for id, (A,X,Y) in enumerate(train_iterator):
            model.zero_grad()
            X = X.float()
            A = A.float()
            X = Variable(X).to(config.device)
            A = Variable(A).to(config.device)
            X_new, A_new, Y_new = model(X,A,Y)
            loss = loss_function(X_new, A_new, Y_new, X,A,Y)
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
