import random
import time

import numpy as np
import torch
from torch import optim, sort
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data import construct_A_from_edge_index
from config import Config
from metric import get_counts
from model import GraphVAE, F_1_U_S, F_2_Y, hgr_correlation, efl, S_decoder, S_recon_loss
import matplotlib.pyplot as plt

def evaluate(model, val_idx, config, data, S_true, S_idx):
    all_preds = []
    X = data.x.float()
    edge_index = data.edge_index.int()
    Y = data.y.to(config.device).long()
    Y[Y != 1] = 0
    X = Variable(X).to(config.device)
    edge_index = Variable(edge_index).to(config.device)
    loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_new, u_S = model(X,edge_index,Y, val_idx)
    # predicted = (Y_new > 0.5).float()
    all_preds = Y_new.data.argmax(dim=1).cpu().numpy()
    all_y = Y.squeeze().cpu().numpy()
    print(np.unique(all_y, return_counts=True))
    print(np.unique(all_preds, return_counts=True))
    score = accuracy_score(all_y, all_preds)
    spd = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), )
    aod = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "aod")
    eod = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "eod")
    recall = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "recall")
    far = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "far")
    precision = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "precision")
    accuracy = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "accuracy")
    F1 = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "F1")
    TPR = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "TPR")
    FPR = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "FPR")
    DI = get_counts(all_preds[S_idx], all_y[S_idx], S_true.squeeze().cpu().numpy(), metric = "DI")
    return score, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI

def train(config: Config, train_idx, val_idx, graph):
    model = GraphVAE(config).to(config.device)
    f1_u_S = F_1_U_S(config).to(config.device)
    f2_Y = F_2_Y(config).to(config.device)
    S_classifier = S_decoder(config).to(config.device)
    optimizer_min = optim.Adam(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_min = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_min, T_max=200, eta_min=1e-5)
    optimizer_max = optim.Adam(list(f1_u_S.parameters()) + list(f2_Y.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_max = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, T_max=200, eta_min=1e-5)
    model.train()
    f1_u_S.train()
    f2_Y.train()
    S_classifier.train()
    train_losses = []
    losses = []
    losses_max = []
    start_time = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(list(f1_u_S.parameters()) + list(f2_Y.parameters()), max_norm=1.0)
    loss_dict = {"min_phase_loss":[],
                "s_recon_loss":[],
                "efl_term":[],
                "Y_recon_loss":[],
                "X_recon_loss":[],
                "A_recon_loss":[],
                "kl_u_y":[],
                "kl_u_s":[],
                "hgr_term_min_phase":[],
                "max_phase_loss":[],
                 "val_accuracy":[],
                 "spd":[],
                 'aod':[],
                 'eod':[],
                 'recall':[],
                 'far':[],
                 'precision':[],
                 'accuracy':[],
                 'F1':[],
                 'TPR':[],
                 'FPR':[],
                 "DI":[]
                 }
    val_acc_max = 70
    spd_min = 1e8
    X = graph.x.float()
    edge_index = graph.edge_index.int()
    Y = graph.y.float().to(config.device)
    X = Variable(X).to(config.device)
    edge_index = Variable(edge_index).to(config.device)
    S_idx = torch.tensor(random.sample(sorted(train_idx), 200), dtype = torch.int32)
    S_true = graph.s.float()[S_idx].to(config.device)
    print(train_idx)
    for epoch in tqdm(range(config.train_epoch)):
        optimizer_min.zero_grad()
        # Min phase
        loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_pred, u_S = model(X, edge_index, Y, train_idx)
        Y_pred = Y_pred.argmax(dim=1).unsqueeze(1)[train_idx]
        S_hat = S_classifier(u_S, edge_index).detach()
        s_recon_loss = S_recon_loss(S_hat[S_idx], S_true)
        efl_term = -abs(config.efl_gamma * efl(S_hat[train_idx], Y_pred))
        loss = loss + s_recon_loss + efl_term
        loss_dict["min_phase_loss"].append(loss.data.cpu().numpy())
        loss_dict["s_recon_loss"].append(s_recon_loss.data.cpu().numpy())
        loss_dict["efl_term"].append(efl_term.data.cpu().numpy())
        loss_dict["Y_recon_loss"].append(Y_recon_loss.data.cpu().numpy())
        loss_dict["X_recon_loss"].append(X_recon_loss.data.cpu().numpy())
        loss_dict["A_recon_loss"].append(A_recon_loss.data.cpu().numpy())
        loss_dict["kl_u_y"].append(kl_u_y.data.cpu().numpy())
        loss_dict["kl_u_s"].append(kl_u_s.data.cpu().numpy())
        loss_dict["hgr_term_min_phase"].append(hgr_term.data.cpu().numpy())

        losses.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer_min.step()
        scheduler_min.step()

        # Max phase
        optimizer_max.zero_grad()
        mu, logvar = model.u_S_encoder(X, edge_index)
        u_S = model.reparameterize(mu, logvar)
        f1_us = f1_u_S(u_S)
        f2_y = f2_Y(Y)
        hgr_loss = hgr_correlation(f1_us, f2_y)
        loss_dict["max_phase_loss"].append(hgr_loss.data.cpu().numpy())

        losses_max.append(hgr_loss.data.cpu().numpy())
        hgr_loss.backward()
        optimizer_max.step()
        scheduler_max.step()

        # print('Epoch: ', epoch,', Loss: ', loss)

        if (epoch) % 100 == 0:
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)
            losses = []
            avg_train_loss_max = np.mean(losses_max)
            train_losses.append(avg_train_loss_max)
            losses_max = []
            val_accuracy, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI = evaluate(model, val_idx, config, graph, S_true, S_idx)
            print(
                "Epoch: [{}/{}],  iter: {}, avg loss min phase: {:.5f}, avg loss max phase: {:.5f}, val accuracy: {:.4f}, spd: {:.4f}, training time = {:.4f}".format(
                    epoch, config.train_epoch, epoch, avg_train_loss, avg_train_loss_max, val_accuracy, spd, time.time() - start_time))
            loss_dict["val_accuracy"].extend([val_accuracy])
            loss_dict["spd"].extend([spd])
            loss_dict['aod'].extend([aod])
            loss_dict['eod'].extend([eod])
            loss_dict['recall'].extend([recall])
            loss_dict['far'].extend([far])
            loss_dict['precision'].extend([precision])
            loss_dict['accuracy'].extend([accuracy])
            loss_dict['F1'].extend([F1])
            loss_dict['TPR'].extend([TPR])
            loss_dict['FPR'].extend([FPR])
            loss_dict["DI"].extend([DI])
            if (val_accuracy*100 > val_acc_max) and (spd < spd_min):
                torch.save({
                    'model1_state_dict': model.state_dict(),
                    'model2_state_dict': S_classifier.state_dict(),
                }, 'model/{}_{:.0f}_{:.0f}_{}.pt'.format(config.dataset_name, val_accuracy*100, spd*100, epoch))
                val_acc_max = val_accuracy*100
                spd_min = spd
        if epoch == config.train_epoch - 1:
            torch.save(loss_dict, "data/loss_dict.pt")

