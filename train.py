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
    data = data.to(config.device)
    X = data.x.float()
    edge_index = data.edge_index.int()
    Y = data.y.to(config.device).long()
    Y[Y != 1] = 0
    X = Variable(X).to(config.device)
    edge_index = Variable(edge_index).to(config.device)
    loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_new, u_S = model(X, edge_index, Y, val_idx)
    S_idx = S_idx.cpu()
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

def train(config: Config, loader):

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
    for epoch in tqdm(range(config.train_epoch)):
        loss_list = []
        s_recon_loss_list = []
        efl_term_list = []
        Y_recon_loss_list = []
        X_recon_loss_list = []
        A_recon_loss_list = []
        kl_u_y_list = []
        kl_u_s_list = []
        hgr_term_list = []
        hgr_loss_list = []

        for graph in loader:

            graph = graph.to(config.device)

            max_node_idx = graph.edge_index.max().item()
            num_nodes = graph.num_nodes
            if max_node_idx >= num_nodes:
                print(f"Error: edge_index contains node index {max_node_idx} which exceeds num_nodes {num_nodes}")

            train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
            val_idx = graph.val_mask.nonzero(as_tuple=False).view(-1)

            X = graph.x.float()
            edge_index = graph.edge_index
            Y = graph.y.float()

            S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx)*0.67)), dtype = torch.int32).to(config.device)
            S_true = graph.s.float()[S_idx]

            # Min phase
            loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_pred, u_S = model(X, edge_index, Y, train_idx)
            Y_pred = Y_pred.argmax(dim=1).unsqueeze(1)[train_idx]
            S_hat = S_classifier(u_S, edge_index).detach()
            s_recon_loss = S_recon_loss(S_hat[S_idx], S_true)
            efl_term = -abs(config.efl_gamma * efl(S_hat[train_idx], Y_pred))
            loss = loss + s_recon_loss + efl_term
            loss_list.append(loss.data.cpu().numpy())
            s_recon_loss_list.append(s_recon_loss.data.cpu().numpy())
            efl_term_list.append(efl_term.data.cpu().numpy())
            Y_recon_loss_list.append(Y_recon_loss.data.cpu().numpy())
            X_recon_loss_list.append(X_recon_loss.data.cpu().numpy())
            A_recon_loss_list.append(A_recon_loss.data.cpu().numpy())
            kl_u_y_list.append(kl_u_y.data.cpu().numpy())
            kl_u_s_list.append(kl_u_s.data.cpu().numpy())
            hgr_term_list.append(hgr_term.data.cpu().numpy())
            losses.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer_min.step()
            optimizer_min.zero_grad()
            scheduler_min.step()

            # Max phase
            optimizer_max.zero_grad()
            mu, logvar = model.u_S_encoder(X, edge_index)
            u_S = model.reparameterize(mu, logvar)
            f1_us = f1_u_S(u_S)
            f2_y = f2_Y(Y)
            hgr_loss = hgr_correlation(f1_us, f2_y)
            hgr_loss_list.append(hgr_loss.data.cpu().numpy())
            losses_max.append(hgr_loss.data.cpu().numpy())
            hgr_loss.backward()
            optimizer_max.step()
            scheduler_max.step()

            # print('Epoch: ', epoch,', Loss: ', loss)
        loss_dict["min_phase_loss"].append(np.mean(loss_list))
        loss_dict["s_recon_loss"].append(np.mean(s_recon_loss_list))
        loss_dict["efl_term"].append(np.mean(efl_term_list))
        loss_dict["Y_recon_loss"].append(np.mean(Y_recon_loss_list))
        loss_dict["X_recon_loss"].append(np.mean(X_recon_loss_list))
        loss_dict["A_recon_loss"].append(np.mean(A_recon_loss_list))
        loss_dict["kl_u_y"].append(np.mean(kl_u_y_list))
        loss_dict["kl_u_s"].append(np.mean(kl_u_s_list))
        loss_dict["hgr_term_min_phase"].append(np.mean(hgr_term_list))
        loss_dict["max_phase_loss"].append(np.mean(hgr_loss_list))

        if (epoch) % 100 == 0:
            avg_train_loss = np.mean(losses)
            losses = []
            avg_train_loss_max = np.mean(losses_max)
            losses_max = []
            val_accuracy_list = []
            spd_list = []
            aod_list = []
            eod_list = []
            recall_list = []
            far_list = []
            precision_list = []
            accuracy_list = []
            F1_list = []
            TPR_list = []
            FPR_list = []
            DI_list = []
            for graph in loader:
                val_accuracy, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI = evaluate(model, val_idx, config, graph, S_true, S_idx)
                val_accuracy_list.append(val_accuracy)
                spd_list.append(spd)
                aod_list.append(aod)
                eod_list.append(eod)
                recall_list.append(recall)
                far_list.append(far)
                precision_list.append(precision)
                accuracy_list.append(accuracy)
                F1_list.append(F1)
                TPR_list.append(TPR)
                FPR_list.append(FPR)
                DI_list.append(DI)
            print(
                "Epoch: [{}/{}]:\navg loss min phase: {:.4f}\navg loss max phase: {:.4f}\nval accuracy: {:.4f}\nspd: {:.4f}\nf1: {:.4f}\nEOD: {:.4f}\ntraining time = {:.4f}".format(
                    epoch, config.train_epoch, avg_train_loss, avg_train_loss_max, np.mean(val_accuracy), np.mean(spd), np.mean(F1), np.mean(eod), time.time() - start_time))

            loss_dict["val_accuracy"].append(np.mean(val_accuracy_list))
            loss_dict["spd"].append(np.mean(spd_list))
            loss_dict['aod'].append(np.mean(aod_list))
            loss_dict['eod'].append(np.mean(eod_list))
            loss_dict['recall'].append(np.mean(recall_list))
            loss_dict['far'].append(np.mean(far_list))
            loss_dict['precision'].append(np.mean(precision_list))
            loss_dict['accuracy'].append(np.mean(accuracy_list))
            loss_dict['F1'].append(np.mean(F1_list))
            loss_dict['TPR'].append(np.mean(TPR_list))
            loss_dict['FPR'].append(np.mean(FPR_list))
            loss_dict["DI"].append(np.mean(DI_list))
            if (np.mean(val_accuracy)*100 > val_acc_max) and (np.mean(spd) < spd_min):
                torch.save({
                    'model1_state_dict': model.state_dict(),
                    'model2_state_dict': S_classifier.state_dict(),
                }, 'model/{}_{:.0f}_{:.0f}_{}.pt'.format(config.dataset_name, np.mean(val_accuracy)*100, np.mean(spd)*100, epoch))
                val_acc_max = np.mean(val_accuracy)*100
                spd_min = np.mean(spd)
        if epoch == config.train_epoch - 1:
            torch.save(loss_dict, "data/loss_dict.pt")
