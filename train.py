import logging
import random
import time

import numpy as np
import torch
from torch import optim, sort
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score, auc
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from data import construct_A_from_edge_index
from config import Config
from metric import get_counts
from model import GraphVAE, F_1_U_S, F_2_Y, hgr_correlation, efl, S_decoder, S_recon_loss
import matplotlib.pyplot as plt
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

def evaluate(model, val_idx, config, data, S_true, S_idx):
    dataset = config.dataset_name
    data = data.to(config.device)
    X = data.x.float()
    edge_index = data.edge_index.int()
    Y = data.y.to(config.device).long()
    Y[Y != 1] = 0
    X = Variable(X).to(config.device)
    edge_index = Variable(edge_index).to(config.device)

    loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_new, u_S = model(X, edge_index, Y, val_idx)

    S_idx = S_idx.cpu()
    y_scores = Y_new[:, 1]
    y_true = Y.view(-1)
    y_scores_np = y_scores.data.cpu().numpy()
    y_true_np = y_true.data.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np, drop_intermediate=False)

    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    roc_auc = auc(fpr, tpr)
    # roc_auc = auc(y_true_np, y_scores_np)

    all_preds = Y_new.data.argmax(dim=1).cpu().numpy()
    all_y = Y.squeeze().cpu().numpy()
    logger.info(f"Evaluating - PREDICTION\nGround-Truth:\t{np.unique(all_y, return_counts=True)}\nPrediction:\t{np.unique(all_preds, return_counts=True)}")
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

    logger.info(f"Evaluating - EVALUATE RESULT\nscore:\t{score}\nspd:\t{spd}\naod:\t{aod}\neod:\t{eod}\nrecall:\t{recall}\nfar:\t{far}\nprecision:\t{precision}\naccuracy:\t{accuracy}\nF1:\t{F1}\nTPR:\t{TPR}\nFPR:\t{FPR}\nDI:\t{DI}\nroc_auc:\t{roc_auc}")

    return score, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI, roc_auc

def train(config: Config, loader, total_data):
    handler = logging.FileHandler(f'log/train_{config.dataset_name}.log')
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    logger.propagate = False


    dataset = config.dataset_name
    logger.info(f"Training - Start training:\nDataset:\t{config.dataset_name}\nNum Epoch:\t{config.train_epoch}")
    logger.info(f"Training - Initializing model...")
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
                 "DI":[],
                 "ROC":[]
                 }
    val_acc_max = 70
    spd_min = 1e8
    logger.info(f"Training - Start training...")
    step = 0
    total_step = config.train_epoch*len(loader)
    for epoch in tqdm(range(config.train_epoch), position = 0, desc="Train Epoch", leave=False, colour="green"):
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
        logger.info(f"Training - Loading subgraphs...")
        for graph in tqdm(loader, position = 1, desc="Subgraphs", leave=True, colour="red"):
            start_subgraph = time.time()
            logger.info(f"Training - Graph: {graph}")
            graph = graph.to(config.device)

            max_node_idx = graph.edge_index.max().item()
            num_nodes = graph.num_nodes
            if max_node_idx >= num_nodes:
                print(f"Error: edge_index contains node index {max_node_idx} which exceeds num_nodes {num_nodes}")

            train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
            logger.info(f"Training - Graph: train_mask: {len(train_idx)}, from {train_idx.min().item()} to {train_idx.max().item()}")

            X = graph.x.float()
            edge_index = graph.edge_index
            Y = graph.y.float()

            S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx)*0.67)), dtype = torch.int32).to(config.device)
            logger.info(f"Training - Graph: S_idx: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

            S_true = graph.s.float()[S_idx]
            logger.info(f"Training - Done loading subgraph in: {time.time() - start_subgraph}s")
            logger.info(f"Training - Min phase:")
            # Min phase
            loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term, Y_pred, u_S = model(X, edge_index, Y, train_idx)
            Y_pred = Y_pred.argmax(dim=1).unsqueeze(1)[train_idx]
            S_hat = S_classifier(u_S, edge_index).detach()
            s_recon_loss = S_recon_loss(S_hat[S_idx], S_true)
            efl_term = -abs(config.efl_gamma * efl(S_hat[train_idx], Y_pred))
            loss = loss + s_recon_loss + efl_term

            # loss_list.append(loss.data.cpu().numpy())
            # s_recon_loss_list.append(s_recon_loss.data.cpu().numpy())
            # efl_term_list.append(efl_term.data.cpu().numpy())
            # Y_recon_loss_list.append(Y_recon_loss.data.cpu().numpy())
            # X_recon_loss_list.append(X_recon_loss.data.cpu().numpy())
            # A_recon_loss_list.append(A_recon_loss.data.cpu().numpy())
            # kl_u_y_list.append(kl_u_y.data.cpu().numpy())
            # kl_u_s_list.append(kl_u_s.data.cpu().numpy())
            # hgr_term_list.append(hgr_term.data.cpu().numpy())
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
            logger.info(f"Training - Training loss min phase:\t{loss.data.cpu().numpy()}")
            loss.backward()
            optimizer_min.step()
            optimizer_min.zero_grad()
            scheduler_min.step()

            # Max phase
            logger.info(f"Training - Max phase:")
            optimizer_max.zero_grad()
            mu, logvar = model.u_S_encoder(X, edge_index)
            u_S = model.reparameterize(mu, logvar)
            f1_us = f1_u_S(u_S)
            f2_y = f2_Y(Y)
            hgr_loss = hgr_correlation(f1_us, f2_y)
            loss_dict["max_phase_loss"].append(hgr_loss.data.cpu().numpy())
            losses_max.append(hgr_loss.data.cpu().numpy())
            logger.info(f"Training - Training loss max phase:\t{hgr_loss.data.cpu().numpy()}")
            hgr_loss.backward()
            optimizer_max.step()
            scheduler_max.step()

            # print('Epoch: ', epoch,', Loss: ', loss)
            # loss_dict["min_phase_loss"].append(np.mean(loss_list))
            # loss_dict["s_recon_loss"].append(np.mean(s_recon_loss_list))
            # loss_dict["efl_term"].append(np.mean(efl_term_list))
            # loss_dict["Y_recon_loss"].append(np.mean(Y_recon_loss_list))
            # loss_dict["X_recon_loss"].append(np.mean(X_recon_loss_list))
            # loss_dict["A_recon_loss"].append(np.mean(A_recon_loss_list))
            # loss_dict["kl_u_y"].append(np.mean(kl_u_y_list))
            # loss_dict["kl_u_s"].append(np.mean(kl_u_s_list))
            # loss_dict["hgr_term_min_phase"].append(np.mean(hgr_term_list))
            # loss_dict["max_phase_loss"].append(np.mean(hgr_loss_list))

            if ((step) % config.log_epoch == 0) and (step != 0):
                logger.info(f"Evaluating - Evaluating - Step: {step}/{total_step}")
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
                roc_auc_list = []
                logger.info(f"Evaluating - Inferring...")
                if step != total_step - 1:
                    indices = random.sample(range(config.num_nodes), int(config.num_nodes / 100)+1)
                    subset = Subset(total_data, indices)
                    val_loader = DataLoader(subset, batch_size=1, shuffle=False)
                else:
                    val_loader = loader
                for graph in val_loader:
                    val_idx = graph.val_mask.nonzero(as_tuple=False).view(-1)
                    train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
                    S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx) * 0.67)),
                                         dtype=torch.int32).to(config.device)
                    S_true = graph.s.float().to(config.device)[S_idx]

                    logger.info(f"Evaluating - Graph: {graph}")
                    logger.info(f"Evaluating - Graph: val_mask: {len(val_idx)}, from {val_idx.min().item()} to {val_idx.max().item()}")
                    logger.info(f"Evaluating - Graph: S_idx val: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

                    val_accuracy, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI, roc_auc = evaluate(model, val_idx, config, graph, S_true, S_idx)
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
                    roc_auc_list.append(roc_auc)
                print(
                    "\nEpoch: [{}/{}]: avg min phase: {:.4f} avg max phase: {:.4f} val accuracy: {:.4f} spd: {:.4f} f1: {:.4f} EOD: {:.4f} ROC: {:.4f}".format(
                        step, total_step, avg_train_loss, avg_train_loss_max, np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, np.mean(F1_list)*100, np.mean(eod_list)*100, np.mean(roc_auc_list)))

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
                loss_dict["ROC"].append(np.mean(roc_auc_list))
                if (np.mean(spd_list)*100 != 0) or (epoch != 0):
                    if (np.mean(val_accuracy_list)*100 > val_acc_max) and (np.mean(spd_list) < spd_min):
                        logger.info(f"FOUND BEST MODEL:\nEpoch:\t{epoch}\nAccuracy:\t{np.mean(val_accuracy_list)*100}\nSPD:\t{np.mean(spd_list)*100}")
                        torch.save({
                            'model1_state_dict': model.state_dict(),
                            'model2_state_dict': S_classifier.state_dict(),
                        }, 'model/{}_{:.0f}_{:.0f}_{}.pt'.format(config.dataset_name, np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, epoch))
                        val_acc_max = np.mean(val_accuracy_list)*100
                        spd_min = np.mean(spd_list)
                logger.info("Evaluating - Done evaluating, saving loss statistics...")
            torch.save(loss_dict, f"data/loss_dict_{config.dataset_name}.pt")
            step += 1
