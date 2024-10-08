import logging
import random
import time

import numpy as np
import torch
from torch import optim, sort, nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score, auc
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from data import construct_A_from_edge_index, plot_loss_dict
from config import config
from metric import get_counts
from model import GraphVAE, F_1_U_S, F_2_Y, hgr_correlation, efl, S_classify, S_recon_loss, Y_prime_decoder, loss_function
import matplotlib.pyplot as plt
from torchviz import make_dot
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
plt.ion()
plt.figure(figsize=(20, 15))
def evaluate(model, val_idx, data, S_true, S_idx):
    with torch.no_grad():
        data = data.to(config.device)
        X = data.x.float()
        edge_index = data.edge_index.int()
        Y = data.y.to(config.device).long()
        Y[Y != 1] = 0
        X = Variable(X).to(config.device)
        edge_index = Variable(edge_index).to(config.device)

        mu_S, logvar_S, mu_Y, logvar_Y, u_S, u_Y, A_new, l, X_new, Y_new, S_new, S_logits = model(X, edge_index, Y, val_idx)

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
def hook_fn(module, grad_input, grad_output):
    print(f"Grad Input: {torch.mean(grad_input[0])}, Grad Output: {torch.mean(grad_output[0])}")

def train(loader, total_data):
    handler = logging.FileHandler(f'log/train_{config.dataset_name}.log')
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    logger.propagate = False

    logger.info(f"Training - Start training:\nDataset:\t{config.dataset_name}\nNum Epoch:\t{config.train_epoch}")
    logger.info(f"Training - Initializing model...")
    config.show(logger)
    model = GraphVAE(config).to(config.device)

    # for name, param in model.named_parameters():
    #     print(name, len(param))

    f1_u_S = F_1_U_S(config).to(config.device)
    f2_Y = F_2_Y(config).to(config.device)
    # S_classifier = S_classify(config).to(config.device)
    Y_classifier = Y_prime_decoder(config).to(config.device)
    optimizer_min = optim.Adam(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_min = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_min, T_max=200, eta_min=1e-5)
    optimizer_max = optim.Adam(list(f1_u_S.parameters()) + list(f2_Y.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_max = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, T_max=200, eta_min=1e-5)
    model.train()
    f1_u_S.train()
    f2_Y.train()
    # S_classifier.train()
    Y_classifier.train()
    losses = []
    losses_max = []
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0)
    # torch.nn.utils.clip_grad_norm_(list(f1_u_S.parameters()) + list(f2_Y.parameters()), max_norm=1.0)
    loss_dict = {
                "step":[],
                "min_phase_loss":[],
                "s_recon_loss":[],
                "efl_term":[],
                "Y_recon_loss":[],
                "X_recon_loss":[],
                "A_recon_loss":[],
                "S_accuracy": [],
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
                 "ROC":[],
                 }
    val_acc_max = 70
    spd_min = 1e8
    logger.info(f"Training - Start training...")
    step = 0
    total_step = config.train_epoch*len(loader)
    min_temperature = 0.1
    initial_temperature = 1
    anneal_rate = 0.0001
    for epoch in tqdm(range(config.train_epoch), position = 0, desc="Train Epoch", leave=False, colour="green"):
        logger.info(f"Training - Loading subgraphs...")
        for graph in tqdm(loader, position = 1, desc="Subgraphs", leave=True, colour="red"):
            config.gumbel_temp = max(min_temperature, initial_temperature * np.exp(-anneal_rate * epoch))
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

            # S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx)*0.67)), dtype = torch.int32).to(config.device)
            # logger.info(f"Training - Graph: S_idx: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

            # S_true = graph.s.float()[S_idx]
            logger.info(f"Training - Done loading subgraph in: {time.time() - start_subgraph}s")
            logger.info(f"Training - Min phase:")
            # Min phase
            mu_S, logvar_S, mu_Y, logvar_Y, u_S, u_Y, A_new, l, X_new, Y_pred, S_hat, S_logits = model(X, edge_index, Y, train_idx)

            elbo, Y_recon_loss, X_recon_loss, A_recon_loss, S_decode_loss, kl_u_y, kl_u_s, hgr_term = loss_function(config, X_new, Y_pred, S_hat,
                                                                                                          X, Y, graph.s.float(),
                                                                                                          logvar_Y,
                                                                                                          mu_Y,
                                                                                                          logvar_S,
                                                                                                          mu_S,
                                                                                                          edge_index, l,
                                                                                                          u_S, u_Y, train_idx)
            Y_pred = Y_pred.argmax(dim=1).unsqueeze(1)[train_idx]
            Y_prime = Y_classifier(X, edge_index)
            Y_prime_loss = nn.MSELoss()(Y_prime[train_idx].float(), Y_pred[train_idx].float())
            # S_hat = S_classifier(u_S, edge_index)
            # s_recon_loss = S_recon_loss(S_hat[S_idx], S_true)
            # S_pred = (S_hat > 0.5).float().cpu().numpy()
            S_pred  = S_hat.argmax(dim=1).unsqueeze(1).cpu().numpy()
            S_accuracy = accuracy_score(graph.s.float().cpu().numpy(), S_pred)
            efl_term = -abs(config.efl_gamma * efl(S_logits[train_idx], Y_prime))

            loss = (elbo + efl_term
                    # + s_recon_loss
                    + Y_prime_loss)
            # s_recon_loss.backward()
            # Y_prime_loss.backward()
            loss_dict["min_phase_loss"].append(loss.data.cpu().numpy())
            loss_dict["s_recon_loss"].append(S_decode_loss.data.cpu().numpy())
            loss_dict["efl_term"].append(efl_term.data.cpu().numpy())
            loss_dict["Y_recon_loss"].append(Y_recon_loss.data.cpu().numpy())
            loss_dict["X_recon_loss"].append(X_recon_loss.data.cpu().numpy())
            loss_dict["A_recon_loss"].append(A_recon_loss.data.cpu().numpy())
            loss_dict["kl_u_y"].append(kl_u_y.data.cpu().numpy())
            loss_dict["kl_u_s"].append(kl_u_s.data.cpu().numpy())
            loss_dict["hgr_term_min_phase"].append(hgr_term.data.cpu().numpy())
            loss_dict["S_accuracy"].append(S_accuracy)
            loss_dict["step"].append(step)
            losses.append(loss.data.cpu().numpy())
            logger.info(f"Training - Training loss min phase:\t{loss.data.cpu().numpy()}")
            loss.backward()
            # if step % 1000 == 0:
            #     plot_grad_flow(model.named_parameters())

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

            if (step % config.log_epoch == 0) and (step != 0):
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
                    indices = random.sample(range(len(loader)), int(len(loader) / 100)+1)
                    subset = Subset(total_data, indices)
                    val_loader = DataLoader(subset, batch_size=1, shuffle=False)
                else:
                    val_loader = loader
                for val_graph in tqdm(val_loader, desc="Validating", leave=True, colour="white"):
                    val_idx = val_graph.val_mask.nonzero(as_tuple=False).view(-1)
                    train_idx = val_graph.train_mask.nonzero(as_tuple=False).view(-1)
                    S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx) * 1)),
                                         dtype=torch.int32).to(config.device)
                    S_true = val_graph.s.float().to(config.device)[S_idx]

                    logger.info(f"Evaluating - Graph: {val_graph}")
                    logger.info(f"Evaluating - Graph: val_mask: {len(val_idx)}, from {val_idx.cpu().numpy().min()} to {val_idx.cpu().numpy().max()}")
                    logger.info(f"Evaluating - Graph: S_idx val: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

                    val_accuracy, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI, roc_auc = evaluate(model, val_idx, val_graph, S_true, S_idx)
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
                    "\nStep: [{}/{}] - {}: avg min phase: {:.4f} avg max phase: {:.4f} val accuracy: {:.4f} spd: {:.4f} f1: {:.4f} EOD: {:.4f} ROC: {:.4f} S_accuracy: {:.4f}".format(
                        step, total_step, config.dataset_name, avg_train_loss, avg_train_loss_max, np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, np.mean(F1_list)*100, np.mean(eod_list)*100, np.mean(roc_auc_list), S_accuracy))

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
                            # 'model2_state_dict': S_classifier.state_dict(),
                        }, 'model/{}_{:.0f}_{:.0f}_{}.pt'.format(config.dataset_name, np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, epoch))
                        val_acc_max = np.mean(val_accuracy_list)*100
                        spd_min = np.mean(spd_list)
                logger.info("Evaluating - Done evaluating, saving loss statistics...")
                torch.save(loss_dict, f"data/loss_dict_{config.dataset_name}_{(step//5000)*5000}.pt")
            # if step%5000 == 0:
            #     plt.savefig('gradient_analysis.png',bbox_inches='tight')
            step += 1


def grid_search_train(loader, total_data, trial):
    handler = logging.FileHandler(f'log/train_{config.dataset_name}.log')
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    logger.propagate = False

    logger.info(f"Training (Grid Search) - Start training:\nDataset:\t{config.dataset_name}\nNum Epoch:\t{config.train_epoch}")
    logger.info(f"Training (Grid Search) - Initializing model...")
    config.show()
    model = GraphVAE(config).to(config.device)
    f1_u_S = F_1_U_S(config).to(config.device)
    f2_Y = F_2_Y(config).to(config.device)
    Y_classifier = Y_prime_decoder(config).to(config.device)
    optimizer_min = optim.Adam(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_min = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_min, T_max=200, eta_min=1e-5)
    optimizer_max = optim.Adam(list(f1_u_S.parameters()) + list(f2_Y.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler_max = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, T_max=200, eta_min=1e-5)
    model.train()
    f1_u_S.train()
    f2_Y.train()
    Y_classifier.train()
    losses = []
    losses_max = []
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(list(f1_u_S.parameters()) + list(f2_Y.parameters()), max_norm=1.0)
    logger.info(f"Training (Grid Search) - Start training...")
    step = 0
    total_step = config.grid
    min_temperature = 0.1
    initial_temperature = 1
    anneal_rate = 0.0001
    loss_dict = {
        "step": [],
        "min_phase_loss": [],
        "s_recon_loss": [],
        "efl_term": [],
        "Y_recon_loss": [],
        "X_recon_loss": [],
        "A_recon_loss": [],
        "S_accuracy": [],
        "kl_u_y": [],
        "kl_u_s": [],
        "hgr_term_min_phase": [],
        "max_phase_loss": [],
        "val_accuracy": [],
        "spd": [],
        'aod': [],
        'eod': [],
        'recall': [],
        'far': [],
        'precision': [],
        'accuracy': [],
        'F1': [],
        'TPR': [],
        'FPR': [],
        "DI": [],
        "ROC": [],
    }
    for epoch in tqdm(range(config.train_epoch), position = 0, desc="Train Epoch", leave=False, colour="green"):
        logger.info(f"Training (Grid Search) - Loading subgraphs...")
        for graph in tqdm(loader, position = 1, desc="Subgraphs", leave=True, colour="red"):
            config.gumbel_temp = max(min_temperature, initial_temperature * np.exp(-anneal_rate * epoch))
            start_subgraph = time.time()
            logger.info(f"Training (Grid Search) - Graph: {graph}")
            graph = graph.to(config.device)

            max_node_idx = graph.edge_index.max().item()
            num_nodes = graph.num_nodes
            if max_node_idx >= num_nodes:
                print(f"Error: edge_index contains node index {max_node_idx} which exceeds num_nodes {num_nodes}")

            train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
            logger.info(f"Training (Grid Search) - Graph: train_mask: {len(train_idx)}, from {train_idx.min().item()} to {train_idx.max().item()}")

            X = graph.x.float()
            edge_index = graph.edge_index
            Y = graph.y.float()

            S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx)*0.67)), dtype = torch.int32).to(config.device)
            logger.info(f"Training (Grid Search) - Graph: S_idx: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

            # S_true = graph.s.float()[S_idx]
            logger.info(f"Training (Grid Search) - Done loading subgraph in: {time.time() - start_subgraph}s")
            logger.info(f"Training (Grid Search) - Min phase:")
            # Min phase
            mu_S, logvar_S, mu_Y, logvar_Y, u_S, u_Y, A_new, l, X_new, Y_pred, S_hat, S_logits = model(X, edge_index, Y,
                                                                                                       train_idx)

            elbo, Y_recon_loss, X_recon_loss, A_recon_loss, S_decode_loss, kl_u_y, kl_u_s, hgr_term = loss_function(
                                                                                                                    config, X_new, Y_pred, S_hat,
                                                                                                                    X, Y, graph.s.float(),
                                                                                                                    logvar_Y,
                                                                                                                    mu_Y,
                                                                                                                    logvar_S,
                                                                                                                    mu_S,
                                                                                                                    edge_index, l,
                                                                                                                    u_S, u_Y, train_idx)
            Y_pred = Y_pred.argmax(dim=1).unsqueeze(1)[train_idx]
            # S_hat = S_classifier(u_S, edge_index)
            # s_recon_loss = S_recon_loss(S_hat[S_idx], S_true)
            # S_pred = (S_hat > 0.5).float().cpu().numpy()
            S_pred = S_hat.argmax(dim=1).unsqueeze(1).cpu().numpy()
            S_accuracy = accuracy_score(graph.s.float().cpu().numpy(), S_pred)
            efl_term = -abs(config.efl_gamma * efl(S_logits[train_idx], Y_pred))
            Y_prime = Y_classifier(X, edge_index)
            Y_prime_loss = nn.MSELoss()(Y_pred[train_idx].float(), Y_prime[train_idx].float())

            loss = (elbo + efl_term
                    # + s_recon_loss
                    + Y_prime_loss)

            loss_dict["min_phase_loss"].append(loss.data.cpu().numpy())
            loss_dict["s_recon_loss"].append(S_decode_loss.data.cpu().numpy())
            loss_dict["efl_term"].append(efl_term.data.cpu().numpy())
            loss_dict["Y_recon_loss"].append(Y_recon_loss.data.cpu().numpy())
            loss_dict["X_recon_loss"].append(X_recon_loss.data.cpu().numpy())
            loss_dict["A_recon_loss"].append(A_recon_loss.data.cpu().numpy())
            loss_dict["kl_u_y"].append(kl_u_y.data.cpu().numpy())
            loss_dict["kl_u_s"].append(kl_u_s.data.cpu().numpy())
            loss_dict["hgr_term_min_phase"].append(hgr_term.data.cpu().numpy())
            loss_dict["S_accuracy"].append(S_accuracy)
            loss_dict["step"].append(step)
            losses.append(loss.data.cpu().numpy())
            logger.info(f"Training (Grid Search) - Training loss min phase:\t{loss.data.cpu().numpy()}")
            loss.backward()
            optimizer_min.step()
            optimizer_min.zero_grad()
            scheduler_min.step()

            # Max phase
            logger.info(f"Training (Grid Search) - Max phase:")
            optimizer_max.zero_grad()
            mu, logvar = model.u_S_encoder(X, edge_index)
            u_S = model.reparameterize(mu, logvar)
            f1_us = f1_u_S(u_S)
            f2_y = f2_Y(Y)
            hgr_loss = hgr_correlation(f1_us, f2_y)
            losses_max.append(hgr_loss.data.cpu().numpy())
            loss_dict["max_phase_loss"].append(hgr_loss.data.cpu().numpy())

            logger.info(f"Training (Grid Search) - Training loss max phase:\t{hgr_loss.data.cpu().numpy()}")
            hgr_loss.backward()
            optimizer_max.step()
            scheduler_max.step()

            if (step % config.log_epoch == 0) and (step != 0):
                logger.info(f"Evaluating (Grid Search) - Evaluating - Step: {step}/{total_step}")
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
                logger.info(f"Evaluating (Grid Search) - Inferring...")


                indices = random.sample(range(len(loader)), int(len(loader) / 100)+1)
                subset = Subset(total_data, indices)
                val_loader = DataLoader(subset, batch_size=1, shuffle=False)

                for val_graph in val_loader:
                    val_idx = val_graph.val_mask.nonzero(as_tuple=False).view(-1)
                    train_idx = val_graph.train_mask.nonzero(as_tuple=False).view(-1)
                    S_idx = torch.tensor(random.sample(sorted(train_idx), int(len(train_idx) * 0.67)),
                                         dtype=torch.int32).to(config.device)
                    S_true = val_graph.s.float().to(config.device)[S_idx]

                    logger.info(f"Evaluating (Grid Search) - Graph: {val_graph}")
                    logger.info(f"Evaluating (Grid Search) - Graph: val_mask: {len(val_idx)}, from {val_idx.cpu().numpy().min()} to {val_idx.cpu().numpy().max()}")
                    logger.info(f"Evaluating (Grid Search) - Graph: S_idx val: {len(S_idx)}, from {S_idx.min().item()} to {S_idx.max().item()}")

                    val_accuracy, spd, aod, eod, recall, far, precision, accuracy, F1, TPR, FPR, DI, roc_auc = evaluate(model, val_idx, val_graph, S_true, S_idx)

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
                print(
                    "\nStep: [{}/{}] - {}: avg min phase: {:.4f} avg max phase: {:.4f} val accuracy: {:.4f} spd: {:.4f} f1: {:.4f} EOD: {:.4f} ROC: {:.4f} S_accuracy: {:.4f}".format(
                        step, total_step, config.dataset_name, avg_train_loss, avg_train_loss_max, np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, np.mean(F1_list)*100, np.mean(eod_list)*100, np.mean(roc_auc_list), S_accuracy))

                logger.info("Evaluating (Grid Search) - Done evaluating, saving loss statistics...")
                if step == config.grid:
                    plot_loss_dict(loss_dict, config, trial)
                    return np.mean(val_accuracy_list)*100, np.mean(spd_list)*100, np.mean(F1_list)*100, np.mean(eod_list)*100, np.mean(roc_auc_list), S_accuracy
            step += 1

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().numpy())
            else:
                print(f"{n} gradient is None")
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(len(layers)), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(ymin=0, ymax=1)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
