import math

import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
import torch_geometric.nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import sys,os
from torch_geometric.nn import GCNConv
from scipy.linalg import eigh
from torch_geometric.profile.benchmark import require_grad

from config import Config
from data import generate_sample_graph, construct_A_from_edge_index, recover_adj_lower

class U_S_encoder(nn.Module):
    def __init__(self, config):
        super(U_S_encoder, self).__init__()
        self.conv1 = GCNConv(in_channels=config.num_feats, out_channels=config.gcn_hidden_dim, cached=True)
        self.conv_mu = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_S, cached=True)
        self.conv_logstd = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_S, cached=True)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), F.softplus(self.conv_logstd(x, edge_index))
class U_Y_encoder(nn.Module):
    def __init__(self, config):
        super(U_Y_encoder, self).__init__()
        self.conv1 = GCNConv(in_channels=config.num_feats+1, out_channels=config.gcn_hidden_dim, cached=True)
        self.conv_mu = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_Y, cached=True)
        self.conv_logstd = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_Y, cached=True)

    def forward(self, x, edge_index, Y):
        x = torch.cat((x,Y), dim=1)
        x = self.conv1(abs(x), edge_index).relu()
        return self.conv_mu(x, edge_index), F.softplus(self.conv_logstd(x, edge_index))

class S_decoder(nn.Module):
    def __init__(self, config):
        super(S_decoder, self).__init__()
        self.gconv1_S = GCNConv(in_channels=config.latent_dim_S, out_channels=config.gcn_hidden_dim).to(config.device)
        self.gconv2_S = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=1).to(config.device)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, edge_index):

        x = self.gconv1_S(abs(x), edge_index).relu()
        x = self.gconv2_S(x, edge_index).relu()
        return self.sigmoid(x)

class A_decoder(nn.Module):
    def __init__(self, config):
        super(A_decoder, self).__init__()
        self.config = config
        self.num_edges = config.num_nodes * (config.num_nodes - 1) // 2 + config.num_nodes
        self.fc1_A = nn.Linear(config.num_nodes*(config.latent_dim_S + config.latent_dim_Y), 512, device=torch.device(config.device))
        self.fc2_A = nn.Linear(512, self.num_edges, device=torch.device(config.device))
        self.sigmoid = nn.Sigmoid()
    def forward(self, u_S, u_Y):

        feat = torch.cat((u_S, u_Y), dim=1).flatten()
        l = self.fc1_A(feat).relu()
        l = self.fc2_A(l)

        A = recover_adj_lower(self.sigmoid(l), self.config)
        return A, l
class X_decoder(nn.Module):
    def __init__(self, config):
        super(X_decoder, self).__init__()
        self.gconv1_X = GCNConv(in_channels=config.latent_dim_S+config.latent_dim_Y, out_channels=config.gcn_hidden_dim).to(config.device)
        self.gconv2_X = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.num_feats).to(
            config.device)
        self.softmax = nn.Softmax()

    def forward(self, edge_index, u_S, u_Y):
        latent = torch.cat((u_S, u_Y), dim=1)
        X = self.gconv1_X(abs(latent), edge_index).relu()
        X = self.gconv2_X(X, edge_index)
        return X
class Y_prime_decoder(nn.Module):
    def __init__(self, config):
        super(Y_prime_decoder, self).__init__()
        self.gconv1_Y_prime = GCNConv(in_channels=config.num_feats, out_channels=512).to(config.device)
        self.gconv2_Y_prime = GCNConv(in_channels=512, out_channels=2).to(config.device)
        self.softmax = nn.Softmax()
    def forward(self, X, edge_index):
        Y_prime = self.gconv1_Y_prime(X, edge_index).relu()
        Y_prime = self.gconv2_Y_prime(Y_prime, edge_index)

        return self.softmax(Y_prime)
class Y_decoder(nn.Module):
    def __init__(self, config):
        super(Y_decoder, self).__init__()
        self.gconv1_Y = GCNConv(in_channels=config.num_feats + config.latent_dim_Y, out_channels=512).to(config.device)
        self.gconv2_Y = GCNConv(in_channels=512, out_channels=2).to(config.device)
        self.softmax = nn.Softmax()
    def forward(self, edge_index, X, u_Y):
        latent = torch.cat((u_Y, X), dim=1)
        Y = self.gconv1_Y(latent, edge_index).relu()
        Y = self.gconv2_Y(Y, edge_index)

        return self.softmax(Y)
class GraphVAE(nn.Module):
    '''
    Notations:
        X: Nodes features, shape (n,k) where n is num_nodes, k is num_dimension
        A: Adjacency Matrix, shape (n,n) but can be unsqueeze to (n*n) vector
        Y: Node label, shape (n,1)
        u_S: latent representation of S (sensitive attribute), shape (latent_dim_S,1), sample by reparameterize from mu_S (latent_dim_S,1) and logvar_S (latent_dim_S,1)
        u_Y: latent representation of Y (graph information), shape (latent_dim_Y,1), sample by reparameterize from mu_Y (latent_dim_Y,1) and logvar_Y (latent_dim_Y,1)
    '''
    def __init__(self, config):
        super(GraphVAE, self).__init__()
        self.config = config
        self.u_S_encoder = U_S_encoder(config)
        self.u_Y_encoder = U_Y_encoder(config)
        self.S_decoder = S_decoder(config)
        self.X_decoder = X_decoder(config)
        self.A_decoder = A_decoder(config)
        self.Y_decoder = Y_decoder(config)
        self.Y_prime_decoder = Y_prime_decoder(config)
        self.S_decoder = S_decoder(config)

        for m in self.modules():
            if isinstance(m, torch_geometric.nn.GCNConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)

    def recon_edge_loss(self, l, A):
        A = construct_A_from_edge_index(A, self.config.num_nodes)
        l_original = A[np.triu_indices_from(A)].to(self.config.device)
        edge_loss =nn.MSELoss()(l, l_original)
        return edge_loss
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def loss_function(self,
                      X_pred,
                      Y_pred,
                      X_true,
                      Y_true,
                      Y_prime,
                      logvar_u_Y,
                      mu_u_Y,
                      logvar_u_S,
                      mu_u_S,
                      edge_index,
                      l,
                      u_S,
                      u_Y,
                      idx):


        # S_recon_loss = self.S_recon_loss(S_hat)
        # Y_pred, _ = torch.max(Y_pred[idx], dim=1, keepdim=True)
        # Y_prime, _ = torch.max(Y_prime[idx], dim=1, keepdim=True)
        Y_true[Y_true != 1] = 0
        Y_true = torch.nn.functional.one_hot(Y_true.long(), num_classes=2).squeeze()
        # Y_prime = Y_prime.argmax(dim=1).unsqueeze(1)[idx]
        Y_recon_loss = abs(nn.MSELoss()(Y_pred[idx].float(), Y_true[idx].float())
                        + nn.MSELoss()(Y_pred[idx].float(), Y_prime[idx].float())
                        )

        # 2. Reconstruction Loss for X (log P(X | U_S, A, U_Y))
        X_recon_loss = F.mse_loss(X_pred[idx].float(), X_true[idx].float())

        A_recon_loss = self.recon_edge_loss(l, edge_index)

        # 4. KL Divergence for U_Y (KL(Q(U_Y | X, A, Y) || P(U_Y)))
        kl_u_y = -0.5 * torch.sum(1 + logvar_u_Y - mu_u_Y.pow(2) - logvar_u_Y.exp() + 1e-8)

        # 5. KL Divergence for U_S (KL(Q(U_S | X, A) || P(U_S)))
        kl_u_s = -0.5 * torch.sum(1 + logvar_u_S - mu_u_S.pow(2) - logvar_u_S.exp() + 1e-8)

        # 6. HGR Regularization Term (lambda * HGR(U_S, Y))
        hgr_term = -self.config.lambda_hgr * hgr_correlation(u_S, u_Y)
        # ELBO is the sum of all these terms
        if math.isnan(kl_u_s) or math.isnan(kl_u_y):
            print(f"X_pred : {X_pred}")
            print(f"Y_pred : {Y_pred}")
            print(f"X_true : {X_true}")
            print(f"Y_prime : {Y_prime}")
            print(f"logvar_u_Y : {logvar_u_Y}")
            print(f"mu_u_Y : {mu_u_Y}")
            print(f"logvar_u_S : {logvar_u_S}")
            print(f"mu_u_S : {mu_u_S}")
            print(f"edge_index : {edge_index}")
            print(f"l : {l}")
            print(f"u_S : {u_S}")
            print(f"u_Y : {u_Y}")
            print(f"kl_u_y loss: {kl_u_y}")
            print(f"kl_u_s loss: {kl_u_s}")
            print(f"A_recon_loss loss: {A_recon_loss}")
            print(f"X_recon_loss loss: {X_recon_loss}")
            print(f"Y_recon_loss loss: {Y_recon_loss}")
            exit()
        elbo = (Y_recon_loss + X_recon_loss + A_recon_loss + kl_u_y + kl_u_s
                + hgr_term
                # + efl_term + S_recon_loss
                )

        return elbo, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term
    def forward(self, X, edge_index, Y, idx):

        mu_S, logvar_S = self.u_S_encoder(X,edge_index)
        mu_Y, logvar_Y = self.u_Y_encoder(X,edge_index,Y)
        u_S = self.reparameterize(mu_S,logvar_S)
        u_Y = self.reparameterize(mu_Y,logvar_Y)
        Y_prime = self.Y_prime_decoder(X,edge_index).detach()
        A_new, l = self.A_decoder(u_S, u_Y)
        X_new = self.X_decoder(edge_index, u_S, u_Y)
        Y_new = self.Y_decoder(edge_index, u_Y, X)
        loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term = self.loss_function(X_new, Y_new, X, Y,Y_prime , logvar_Y, mu_Y, logvar_S, mu_S, edge_index, l, u_S, u_Y, idx)
        return (loss, Y_recon_loss, X_recon_loss, A_recon_loss, kl_u_y, kl_u_s, hgr_term,
                Y_new,
                u_S)
class F_1_U_S(nn.Module):
    '''
    Notations:
        X: Nodes features, shape (n,k) where n is num_nodes, k is num_dimension
        A: Adjacency Matrix, shape (n,n) but can be unsqueeze to (n*n) vector
        Y: Node label, shape (n,1)
        u_S: latent representation of S (sensitive attribute), shape (latent_dim_S,1), sample by reparameterize from mu_S (latent_dim_S,1) and logvar_S (latent_dim_S,1)
        u_Y: latent representation of Y (graph information), shape (latent_dim_Y,1), sample by reparameterize from mu_Y (latent_dim_Y,1) and logvar_Y (latent_dim_Y,1)
    '''
    def __init__(self, config):
        super(F_1_U_S, self).__init__()
        self.config = config
        self.fc_1 = nn.Linear(config.latent_dim_S, 512).to(config.device)
        self.fc_2 = nn.Linear(512, 128).to(config.device)
    def forward(self, S):
        S = torch.relu(self.fc_1(S))
        return self.fc_2(S)

class F_2_Y(nn.Module):
    def __init__(self, config):
        super(F_2_Y, self).__init__()
        self.config = config
        self.fc_1 = nn.Linear(1, 512).to(config.device)
        self.fc_2 = nn.Linear(512, 128).to(config.device)
    def forward(self, Y):
        Y = torch.relu(self.fc_1(Y))
        return self.fc_2(Y)


def run_model_debug(config: Config):
    num_nodes = config.num_nodes
    num_feats = config.num_feats
    num_labels = config.num_labels
    # Instantiate the ConvVAE model
    model = GraphVAE(config)

    # Create the sample input
    X, A, Y = generate_sample_graph(num_nodes, num_feats, num_labels)
    edge_index = torch.nonzero(A, as_tuple=False).t()

    # Run one forward pass through the model
    loss, Y_new, u_S = model(X.to(config.device), edge_index.to(config.device), Y.to(config.device))

def hgr_correlation(X, Y, n_components=10):
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

    # Center the data (remove mean)
    std_Y = torch.std(Y)
    std_X = torch.std(X)

    X_centered = (X - torch.mean(X, dim=0))/std_X
    Y_centered = (Y - torch.mean(Y, dim=0))/std_Y
    E_fxgy = torch.mean(X_centered * Y_centered)
    E_fx2 = torch.mean(torch.std(X_centered))
    E_gy2 = torch.mean(torch.std(Y_centered))

    # HGR objective: maximize covariance / sqrt(var_fx * var_gy)
    hgr = E_fxgy / (torch.sqrt(E_fx2 * E_gy2) + 1e-8)
    # Covariance matrix approximation
    return -abs(hgr)
def efl(S_hat, Y_pred):
    p_s1 = S_hat
    p_s0 = 1 - S_hat

    sum_s_1 = torch.sum(p_s1 * Y_pred)  # Sum of positive outcomes for Ŝi = 1 group
    sum_s_0 = torch.sum(p_s0 * Y_pred)  # Sum of positive outcomes for Ŝi = 0 group

    # Normalization terms (the sum of probabilities for each group)
    norm_s_1 = torch.sum(p_s1)
    norm_s_0 = torch.sum(p_s0)

    # Calculate the weighted average of positive outcomes for each group
    avg_s_1 = sum_s_1 / norm_s_1  # Average for Ŝi = 1 group
    avg_s_0 = sum_s_0 / norm_s_0  # Average for Ŝi = 0 group

    # Compute the Estimated Fairness Loss (EFL)
    efl = avg_s_1 - avg_s_0

    return efl
def S_recon_loss(S_hat, S_true):
    # p_s1 = S_hat
    # p_s0 = 1 - S_hat
    #
    # # Compute the entropy loss
    # entropy_loss = - torch.sum(p_s1 * torch.log(p_s1 + 1e-8) + p_s0 * torch.log(p_s0 + 1e-8))
    #
    loss = nn.MSELoss()(S_hat, S_true)
    return loss