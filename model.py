import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import sys,os
from torch_geometric.nn import GCNConv
from scipy.linalg import eigh

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
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
class U_Y_encoder(nn.Module):
    def __init__(self, config):
        super(U_Y_encoder, self).__init__()
        self.conv1 = GCNConv(in_channels=config.num_feats+1, out_channels=config.gcn_hidden_dim, cached=True)
        self.conv_mu = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_Y, cached=True)
        self.conv_logstd = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.latent_dim_Y, cached=True)

    def forward(self, x, edge_index, Y):
        x = torch.cat((x,Y), dim=1)
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class S_decoder(nn.Module):
    def __init__(self, config):
        super(S_decoder, self).__init__()
        self.gconv1_S = GCNConv(in_channels=config.latent_dim_S, out_channels=config.gcn_hidden_dim).to(config.device)
        self.gconv2_S = GCNConv(in_channels=config.gcn_hidden_dim, out_channels=config.num_sensitive_class).to(config.device)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, edge_index):

        x = self.gconv1_S(x, edge_index).relu()
        x = self.gconv2_S(x, edge_index).relu()
        return self.sigmoid(x)

class A_decoder(nn.Module):
    def __init__(self, config):
        super(A_decoder, self).__init__()
        self.config = config
        self.num_edges = config.num_nodes * (config.num_nodes - 1) // 2 + config.num_nodes
        self.fc1_A = nn.Linear(config.num_nodes*(config.latent_dim_S + config.latent_dim_Y), 512, device=torch.device(config.device))
        self.fc2_A = nn.Linear(512, self.num_edges, device=torch.device(config.device))

    def forward(self, u_S, u_Y):

        feat = torch.cat((u_S, u_Y), dim=1).flatten()
        l = self.fc1_A(feat)
        l = self.fc2_A(l)
        A = recover_adj_lower(l, self.config)
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
        X = self.gconv1_X(latent, edge_index)
        X = self.gconv2_X(X, edge_index)
        return X
class Y_prime_decoder(nn.Module):
    def __init__(self, config):
        super(Y_prime_decoder, self).__init__()
        self.gconv1_Y_prime = GCNConv(in_channels=config.num_feats, out_channels=512).to(config.device)
        self.gconv2_Y_prime = GCNConv(in_channels=512, out_channels=config.num_labels).to(config.device)
        self.softmax = nn.Softmax()

    def forward(self, X, edge_index):
        Y_prime= self.gconv1_Y_prime(X, edge_index)
        Y_prime = self.gconv2_Y_prime(Y_prime, edge_index)
        Y_prime = self.softmax(Y_prime)

        return Y_prime
class Y_decoder(nn.Module):
    def __init__(self, config):
        super(Y_decoder, self).__init__()
        self.gconv1_Y = GCNConv(in_channels=config.num_feats + config.latent_dim_Y, out_channels=512).to(config.device)
        self.gconv2_Y = GCNConv(in_channels=512, out_channels=config.num_labels).to(config.device)
        self.softmax = nn.Softmax()

    def forward(self, edge_index, X, u_Y):
        latent = torch.cat((X, u_Y), dim=1)
        Y = self.gconv1_Y(latent, edge_index)
        Y = self.gconv2_Y(Y, edge_index)
        Y = self.softmax(Y)

        return Y
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
        self.bce_loss = nn.BCEWithLogitsLoss()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def recon_edge_loss(self, l, A):
        A = construct_A_from_edge_index(A, self.config.num_nodes)
        l_original = A[np.triu_indices_from(A)]
        edge_loss = self.bce_loss(l, l_original)
        return edge_loss
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def S_recon_loss(self, S_hat):
        p_s1 = S_hat
        p_s0 = 1 - S_hat

        # Compute the entropy loss
        entropy_loss = - torch.sum(p_s1 * torch.log(p_s1) + p_s0 * torch.log(p_s0))

        return entropy_loss

    def efl(self, S_hat, Y_pred):
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
    def hgr_correlation(self, X, Y, n_components=10):
        """
        计算Hirschfeld-Gebelein-Rényi (HGR) 相关性

        参数:
        X: 第一个随机变量，numpy数组或 torch.Tensor
        Y: 第二个随机变量，numpy数组或 torch.Tensor
        n_components: 使用的主成分数量

        返回:
        HGR相关性
        """
        # 确保 X 和 Y 都是张量
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.config)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).to(self.config)
        # 创建张量的副本
        X_copy = X.clone().detach().cpu().numpy()
        Y_copy = Y.clone().detach().cpu().numpy()
        num_samples = 1000

        # 数据中心化
        X_centered = X_copy - np.mean(X_copy)
        Y_centered = Y_copy - np.mean(Y_copy)
        X_centered = X_centered[:num_samples, :]
        Y_centered = Y_centered[:num_samples, :]
        # 计算协方差矩阵
        C = np.cov(X_centered, Y_centered)
        # 奇异值分解
        S, U = eigh(C)

        # 选择前n_components个主成分
        S = S[-n_components:]
        # 计算HGR相关性
        hgr = np.sum(S) / (np.std(X_centered) * np.std(Y_centered))

        # 返回结果
        return hgr
    def loss_function(self, X_pred,
                      Y_pred,
                      X_true,
                      Y_prime,
                      logvar_u_Y,
                      mu_u_Y,
                      logvar_u_S,
                      mu_u_S,
                      edge_index,
                      l,
                      u_S,
                      u_Y,
                      S_hat):

        S_recon_loss = self.S_recon_loss(S_hat)
        prediction = (Y_pred > 0.5).float()
        labels = (Y_prime > 0.5).float()
        Y_pred = prediction.argmax(dim=1).unsqueeze(1)
        Y_prime = labels.argmax(dim=1).unsqueeze(1)
        Y_recon_loss = F.cross_entropy(Y_pred.float(), Y_prime.float())

        # 2. Reconstruction Loss for X (log P(X | U_S, A, U_Y))
        X_recon_loss = F.mse_loss(X_pred.float(), X_true.float())

        A_recon_loss = self.recon_edge_loss(l, edge_index)

        # 4. KL Divergence for U_Y (KL(Q(U_Y | X, A, Y) || P(U_Y)))
        kl_u_y = -0.5 * torch.sum(1 + logvar_u_Y - mu_u_Y.pow(2) - logvar_u_Y.exp())

        # 5. KL Divergence for U_S (KL(Q(U_S | X, A) || P(U_S)))
        kl_u_s = -0.5 * torch.sum(1 + logvar_u_S - mu_u_S.pow(2) - logvar_u_S.exp())

        efl_term = self.efl(S_hat, Y_pred)
        # 6. HGR Regularization Term (lambda * HGR(U_S, Y))
        hgr_term = self.config.lambda_hgr * self.hgr_correlation(u_S, u_Y)

        # ELBO is the sum of all these terms
        elbo = (Y_recon_loss + X_recon_loss + A_recon_loss + kl_u_y + kl_u_s
                + hgr_term + efl_term + S_recon_loss
                )

        return elbo
    def forward(self, X, edge_index, Y):

        mu_S, logvar_S = self.u_S_encoder(X,edge_index)
        mu_Y, logvar_Y = self.u_Y_encoder(X,edge_index,Y)
        u_S = self.reparameterize(mu_S,logvar_S)
        u_Y = self.reparameterize(mu_Y,logvar_Y)
        S_hat = self.S_decoder(u_S, edge_index).detach()
        Y_prime = self.Y_prime_decoder(X,edge_index)
        A_new, l = self.A_decoder(u_S, u_Y)
        X_new = self.X_decoder(edge_index, u_S, u_Y)
        Y_new = self.Y_decoder(edge_index, u_Y, X).detach()
        return (self.loss_function(X_new, Y_new, X, Y_prime , logvar_Y, mu_Y, logvar_S, mu_S, edge_index, l, u_S, u_Y, S_hat),
                Y_new,
                S_hat)

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
    loss, Y_new, S_hat = model(X.to(config.device), edge_index.to(config.device), Y.to(config.device))
