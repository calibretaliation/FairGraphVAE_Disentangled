import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import sys,os
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y

class GraphVAE(nn.Module):
    '''
    Notations:
        X: Nodes features, shape (n,k) where n is num_nodes, k is num_dimension
        A: Adjacency Matrix, shape (n,n) but can be unsqueeze to (n*n) vector
        Y: Node label, shape (n,1)
        u_S: latent representation of S (sensitive attribute), shape (latent_dim_S,1), sample by reparameterize from mu_S (latent_dim_S,1) and logvar_S (latent_dim_S,1)
        u_Y: latent representation of Y (graph information), shape (latent_dim_Y,1), sample by reparameterize from mu_Y (latent_dim_Y,1) and logvar_Y (latent_dim_Y,1)
    '''
    def __init__(self, num_nodes, num_feats, latent_dim_S, latent_dim_Y, gcn_hidden_dim, num_labels, device="cuda:2", pool="attention"):
        super(GraphVAE, self).__init__()
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.latent_dim_S = latent_dim_S
        self.latent_dim_Y = latent_dim_Y
        self.device = torch.device(device)
        self.gcn_hidden_dim = gcn_hidden_dim
        self.pool = pool
        self.num_labels = num_labels


        # u_S encoder
        self.graph_conv1_S = GraphConv(input_dim=num_feats, output_dim=self.gcn_hidden_dim)
        self.bn1_S = nn.BatchNorm1d(num_feats)
        self.graph_conv2_S = GraphConv(input_dim=self.gcn_hidden_dim, output_dim=self.gcn_hidden_dim)
        self.bn2_S = nn.BatchNorm1d(num_feats)
        self.fc_mu_S = nn.Linear(num_nodes, latent_dim_S)
        self.fc_logvar_S = nn.Linear(num_nodes, latent_dim_S)

        # u_Y encoder
        self.graph_conv1_Y = GraphConv(input_dim=num_feats+1, output_dim=self.gcn_hidden_dim)
        self.bn1_Y = nn.BatchNorm1d(num_feats+1)
        self.graph_conv2_Y = GraphConv(input_dim=self.gcn_hidden_dim, output_dim=self.gcn_hidden_dim)
        self.bn2_Y = nn.BatchNorm1d(num_feats+1)
        self.fc_mu_Y = nn.Linear(num_nodes, latent_dim_Y)
        self.fc_logvar_Y = nn.Linear(num_nodes, latent_dim_Y)
        self.relu = nn.ReLU()

        #A decoder
        self.num_edges= self.num_nodes * (self.num_nodes + 1) // 2
        self.fc1_A = nn.Linear(latent_dim_S+latent_dim_Y, 512)
        self.fc2_A = nn.Linear(512, self.num_edges)

        # X decoder
        self.fc1_X= nn.Linear(latent_dim_S+latent_dim_Y, 512)
        self.fc2_X = nn.Linear(512, self.num_nodes* self.num_feats)

        # Y decoder not using X and A:
        self.fc1_Y = nn.Linear(latent_dim_Y, 512)
        self.fc2_Y = nn.Linear(512, self.num_nodes * self.num_labels)
        self.sigmoid = nn.Sigmoid()

        # Y decoder using X and A
        self.graph_conv3_Y = GraphConv(input_dim=self.num_feats + latent_dim_Y, output_dim=self.num_labels)
    def pool_graph(self, x):
        '''
        Pool graph representation from n*hidden_size to n
        Args:
            x: n*hidden_size matrix

        Returns: (n,1) matrix

        '''
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif self.pool == 'attention':
            attention_weights = torch.nn.Parameter(torch.randn(self.gcn_hidden_dim))
            # Calculate attention scores and apply softmax
            attention_scores = F.softmax(x @ attention_weights, dim=1)  # Shape: (n*, 1)
            # Multiply the attention scores by the original features to perform weighted sum pooling
            out = (x * attention_scores.unsqueeze(-1)).sum(dim=1, keepdim=True)
        return out

    def encode_u_S(self, X, A):
        '''
        Learn the latent representation of sensitive attribute S, then generate incomplete S and also contribute fairness to prediction of A and X
        Args:
            X:
            A:

        Returns:
        '''
        x = self.graph_conv1_S(X, A)
        x = self.bn1_S(x)
        x = self.graph_conv2_S(x, A)
        x = self.bn2_S(x)
        x = self.relu(x)
        x = self.pool_graph(x)
        mu_S = self.fc_mu_S(x)
        logvar_S = self.fc_logvar_S(x)
        u_S = self.reparameterize(mu_S, logvar_S)
        return mu_S, logvar_S, u_S

    def encode_u_Y_using_Linear(self, X, A, Y):
        '''
        Traditional VAE or GraphVAE, which encode X,A,Y to learn the distribution of graph then predict X,A,Y for reconstruction using decode_X, decode_A and decode_Y
        Args:
            X: Nodes features, shape (n,k) where n is num_nodes, k is num_dimension
            A: Adjacency Matrix, shape (n,n) but can be unsqueezed to (n*n) vector
            Y: Node label, shape (n,1)

        Returns:
        '''
        feat = torch.cat((X, Y), dim=1)
        x = self.graph_conv1_Y(feat, A)
        x = self.bn1_Y(x)
        x = self.graph_conv2_Y(x, A)
        x = self.bn2_Y(x)
        x = self.relu(x)
        x = self.pool_graph(x)
        mu_Y = self.fc_mu_Y(x)
        logvar_Y = self.fc_logvar_Y(x)
        u_Y = self.reparameterize(mu_Y, logvar_Y)
        return mu_Y, logvar_Y, u_Y

    def encode_u_Y(self, X, A, Y):
        '''
        Traditional VAE or GraphVAE, which encode X,A,Y to learn the distribution of graph then predict X,A,Y for reconstruction using decode_X, decode_A and decode_Y
        Args:
            X: Nodes features, shape (n,k) where n is num_nodes, k is num_dimension
            A: Adjacency Matrix, shape (n,n) but can be unsqueezed to (n*n) vector
            Y: Node label, shape (n,1)

        Returns:
        '''
        feat = torch.cat((X, Y), dim=1)
        x = self.graph_conv1_Y(feat, A)
        x = self.bn1_Y(x)
        x = self.graph_conv2_Y(x, A)
        x = self.bn2_Y(x)
        x = self.relu(x)
        x = self.pool_graph(x)
        mu_Y = self.fc_mu_Y(x)
        logvar_Y = self.fc_logvar_Y(x)
        u_Y = self.reparameterize(mu_Y, logvar_Y)
        return mu_Y, logvar_Y, u_Y

    def decode_A(self, u_S, u_Y):
        '''
        Construct adjacency matrix based on the latent variable of u_Y (graph) and u_S (sensitive attribute) by predicting edge probability
        Args:
            u_S:
            u_Y:

        Returns:
        '''
        feat = torch.cat((u_S, u_Y), dim=0)
        l = self.fc1_A(feat)
        l = self.fc2_A(l)
        A = self.recover_adj_lower(l)
        return A

    def decode_X(self, u_S, u_Y, A):
        '''
        Construct node features based on the latent variable of u_Y (graph) and u_S (sensitive attribute), then used with generated A to match the generated graph structure
        Args:
            u_X:
            A:
            u_Y:

        Returns:
        '''
        feat = torch.cat((u_S, u_Y), dim=0)
        X = self.fc1_A(feat)
        X = self.fc2_A(X)
        X = torch.matmul(torch.diag(A),X)
        return X

    def decode_Y_from_u_S_using_GCN(self, u_Y):
        '''
        Predict generated node label from u_Y (graph representation) and original A and X
        Args:
            X:
            A:
            u_Y:

        Returns:
        '''
        Y = self.fc1_Y(u_Y)
        Y = self.fc2_Y(Y)
        Y = self.sigmoid(Y)
        return Y

    def decode_Y_from_u_S_X_A(self, u_Y, A, X):
        '''
        Predict generated node label from u_Y (graph representation) and original A and X
        Args:
            X:
            A:
            u_Y:

        Returns:
        '''
        u_Y = u_Y.view(1, len(u_Y))
        X = torch.cat((X, u_Y*self.num_nodes), dim = 1)
        Y = self.graph_conv3_Y(X, A)
        Y = self.sigmoid(Y)
        return Y
    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        #l: flatten vector of the upper triangular part of the adjacency matrix of a graph
        #for example matrix 3x3: l =[a00, a01, a02, a11, a12, a22]
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes) #adjacency matrix
        # torch.triu: create upper triangular matrix filled with ones
        #adj[..] = 1: replace the ones with value of l
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj
    def loss_function(self):
        return None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X, A, Y):
        mu_S, logvar_S, u_S = self.encode_u_S(X,A)
        mu_Y, logvar_Y, u_Y = self.encode_u_Y(X,A,Y)

        A_new = self.decode_A(u_S, u_Y)
        X_new = self.decode_X(u_S, u_Y, A_new)
        Y_new = self.decode_Y(u_Y, A, X)
        return X_new, A_new, Y_new