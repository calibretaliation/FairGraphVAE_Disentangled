from scipy.io import loadmat

from config import Config
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def generate_sample_graph(num_nodes, num_feats, num_labels):
    X = np.random.rand(num_nodes, num_feats)
    normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = torch.tensor(normalized_X, dtype=torch.float32)
    Y = torch.tensor(np.random.randint(num_labels+1, size=(num_nodes, 1)))
    upper_tri = np.triu(np.random.randint(2, size=(num_nodes, num_nodes)), 1)
    A = upper_tri + upper_tri.T
    for i in range(num_nodes):
        A[i, i] = 1
    A = torch.tensor(A, dtype=torch.float32)
    return X, A, Y

def read_graphs(path, type = "mat"):
    if type == "mat":
        data = loadmat(path)
class GraphDataset(Dataset):
    def __init__(self, graph_data):
        """
        Args:
            graph_data (list): A list of tuples, where each tuple represents a graph.
                               Each tuple contains:
                               - adjacency matrix (torch.Tensor of shape (n, n))
                               - node features (torch.Tensor of shape (n, k))
                               - node labels (torch.Tensor of shape (n, 1))
        """
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]


class GraphDataset(Dataset):
    def __init__(self, graph_data):
        """
        Args:
            graph_data (list): A list of tuples, where each tuple represents a graph.
                               Each tuple contains:
                               - adjacency matrix (torch.Tensor of shape (n, n))
                               - node features (torch.Tensor of shape (n, k))
                               - node labels (torch.Tensor of shape (n, 1))
        """
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]

def collate_graphs(batch):
    """
    Custom collate function to handle variable-sized graphs and create a batch.

    Args:
        batch (list): A list of tuples, where each tuple contains:
                      - adjacency matrix of shape (n, n)
                      - node features of shape (n, k)
                      - node labels of shape (n, 1)

    Returns:
        A batch of adjacency matrices, node features, and node labels, all padded to the
        size of the largest graph in the batch.
    """
    # Find the largest number of nodes in the batch
    max_num_nodes = max([graph[0].size(0) for graph in batch])

    adj_matrices = []
    node_features = []
    node_labels = []

    for adj_matrix, features, labels in batch:
        n = adj_matrix.size(0)  # Number of nodes in the graph

        # Pad adjacency matrix to (max_num_nodes, max_num_nodes)
        padded_adj_matrix = F.pad(adj_matrix, (0, max_num_nodes - n, 0, max_num_nodes - n))
        adj_matrices.append(padded_adj_matrix)

        # Pad node features to (max_num_nodes, k)
        padded_features = F.pad(features, (0, 0, 0, max_num_nodes - n))
        node_features.append(padded_features)

        # Pad node labels to (max_num_nodes, 1)
        padded_labels = F.pad(labels, (0, 0, 0, max_num_nodes - n))
        node_labels.append(padded_labels)

    # Stack all the padded tensors to create batch
    adj_matrices = torch.stack(adj_matrices, dim=0)
    node_features = torch.stack(node_features, dim=0)
    node_labels = torch.stack(node_labels, dim=0)

    return adj_matrices, node_features, node_labels


def load_dataset(config: Config, dataset = "generate"):
    assert dataset in ['generate'], \
    "dataset parameter should be one of: ['generate']"
    if dataset == "generate":
        graph_list = []
        for i in range(1000):
            graph_list.append((generate_sample_graph(config.num_nodes, config.num_feats, config.num_labels)))
        dataset = GraphDataset(graph_list)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_graphs)

        return data_loader
dataset = load_dataset(Config(), "generate")
for id, (A,X,Y) in enumerate(dataset):
    print("Adjacency Matrices:", A.shape)
    print("Node Features:", X.shape)
    print("Node Labels:", Y.shape)
    break