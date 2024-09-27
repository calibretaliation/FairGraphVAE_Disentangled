import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import os, sys
from config import Config
import torch
import numpy as np
import torch
import torch.nn.functional as F

def generate_sample_graph(num_nodes, num_feats, num_labels):
    X = np.random.rand(num_nodes, num_feats)
    normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = torch.tensor(normalized_X, dtype=torch.float32)
    Y = torch.tensor(np.random.randint(num_labels, size=(num_nodes, 1)), dtype = torch.float32)
    upper_tri = np.triu(np.random.randint(2, size=(num_nodes, num_nodes)), 1)
    A = upper_tri + upper_tri.T
    for i in range(num_nodes):
        A[i, i] = 1
    A = torch.tensor(A, dtype=torch.float32)
    return X, A, Y
def save_graphs(data, path, format):
    if format == "mat":
        savemat(data,path)
    elif format == "pt":
        torch.save(data,path)
def read_graphs(path, type = "mat"):
    if type == "mat":
        data = loadmat(path)

    elif type == "pt":
        data = torch.load(path)

    return data
class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 返回原始数据文件名列表
        return ['{}.pt'.format(self.dataset_name)]

    @property
    def processed_file_names(self):
        # 返回处理后的数据文件名列表
        return ['processed_{}_graph.pt'.format(self.dataset_name)]

    def download(self):
        # 下载数据，如果需要的话
        pass

    def process(self):
        # Here you load your data and convert it into a Data object
        data_list = []
        for raw_path in self.raw_paths:
            # Load raw data, e.g. from a .pt file
            if self.dataset_name == "generate":
                all_data = read_graphs(raw_path, "pt")
                for graph in all_data:
                    data = Data(x=graph['X'], edge_index=graph['edge_index'], y=graph['Y'], s=graph["S"])
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
            else:
                graph = read_graphs(raw_path, "pt")
                data = Data(x=graph['X'], edge_index=graph['edge_index'], y=graph['Y'], s=graph["S"])
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 将数据存储到磁盘上
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_nba_data(config):
    dataset_name = "nba"
    nba_node_feature = []
    nba_label = []
    nba_sensitive = []
    nba_node_dict = {}
    nba_label_dict = {}
    nba_sensitive_dict = {}
    with open(config.data_path + "/{}".format(dataset_name) + "/nba.csv", "r") as file:
        line = file.readline()
        feature = line.split(",")
        sensitive_feature = feature.index("country")
        line = file.readline()
        label_list = []
        sensitive_list = []
        while line:
            line = line.rstrip("\n")
            line = line.split(",")
            node_id = line[0]
            label = line[1]
            feature = [float(x) for x in line[2:sensitive_feature] + line[sensitive_feature + 1:]]
            sensitive = line[sensitive_feature]
            nba_node_feature.append(feature)
            if node_id not in nba_node_dict.keys():
                count = len(nba_node_dict)
                nba_node_dict.update({node_id: count})
            if sensitive not in nba_sensitive_dict.keys():
                nba_sensitive_dict.update({sensitive: len(sensitive_list)})
                sensitive_list.append(sensitive)
            nba_sensitive.append([nba_sensitive_dict[sensitive]])
            if label not in nba_label_dict.keys():
                nba_label_dict.update({label: len(label_list)})
                label_list.append(label)
            nba_label.append([nba_label_dict[label]])
            line = file.readline()
    nba_node_feature = np.array(nba_node_feature)
    print("NBA DATASET")
    print("node feature:", nba_node_feature.shape)
    print("label :", np.unique(nba_label))
    print("total label :", len(nba_label))
    print("nba_label_dict ", nba_label_dict)
    print("sensitive :", np.unique(nba_sensitive))
    print("total sensitive :", len(nba_sensitive))
    print("nba_sensitive_dict ", nba_sensitive_dict)
    dataset = {}
    network = np.zeros((len(nba_node_dict), len(nba_node_dict)))
    for i in range(len(network)):
        network[i, i] = 1
    with open(config.data_path + "/{}".format(dataset_name) + "/nba_relationship.txt", "r") as file:
        line = file.readline()
        while line:
            line = line.rstrip("\n")
            line = line.split("\t")
            network[nba_node_dict[line[0]], nba_node_dict[line[1]]] = 1
            network[nba_node_dict[line[1]], nba_node_dict[line[0]]] = 1
            line = file.readline()
    dataset["edge_index"] = torch.nonzero(torch.Tensor(network).float(), as_tuple=False).t()
    dataset["Y"] = torch.Tensor(nba_label)
    dataset["X"] = min_max_scale_features(torch.Tensor(np.array(nba_node_feature)))
    dataset["S"] = min_max_scale_features(torch.Tensor(nba_sensitive))
    print(dataset.keys())
    print(dataset["edge_index"].shape)
    print(dataset["Y"].shape)
    print(dataset["X"].shape)
    print(dataset["S"].shape)
    config.num_nodes = len(nba_node_dict)
    config.num_feats = len(feature)
    config.num_sensitive_class = len(nba_sensitive_dict)
    config.num_labels = len(nba_label_dict)
    config.dataset_name = "nba"

    save_graphs(dataset, config.data_path + "/{}/raw/{}.pt".format(dataset_name,dataset_name), "pt")
    result = GraphDataset(root=config.data_path + "/{}".format(dataset_name), dataset_name=dataset_name)
    return result

def load_german_data(config):
    dataset_name = "german"
    dataset_name = "german"
    data = pd.read_csv(config.data_path + "/{}".format(dataset_name) + "/german.csv")

    # Step 1: Transform 'Gender' to binary (1 for 'Male', 0 for 'Female')
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # Step 2: One-hot encode 'PurposeOfLoan' as it has multiple categories
    data = pd.get_dummies(data, columns=['PurposeOfLoan'], drop_first=True)

    german_label = data['GoodCustomer'].values.astype(np.float32).reshape(-1, 1)
    # Extract sensitive matrix from 'Gender' column
    german_sensitive = data['Gender'].values.astype(np.float32).reshape(-1, 1)
    # Drop 'GoodCustomer' and 'Gender' to create feature matrix
    feature_columns = data.drop(columns=['GoodCustomer', 'Gender'])
      # Convert to matrix form
    feature_columns = feature_columns.apply(pd.to_numeric, errors='coerce')
    german_node_feature = feature_columns.values.astype(np.float32)
    network = np.zeros((len(german_node_feature), len(german_node_feature)))
    for i in range(len(network)):
        network[i, i] = 1
    with open(config.data_path + "/{}".format(dataset_name) + "/german_edges.txt", "r") as file:
        line = file.readline()
        while line:
            line = line.rstrip("\n")
            line = list(map(lambda x: int(float(x)), line.split(' ')))
            network[line[0], line[1]] = 1
            network[line[1], line[0]] = 1
            line = file.readline()
    print(german_node_feature)
    dataset = {}
    dataset["edge_index"] = torch.nonzero(torch.tensor(network).float(), as_tuple=False).t()
    dataset["Y"] = torch.tensor(german_label)
    dataset["X"] = min_max_scale_features(torch.tensor(german_node_feature)
)
    dataset["S"] = min_max_scale_features(torch.tensor(german_sensitive))
    config.num_nodes = german_node_feature.shape[0]
    config.num_feats = german_node_feature.shape[1]
    config.num_sensitive_class = len(np.unique(german_sensitive))
    config.num_labels = 2
    config.dataset_name = "german"
    save_graphs(dataset, config.data_path + "/{}/raw/{}.pt".format(dataset_name, dataset_name), "pt")
    result = GraphDataset(root=config.data_path + "/{}".format(dataset_name), dataset_name=dataset_name)
    return result

def load_dataset(config: Config, dataset = "nba"):
    assert dataset in ['generate','nba','german'], \
    "dataset parameter should be one of: ['generate','nba','german']"
    if not os.path.exists(config.data_path + "/{}/raw".format(dataset)):
        os.mkdir(config.data_path + "/{}/raw".format(dataset))

    if os.path.isfile(config.data_path + "/{}/raw/{}.pt".format(dataset, dataset)) and os.path.getsize(
            config.data_path + "/{}/raw/{}.pt".format(dataset, dataset)) < 0:
        print("AAAAAAAAA")
    if dataset == "generate":
        graph_list = []
        for i in range(100):
            graph = {}
            X,A,Y = generate_sample_graph(config.num_nodes, config.num_feats, config.num_labels)
            graph['X'] = X
            graph['edge_index'] = torch.nonzero(A, as_tuple=False).t()
            graph['Y'] = Y
            graph_list.append(graph)
        save_graphs(graph_list, config.data_path + "/{}/raw/{}.pt".format(dataset, dataset), "pt")
        dataset = GraphDataset(root=config.data_path + "/{}".format(dataset), dataset_name=dataset)
    elif dataset == "nba":
        dataset = load_nba_data(config)
    elif dataset == "german":
        dataset = load_german_data(config)
    return dataset
def split_data_train_val(dataset, config):
    train_size = int(config.train_size * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader([dataset[i] for i in train_dataset.indices], shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_dataset.indices], shuffle=False)
    return train_loader, val_loader


def split_graph_train_val(data, val_ratio=0.2):
    # Assuming 'data' is a PyTorch Geometric 'Data' object with 'y' as labels.
    num_nodes = len(data.x)
    # Get the indices of all nodes
    nodes = torch.arange(num_nodes)

    num_train_nodes = int(num_nodes*(1-val_ratio))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: num_train_nodes] = True

    # Add train mask to data object
    data.train_mask = train_mask

    # Create a boolean mask for test mask
    val_mask = ~data.train_mask
    data.val_mask = val_mask
    # Split the nodes into train and validation sets

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)

    return data, train_idx, val_idx

def construct_A_from_edge_index(edge_index, num_nodes):
    # Initialize an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Set matrix entries corresponding to edges in the edge_index
    adj_matrix[edge_index[0], edge_index[1]] = 1

    return adj_matrix
def recover_adj_lower(l, config):
    # NOTE: Assumes 1 per minibatch
    #l: flatten vector of the upper triangular part of the adjacency matrix of a graph
    #for example matrix 3x3: l =[a00, a01, a02, a11, a12, a22]
    adj_matrix = torch.zeros((config.num_nodes, config.num_nodes), device=torch.device(config.device)) #adjacency matrix


    # Step 3: Fill the upper triangular part of the adjacency matrix (excluding the diagonal)
    upper_tri_indices = torch.triu_indices(config.num_nodes, config.num_nodes, offset=0)  # Indices for the upper triangular part
    adj_matrix[upper_tri_indices[0], upper_tri_indices[1]] = l

    # Step 4: Copy the upper triangular part to the lower triangular part to ensure symmetry
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix = torch.where(adj_matrix >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

    return adj_matrix
def z_score_normalize(data):
    mean_vals = data.mean(dim=0, keepdim=True)  # Mean of each feature
    std_vals = data.std(dim=0, keepdim=True)    # Standard deviation of each feature

    # Standardize the data
    normalized_data = (data - mean_vals) / (std_vals + 1e-8)  # Add a small value to avoid division by zero
    return normalized_data
def standardize_features(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    X_standardized = (X - mean) / std
    return X_standardized
def min_max_scale_features(X, min_val=0.0, max_val=1.0):
    """
    Scales the features to a specified range [min_val, max_val].

    Args:
        X (torch.Tensor): Input tensor of shape (n, f).
        min_val (float): Minimum value of the scaled data.
        max_val (float): Maximum value of the scaled data.

    Returns:
        torch.Tensor: Min-max scaled tensor.
    """
    X_min = X.min(dim=0, keepdim=True)[0]
    X_max = X.max(dim=0, keepdim=True)[0]
    # To avoid division by zero
    X_max[X_max == X_min] = X_min[X_max == X_min] + 1.0
    X_scaled = (X - X_min) / (X_max - X_min)
    X_scaled = X_scaled * (max_val - min_val) + min_val
    return X_scaled