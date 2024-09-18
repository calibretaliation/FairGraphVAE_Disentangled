from scipy.io import loadmat, savemat
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, Data
import os, sys
from config import Config
import torch
import numpy as np
import torch
from torch_geometric.loader import DataLoader
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
class GeneratedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GeneratedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 返回原始数据文件名列表
        return ['generate.pt']

    @property
    def processed_file_names(self):
        # 返回处理后的数据文件名列表
        return ['processed_generated_graph.pt']

    def download(self):
        # 下载数据，如果需要的话
        pass

    def process(self):
        # Here you load your data and convert it into a Data object
        data_list = []
        for raw_path in self.raw_paths:
            # Load raw data, e.g. from a .pt file
            all_data = read_graphs(raw_path, "pt")
            for graph in all_data:
                data = Data(x=graph['X'], edge_index=graph['edge_index'], y=graph['Y'])
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 将数据存储到磁盘上
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_dataset(config: Config, dataset = "generate"):
    assert dataset in ['generate'], \
    "dataset parameter should be one of: ['generate']"
    if dataset == "generate":
        if os.path.isfile(config.data_path + "/raw/{}.pt".format(dataset)) and os.path.getsize(config.data_path + "/raw/{}.pt".format(dataset)) < 0:
            pass
        else:
            graph_list = []
            for i in range(100):
                graph = {}
                X,A,Y = generate_sample_graph(config.num_nodes, config.num_feats, config.num_labels)
                graph['X'] = X
                graph['edge_index'] = torch.nonzero(A, as_tuple=False).t()
                graph['Y'] = Y
                graph_list.append(graph)
            save_graphs(graph_list, config.data_path + "/raw/{}.pt".format(dataset), "pt")
        dataset = GeneratedDataset(root=config.data_path)
        return dataset
def split_data_train_val(dataset, config):
    train_size = int(config.train_size * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader([dataset[i] for i in train_dataset.indices], shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_dataset.indices], shuffle=False)
    return train_loader, val_loader
def construct_A_from_edge_index(edge_index, num_nodes):
    # Initialize an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Set matrix entries corresponding to edges in the edge_index
    adj_matrix[edge_index[0], edge_index[1]] = 1

    return adj_matrix