from collections import defaultdict

import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import to_networkx
import os, sys

from tqdm import tqdm

from config import Config
import torch
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.cluster import SpectralClustering

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
def fair_transfer_probability(neighbor_list, sen_list):
    prob_list = np.zeros(len(sen_list))
    sen_dict = {}
    for neighbor in neighbor_list:
        sensitive_value = int(sen_list[neighbor])
        if sensitive_value in sen_dict.keys():
            sen_dict[sensitive_value] += 1
        else:
            sen_dict[sensitive_value] = 1
    group_prob = 1/len(sen_dict)
    for neighbor in neighbor_list:
        prob = group_prob/sen_dict[int(sen_list[neighbor])]
        prob_list[neighbor] = prob
    return prob_list

def cumulative_random_choice(c):
    r = np.random.random()
    cumulative_sum = 0
    for i, prob in enumerate(c):
        cumulative_sum += prob
        if cumulative_sum >= r:
            return i

def fair_subgraph(node, neighbor_dict, sen_list, graph, num_sample=10, walk_length=50):
    time = 0
    sen_list = sen_list.data.cpu().numpy()
    subgraph_nodes = set()
    subgraph_edges = []
    while time < num_sample:
        current_node = node
        for i in range(walk_length):
            neighbor_list = list(neighbor_dict[current_node])
            if len(neighbor_list) > 1 :
                prob_list = fair_transfer_probability(neighbor_list, sen_list)
                next_node = neighbor_list[cumulative_random_choice(prob_list)]
                subgraph_nodes.add(next_node)
                subgraph_edges.append([current_node, next_node])
                current_node = next_node
            elif (len(neighbor_list) == 1) and (neighbor_list[0] not in subgraph_nodes):
                prob_list = fair_transfer_probability(neighbor_list, sen_list)
                next_node = neighbor_list[cumulative_random_choice(prob_list)]
                subgraph_nodes.add(next_node)
                subgraph_edges.append([current_node, next_node])
                current_node = next_node
            elif len(neighbor_list) == 0:
                time += 1
                break
        time += 1

    network = np.zeros((max(subgraph_nodes)+1, max(subgraph_nodes)+1))
    for i in range(len(network)):
        network[i, i] = 1
    for edge in subgraph_edges:
        network[edge[0], edge[1]] = 1
        network[edge[1], edge[0]] = 1
    edge_index = torch.nonzero(torch.Tensor(network).float(), as_tuple=False).t()
    print(f"DONE RANDOM WALK: {len(subgraph_nodes)} NODES, {edge_index.shape[1]} EDGES")
    return Data(x=graph['X'][list(subgraph_nodes)], edge_index=edge_index, y=graph['Y'][list(subgraph_nodes)], s=graph["S"][list(subgraph_nodes)], subgraph_nodes = torch.tensor(list(subgraph_nodes)))

def fair_subgraph_concurrent(A, P, X, Y, S, device, L = 10, num_walks = 50):

    A = torch.tensor(A, dtype=torch.float32, device=device)
    P = torch.tensor(P, dtype=torch.float32, device=device)

    # Normalize P to ensure each row sums to 1
    P = P / P.sum(dim=1, keepdim=True)

    num_nodes = A.shape[0]

    # Precompute cumulative sums of the transition probabilities
    P_cumsum = torch.cumsum(P, dim=1)  # Shape: (n, n)

    # Initialize walks: shape (n, num_walks, L + 1)
    walks = torch.zeros((num_nodes, num_walks, L + 1), dtype=torch.long, device=device)
    walks[:, :, 0] = torch.arange(num_nodes, device=device).unsqueeze(1)  # Starting nodes

    for t in range(1, L + 1):
        current_nodes = walks[:, :, t - 1]  # Shape: (n, num_walks)
        current_nodes_flat = current_nodes.reshape(-1)  # Shape: (n * num_walks)

        # Get the cumulative distribution for the current nodes
        cdf = P_cumsum[current_nodes_flat, :]  # Shape: (n * num_walks, n)

        # Generate random values for sampling
        random_values = torch.rand(num_nodes * num_walks, device=device).unsqueeze(1)

        # Sample next nodes
        next_nodes_flat = torch.sum(cdf < random_values, dim=1)

        # Reshape back to (n, num_walks)
        next_nodes = next_nodes_flat.view(num_nodes, num_walks)

        # Update the walks with the next nodes
        walks[:, :, t] = next_nodes

    # Extract subgraphs
    subgraphs = []

    # Process each node to extract its subgraph
    for i in range(num_nodes):
        # Extract walks starting from node i
        walks_i = walks[i, :, :]  # Shape: (num_walks, L + 1)

        # Collect all unique nodes visited
        nodes_in_walk = torch.unique(walks_i)

        # Collect edges from walks_i
        u = walks_i[:, :-1].reshape(-1)
        v = walks_i[:, 1:].reshape(-1)
        edges = torch.stack((u, v), dim=1)
        edges = torch.sort(edges, dim=1)[0]
        edges_unique = torch.unique(edges, dim=0)

        # Relabel nodes
        nodes_in_walk_np = nodes_in_walk.cpu().numpy()
        num_nodes_in_subgraph = len(nodes_in_walk_np)
        mapping = {old_id: new_id for new_id, old_id in enumerate(nodes_in_walk_np)}
        X_sub= X[nodes_in_walk_np]
        Y_sub = Y[nodes_in_walk_np]
        S_sub = S[nodes_in_walk_np]

        # Map edges
        edges_np = edges_unique.cpu().numpy()
        edges_mapped = np.array([[mapping[u], mapping[v]] for u, v in edges_np])
        edge_index = edges_mapped.T

        # Store the subgraph for node i
        subgraphs.append({
            'nodes': torch.tensor(np.arange(num_nodes_in_subgraph), dtype=torch.float32, device=device),
            'edge_index': torch.tensor(edge_index, dtype=torch.int64, device=device),
            'mapping': mapping,
            'X': X_sub,
            "Y": Y_sub,
            'S': S_sub
        })
    return subgraphs


def custom_sample_cdf(cdf):
    # Generate random values, ensuring they are less than 1.0
    random_values = torch.rand(cdf.size(0), device=cdf.device)
    # Use torch.searchsorted to find the indices efficiently
    next_nodes = torch.searchsorted(cdf, random_values.unsqueeze(1), right=False).squeeze(1)
    return next_nodes
def fair_subgraph_concurrent_with_batch(A, P, X, Y, S, device, L = 10, num_walks = 50):

    A = torch.tensor(A, dtype=torch.float32, device=device)
    P = torch.tensor(P, dtype=torch.float32, device=device)
    row_sums = P.sum(dim=1, keepdim=True)
    zero_row_mask = row_sums == 0

    # For nodes with no outgoing edges, create self-loops
    P[zero_row_mask.squeeze(), :] = 0
    P[zero_row_mask.squeeze(), zero_row_mask.squeeze()] = 1.0
    # Normalize P to ensure each row sums to 1
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / row_sums

    num_nodes = A.shape[0]

    # Precompute cumulative sums of the transition probabilities
    P_cumsum = torch.cumsum(P, dim=1)  # Shape: (n, n)
    P_cumsum[:, -1] = 1.0

    batch_size = 200
    subgraphs = []

    accumulated_nodes = defaultdict(set)  # Key: node ID, Value: set of node IDs visited
    accumulated_edges = defaultdict(set)
    for batch_start in tqdm(range(0, num_nodes, batch_size), position = 0, desc="Nodes", leave=False, colour="red"):
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_indices = torch.arange(batch_start, batch_end, device=device)
        batch_size_actual = len(batch_indices)

        # Initialize walks for the batch
        walks_batch = torch.zeros((len(batch_indices), num_walks, L + 1), dtype=torch.long, device=device)
        walks_batch[:, :, 0] = batch_indices.unsqueeze(1)
        for t in tqdm(range(1, L + 1), position = 1, desc="Walk length", leave=False, colour="green"):
            current_nodes = walks_batch[:, :, t - 1]  # Shape: (batch_size_actual, num_walks)
            current_nodes_flat = current_nodes.reshape(-1)
            cdf = P_cumsum[current_nodes_flat, :]

            next_nodes_flat = custom_sample_cdf(cdf)
            # Generate random values for sampling
            next_nodes = next_nodes_flat.view(batch_size_actual, num_walks)

            # Update the walks with the next nodes
            walks_batch[:, :, t] = next_nodes

            # Update the walks with the next nodes
            walks_batch[:, :, t] = next_nodes
            for idx, i in tqdm(enumerate(batch_indices), position = 2, leave=False):
                walks_i = walks_batch[idx, :, :]

                # Collect all unique nodes visited
                nodes_in_walk = torch.unique(walks_i)
                accumulated_nodes[i.cpu().item()].update(nodes_in_walk.cpu().numpy())
                # Initialize a set to collect undirected edges
                u = walks_i[:, :-1].reshape(-1)
                v = walks_i[:, 1:].reshape(-1)
                edges = torch.stack((u, v), dim=1)
                edges = torch.sort(edges, dim=1)[0]
                edges_unique = torch.unique(edges, dim=0)

                # Collect edges from all walks
                # for walk in walks_i:
                #     u = walk[:-1]
                #     v = walk[1:]
                #     edges = torch.stack((u, v), dim=1)
                #     # Convert edges to sorted tuples to represent undirected edges
                #     edges = torch.sort(edges, dim=1)[0]
                #     # Convert to list of tuples and add to set
                #     edges_set.update([tuple(edge.cpu().numpy()) for edge in edges])
                nodes_in_walk_np = nodes_in_walk.cpu().numpy()

                # Map the edges to the new node IDs
                edges_np = edges_unique.cpu().numpy()
                accumulated_edges[i.cpu().item()].update([tuple(edge) for edge in edges_np])
            for i in range(num_nodes):
                # Get accumulated nodes and edges for node i
                nodes_in_walk = np.array(list(accumulated_nodes[i]))
                edges_set = accumulated_edges[i]
                num_nodes_in_subgraph = len(nodes_in_walk)
                mapping = {old_id: new_id for new_id, old_id in enumerate(nodes_in_walk)}

                # Map edges to new node IDs
                edges_np = np.array(list(edges_set))
                edges_mapped = np.array([[mapping[u], mapping[v]] for u, v in edges_np])
                edge_index = edges_mapped.T
                # Transpose edges to get edge_index format if needed
                X_sub= X[nodes_in_walk]
                Y_sub = Y[nodes_in_walk]
                S_sub = S[nodes_in_walk]

                subgraphs.append({
                    'nodes': torch.tensor(np.arange(num_nodes_in_subgraph), dtype=torch.float32, device=device),
                    'edge_index': torch.tensor(edge_index, dtype=torch.int64, device=device),
                    'mapping': mapping,
                    'X': X_sub,
                    "Y": Y_sub,
                    'S': S_sub
                })
    return subgraphs
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
                print("Loading Graph...")
                graph = read_graphs(raw_path, "pt")
                device = "cuda:1"
                neighbor_dict = defaultdict(set)
                for src, dst in zip(graph['edge_index'][0], graph['edge_index'][1]):
                    src = int(src.data.cpu())
                    dst = int(dst.data.cpu())
                    neighbor_dict[src].add(dst)
                    neighbor_dict[dst].add(src)
                network = construct_A_from_edge_index(graph['edge_index'], len(graph['X']))
                print("Creating subgraphs...")
                P = [fair_transfer_probability(list(neighbor_dict[node]), graph["S"].data.cpu().numpy()) for node in
                     neighbor_dict.keys()]
                subgraphs = []
                new_subgraphs = fair_subgraph_concurrent_with_batch(network, P, graph["X"], graph["Y"], graph["S"], device = device)
                print("Processing subgraphs...")
                for subgraph in new_subgraphs:
                    data_sub = split_graph_train_val(Data(x=subgraph['X'], edge_index=subgraph['edge_index'], y=subgraph['Y'], s=subgraph['S']))
                    subgraphs.append(data_sub)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

        if self.pre_filter is not None:
            subgraphs = [data_sub for data_sub in subgraphs if self.pre_filter(data_sub)]

        data_sub, slices = self.collate(subgraphs)
        torch.save((data_sub, slices), self.processed_paths[0])

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

def load_credit_data(config):
    dataset_name = "credit"
    data = pd.read_csv(config.data_path + "/{}".format(dataset_name) + "/credit.csv")
    # Extract sensitive matrix from 'Gender' column
    credit_sensitive = data['Age'].values.astype(np.float32).reshape(-1, 1)
    credit_label = data['NoDefaultNextMonth'].values.astype(np.float32).reshape(-1, 1)
    # Drop 'GoodCustomer' and 'Gender' to create feature matrix
    feature_columns = data.drop(columns=['NoDefaultNextMonth', 'Age'])
      # Convert to matrix form
    feature_columns = feature_columns.apply(pd.to_numeric, errors='coerce')
    credit_node_feature = feature_columns.values.astype(np.float32)
    network = np.zeros((len(credit_node_feature), len(credit_node_feature)))
    for i in range(len(network)):
        network[i, i] = 1
    with open(config.data_path + "/{}".format(dataset_name) + "/credit_edges.txt", "r") as file:
        line = file.readline()
        while line:
            line = line.rstrip("\n")
            line = list(map(lambda x: int(float(x)), line.split(' ')))
            network[line[0], line[1]] = 1
            network[line[1], line[0]] = 1
            line = file.readline()
    dataset = {}
    dataset["edge_index"] = torch.nonzero(torch.tensor(network).float(), as_tuple=False).t()
    dataset["Y"] = torch.tensor(credit_label)
    dataset["X"] = min_max_scale_features(torch.tensor(credit_node_feature))
    dataset["S"] = min_max_scale_features(torch.tensor(credit_sensitive))
    config.num_nodes = credit_node_feature.shape[0]
    config.num_feats = credit_node_feature.shape[1]
    config.num_sensitive_class = len(np.unique(credit_sensitive))
    config.num_labels = 2
    config.dataset_name = "credit"
    save_graphs(dataset, config.data_path + "/{}/raw/{}.pt".format(dataset_name, dataset_name), "pt")
    result = GraphDataset(root=config.data_path + "/{}".format(dataset_name), dataset_name=dataset_name)
    print(f"DONE!\nDATASET: {config.dataset_name}\nNUM NODES: {config.num_nodes}\nNUM FEATS: {config.num_feats}")
    return result
def load_pokecz_data(config):
    dataset_name = "pokecz"
    pokecz = pd.read_csv(config.data_path + "/{}".format(dataset_name) + "/region_job_2.csv")
    pokecz["I_am_working_in_field"] = pokecz["I_am_working_in_field"].apply(lambda x: int(float(x == -1)))
    pokecz["region"].value_counts()
    pokecz_label = pokecz['I_am_working_in_field'].values.astype(np.float32).reshape(-1, 1)
    # Extract sensitive matrix from 'Gender' column
    pokecz_sensitive = pokecz['region'].values.astype(np.float32).reshape(-1, 1)
    feature_columns = pokecz.drop(columns=['I_am_working_in_field', 'region', 'user_id'])
    pokecz_node_feature = feature_columns.values.astype(np.float32)
    network = np.zeros((len(pokecz_node_feature), len(pokecz_node_feature)))
    node_dict = {user: id for id, user in enumerate(pokecz['user_id'])}

    with open(config.data_path + "/{}".format(dataset_name) + "/region_job_2_relationship.txt", "r") as file:
        line = file.readline()
        while line:
            line = line.rstrip("\n")
            line = list(map(lambda x: int(float(x)), line.split('\t')))
            network[node_dict[line[0]], node_dict[line[1]]] = 1
            network[node_dict[line[1]], node_dict[line[0]]] = 1
            line = file.readline()
    dataset = {}
    dataset["edge_index"] = torch.nonzero(torch.Tensor(network).float(), as_tuple=False).t()
    dataset["Y"] = torch.tensor(pokecz_label)
    dataset["X"] = min_max_scale_features(torch.tensor(pokecz_node_feature))
    dataset["S"] = min_max_scale_features(torch.tensor(pokecz_sensitive))
    config.num_nodes = pokecz_node_feature.shape[0]
    config.num_feats = pokecz_node_feature.shape[1]
    config.num_sensitive_class = len(np.unique(pokecz_sensitive))
    config.num_labels = 2
    config.dataset_name = "pokecz"
    save_graphs(dataset, config.data_path + "/{}/raw/{}.pt".format(dataset_name, dataset_name), "pt")
    result = GraphDataset(root=config.data_path + "/{}".format(dataset_name), dataset_name=dataset_name)
    print(f"DONE!\nDATASET: {config.dataset_name}\nNUM NODES: {config.num_nodes}\nNUM FEATS: {config.num_feats}")
    return result

def load_bail_data(config):
    dataset_name = "bail"
    path = config.data_path + "/{}/".format(dataset_name)
    bail = pd.read_csv(path + "bail.csv")

    bail_label = bail['RECID'].values.astype(np.float32).reshape(-1, 1)
    # # Extract sensitive matrix from 'Gender' column
    bail_sensitive = bail['WHITE'].values.astype(np.float32).reshape(-1, 1)
    feature_columns = bail.drop(columns=['RECID', 'WHITE'])
    bail_node_feature = feature_columns.values.astype(np.float32)
    network = np.zeros((len(bail_node_feature), len(bail_node_feature)))

    with open(path + "bail_edges.txt", "r") as file:
        line = file.readline()
        while line:
            line = line.rstrip("\n")
            line = list(map(lambda x: int(float(x)), line.split(' ')))
            network[line[0], line[1]] = 1
            network[line[1], line[0]] = 1
            line = file.readline()
    dataset = {}
    dataset["edge_index"] = torch.nonzero(torch.Tensor(network).float(), as_tuple=False).t()
    dataset["Y"] = torch.tensor(bail_label)
    dataset["X"] = min_max_scale_features(torch.tensor(bail_node_feature))
    dataset["S"] = min_max_scale_features(torch.tensor(bail_sensitive))
    config.num_nodes = bail_node_feature.shape[0]
    config.num_feats = bail_node_feature.shape[1]
    config.num_sensitive_class = len(np.unique(bail_sensitive))
    config.num_labels = 2
    config.dataset_name = "bail"
    save_graphs(dataset, config.data_path + "/{}/raw/{}.pt".format(dataset_name, dataset_name), "pt")
    result = GraphDataset(root=config.data_path + "/{}".format(dataset_name), dataset_name=dataset_name)
    print(f"DONE!\nDATASET: {config.dataset_name}\nNUM NODES: {config.num_nodes}\nNUM FEATS: {config.num_feats}")
    return result
def load_dataset(config: Config, dataset = "nba"):
    assert dataset in ['generate','nba','german', 'credit',"pokecz", 'bail'], \
    "dataset parameter should be one of: ['generate','nba','german', 'credit','pokecz','bail']"
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
    elif dataset == "credit":
        dataset = load_credit_data(config)
    elif dataset == "pokecz":
        dataset = load_pokecz_data(config)
    elif dataset == "bail":
        dataset = load_bail_data(config)
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
    num_train_nodes = int(num_nodes*(1-val_ratio))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: num_train_nodes] = True
    # Create a boolean mask for test mask
    val_mask = ~train_mask
    # Split the nodes into train and validation sets
    return Data(x = data.x, y = data.y, edge_index=data.edge_index, s = data.s, train_mask = train_mask, val_mask = val_mask)

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