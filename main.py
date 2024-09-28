import argparse
import random
import re

import torch

from config import Config
from data import load_dataset, split_data_train_val, split_graph_train_val
from model import GraphVAE, run_model_debug
from torch.utils.data import random_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
from train import train
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser(
                    description="Implementation of: Graph Fairness without Demographics through Fair Inference")
parser.add_argument("--mode", default = "train", choices=["debug", "train", "test"],
                    help = "Mode debug, train or test GraphVAE model")
parser.add_argument("--dataset", default = "nba", choices=["generate", "nba", "german", "credit"],
                    help = "Dataset to train model")
parser.add_argument("--device", default = "cpu", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    help = "Device config")
parser.add_argument("--num_epoch", default = 10001,
                    help = "Train epoch config")
parser.add_argument("--train_size", default = 0.8,
                    help = "Train size for split config")
parser.add_argument("--efl", default = 10000,
                    help = "EFL coefficient for fair loss")
parser.add_argument("--hgr", default = 10000,
                    help = "HGR coefficient for u_Y and S in dependence")

if __name__ == '__main__':
    args = parser.parse_args()
    config = Config()

    if args.device is not None:
        if args.device != "cpu":
            match = re.search(r':(\d)', args.device)
            config.device = args.device
            torch.cuda.set_device(int(match.group(1)))
        else:
            config.device = "cpu"
    if args.num_epoch is not None:
        config.train_epoch = int(args.num_epoch)
    if args.train_size is not None:
        config.train_size = float(args.train_size)
    if args.efl is not None:
        config.efl_gamma = int(float(args.efl))
    if args.hgr is not None:
        config.efl_gamma = int(float(args.hgr))


    if args.mode == "debug":
        run_model_debug(config)
        exit()
    elif args.mode == "train":
        dataset = load_dataset(config, args.dataset)
        for data in dataset:
            data.validate()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for graph in loader:
            print(graph)
            print(graph.edge_index.max())
        train(config, loader)
    elif args.mode == "test":
        dataset = load_dataset(config,  args.dataset)
        for i in range(len(dataset)):
            print(dataset[i])
        dataset = split_graph_train_val(dataset, config.train_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        train(config, loader)
    else:
        print("mode should be one of [debug, train, test]")

