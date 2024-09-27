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

parser = argparse.ArgumentParser(
                    description="Implementation of: Graph Fairness without Demographics through Fair Inference")
parser.add_argument("--mode", default = "train", choices=["debug", "train", "test"],
                    help = "Mode debug, train or test GraphVAE model")
parser.add_argument("--dataset", default = "nba", choices=["generate", "nba", "german"],
                    help = "Dataset to train model")
parser.add_argument("--device", default = "cpu", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    help = "Device config")
parser.add_argument("--num_epoch", default = 10001,
                    help = "Train epoch config")
parser.add_argument("--train_size", default = 0.8,
                    help = "Train epoch config")

if __name__ == '__main__':
    args = parser.parse_args()
    config = Config()

    if args.device is not None:
        match = re.search(r':(\d)', args.device)
        config.device = args.device
        torch.cuda.set_device(int(match.group(1)))

    if args.num_epoch is not None:
        config.train_epoch = int(args.num_epoch)
    if args.train_size is not None:
        config.train_size = float(args.train_size)
    if args.mode == "debug":
        run_model_debug(config)
        exit()
    elif args.mode == "train":
        dataset = load_dataset(config, args.dataset)
        for i in range(len(dataset)):
            print(dataset[i])
        graph, train_idx, val_idx = split_graph_train_val(dataset, 1 - config.train_size)
        train(config, train_idx, val_idx, graph)
    elif args.mode == "test":
        dataset = load_dataset(config,  args.dataset)
        for i in range(len(dataset)):
            print(dataset[i])
        graph, train_idx, val_idx = split_graph_train_val(dataset, config.train_size)
        train(config, train_idx, val_idx, graph)
    else:
        print("mode should be one of [debug, train, test]")

