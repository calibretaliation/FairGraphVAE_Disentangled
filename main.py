import argparse
import random

import torch

from config import Config
from data import load_dataset, split_data_train_val
from model import GraphVAE, run_model_debug
from torch.utils.data import random_split
import numpy as np

from train import train

parser = argparse.ArgumentParser(
                    description="Implementation of: Graph Fairness without Demographics through Fair Inference")
parser.add_argument("--mode", default = "train", choices=["debug", "train", "test"],
                    help = "Mode debug, train or test GraphVAE model")


if __name__ == '__main__':
    config = Config()
    args = parser.parse_args()
    if args.mode == "debug":
        run_model_debug(config)
        exit()
    elif args.mode == "train":
        dataset = load_dataset(config, "generate")
        for i in range(len(dataset)):
            print(dataset[i])
        train_loader, val_loader = split_data_train_val(dataset, config)
        train(config, train_loader, val_loader)
    elif args.mode == "test":
        dataset = load_dataset(config, "generate")
        for i in range(len(dataset)):
            print(dataset[i])
        train_loader, val_loader = split_data_train_val(dataset, config)
        train(config, train_loader, val_loader)
    else:
        print("mode should be one of [debug, train, test]")

