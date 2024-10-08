import argparse
import random
import re
import pandas as pd
import torch

from config import config
from data import load_dataset, split_data_train_val, split_graph_train_val
from model import GraphVAE, run_model_debug
from torch.utils.data import random_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import os
from train import train, grid_search_train
import logging

logging.basicConfig(filename='log/all.log',
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    encoding='utf-8',
                    level=logging.DEBUG)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
handler = logging.FileHandler('log/main.log')
handler.setFormatter(log_format)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser(
    description="Implementation of: Graph Fairness without Demographics through Fair Inference")
parser.add_argument("--mode", default="train", choices=["debug", "train", "test", "grid"],
                    help="Mode debug, train, test or run grid search GraphVAE model")
parser.add_argument("--dataset", default="nba", choices=["generate", "nba", "german", "credit", "pokecz", "bail","pokecn"],
                    help="Dataset to train model")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    help="Device config")
parser.add_argument("--num_epoch", default=10001,
                    help="Train epoch config")
parser.add_argument("--train_size", default=0.8,
                    help="Train size for split config")
parser.add_argument("--efl", default=1,
                    help="EFL coefficient for fair loss")
parser.add_argument("--hgr", default=1,
                    help="HGR coefficient for u_Y and S in dependence")
parser.add_argument("--log_epoch", default=10,
                    help="Number of epochs per evaluation")
parser.add_argument("--walk_length", default=50,
                    help="Number of walk length for subgraph")
parser.add_argument("--num_walk", default=20,
                    help="Number of random walk sample for subgraph")

if __name__ == '__main__':
    args = parser.parse_args()
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
        config.lambda_hgr = int(float(args.hgr))
    if args.log_epoch is not None:
        config.log_epoch = int(float(args.log_epoch))
    if args.walk_length is not None:
        config.random_walk_length = args.walk_length
    if args.num_walk is not None:
        config.num_random_walk_sample = args.num_walk
    print("DEVICE: ", config.device)
    print("Train Epoch: ", config.train_epoch)
    print("Train Size: ", config.train_size)
    print("efl_gamma: ", config.efl_gamma)
    print("lambda_hgr: ", config.lambda_hgr)
    print("random_walk_length: ", config.random_walk_length)
    print("num_random_walk_sample: ", config.num_random_walk_sample)
    if args.mode == "debug":
        run_model_debug(config)
        exit()
    elif args.mode == "train":
        dataset = load_dataset(config, args.dataset)
        for data in dataset:
            data.validate()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        train(loader, dataset)
    elif args.mode == "test":
        dataset = load_dataset(config, args.dataset)
        for i in range(len(dataset)):
            print(dataset[i])
        dataset = split_graph_train_val(dataset, config.train_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        train(loader, dataset)
    elif args.mode == "grid":
        dataset = load_dataset(config, args.dataset)
        for data in dataset:
            data.validate()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        grid_summary = ""
        grid_df = pd.DataFrame(columns = ["trial_number","efl", "hgr","val_acc","SDP","F1","EOD","ROC","S_acc"])
        count = 0
        for efl in [-1e4,-1e2,0,1e2,1e4]:
            for hgr in [-1e4,-1e2,0,1e2,1e4]:
                    config.efl_gamma = efl
                    config.lambda_hgr = hgr
                    # config.gcn_hidden_dim = gcn_hidden
                    config.grid = 10000
                    config.show()
                    val_acc, spd, F1, eod, roc, S_acc = grid_search_train(loader, dataset, count)
                    grid_summary += f"\nTRIAL {count}:\nEFL: {efl}\nHGR: {hgr}\nRESULT:\tVal Acc: {val_acc}\tSPD: {spd}\t F1: {F1}\tEOD: {eod}\tROC: {roc}\tS_accuracy: {S_acc}"
                    grid_df.loc[count] = [count, efl, hgr, val_acc, spd, F1, eod, roc, S_acc]
                    count += 1
                    print("--------------------DONE 1 GRID SEARCH LOOP---------------------")
                    print(grid_summary)
                    grid_df.to_csv("grid_result.csv", index=False)
    else:
        print("mode should be one of [debug, train, test, grid]")
