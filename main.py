import argparse
import random

import torch

from config import Config
from data import load_dataset
from model import GraphVAE, run_model_debug
import numpy as np

from train import train

parser = argparse.ArgumentParser(
                    description="Implementation of: Graph Fairness without Demographics through Fair Inference")
parser.add_argument("--mode", default = "debug", choices=["debug", "train", "test"],
                    help = "Mode debug, train or test GraphVAE model")


if __name__ == '__main__':
    config = Config()
    args = parser.parse_args()
    if args.mode == "debug":
        run_model_debug(config)
        exit()
    elif args.mode == "train":
        load_dataset(config, "generate")
        train(config)
    elif args.mode == "test":
        load_dataset(config)
        train(config)

