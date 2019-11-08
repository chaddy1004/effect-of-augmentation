import argparse
import random

import numpy as np
import tensorflow as tf

from dataloader import create_dataset
from network import VGG11
from trainer import VGG11Trainer
from utils import process_config


def main(config_path: str):
    config = process_config(config_path)
    train_dataset, test_dataset = create_dataset(config)
    vgg11 = VGG11(config).compile_model()
    trainer = VGG11Trainer(vgg11, train_dataset, test_dataset, config)
    trainer.train()


if __name__ == "__main__":
    seed = 480
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config-vanilla.yaml", help="config path to use")
    args = vars(ap.parse_args())

    main(args["config"])