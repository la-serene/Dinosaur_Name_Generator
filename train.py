import argparse

from model import Generator
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, default="./data/dino.txt")
    parser.add_argument('--pretrain', type=str, default="./data/dataset/data/Vertebrata")
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default="./weight/checkpoint.keras")
    parser.add_argument('--gpu', type=bool, default=False)

    return parser.parse_args()


def main():
    args = get_args()

    vocab_size = len(vocab) + 1

    generator = Generator(vocab_size, args.embedding_size, args.hidden_units)

    # generator.train()

    return 0


if __name__ == "__main__":
    main()
