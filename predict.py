import argparse
from model import Generator
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--save_path', type=str, default="./weights/model_v2.h5")

    return parser.parse_args()


def main():
    args = get_args()

    vocab_size = len(vocab)
    inputs = args.start
    generator = Generator(vocab_size, args.embedding_size, args.hidden_units)

    # Since there is no build() method, model needs calling before
    # importing weight
    generator.predict("placeholder")

    generator.load_weights(args.save_path)
    result = generator.predict(inputs)
    print(result)

    return result


if __name__ == "__main__":
    main()
