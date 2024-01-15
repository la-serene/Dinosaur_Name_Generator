import argparse
import tensorflow as tf
from model import Generator
from utils import *
from data_loading import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, default="./data/dino.txt")
    parser.add_argument('--pretrain', type=str, default="./data/dataset/data/Vertebrata")
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--pretrain_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default="./weight/model_v1.keras")

    return parser.parse_args()


def main():
    args = get_args()

    vocab_size = len(vocab) + 1
    generator = Generator(vocab_size, args.embedding_size, args.hidden_units)

    dataset = get_training(args.training, 1024, args.batch_size)
    pretrain = get_pretrain(args.pretrain, 1024, args.batch_size)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Pretraining on the Vertebrata name dataset
    generator.train(pretrain, loss_fn, optimizer, epochs=args.pretrain_epochs)

    # Training on the Dinosaur name dataset
    generator.train(dataset, loss_fn, optimizer, epochs=args.epochs)

    generator.save(args.save_path)

    return 0


if __name__ == "__main__":
    main()
