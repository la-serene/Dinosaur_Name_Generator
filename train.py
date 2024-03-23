import argparse

import tensorflow as tf

from tqdm import tqdm
from data_loading import *
from model import Generator
from utils import *


@tf.function
def train_step(x, y,
               model,
               loss_fn,
               optimizer):
    with tf.GradientTape() as tape:
        states = model(x, training=True)
        logits = model.dense(states)
        loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value


def train(model,
          dataset,
          loss_fn,
          optimizer,
          epochs=40):
    """
    """

    for epoch in range(epochs):
        loss = 0
        for step, (x, y) in enumerate(tqdm(dataset)):
            loss_value = train_step(x, y,
                                    model,
                                    loss_fn,
                                    optimizer)
            loss += loss_value

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}, loss = {loss}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=str, default="./data/dino.txt")
    parser.add_argument('--pretrain', type=str, default="./data/pretrain")
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--pretrain_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default="./weights/model_v2.h5")

    return parser.parse_args()


def main():
    args = get_args()

    vocab_size = len(vocab) + 1
    generator = Generator(vocab_size, args.embedding_size, args.hidden_units)

    dataset = get_training(args.training, 1024, args.batch_size)
    pretrain = get_pretrain(args.pretrain, 1024, args.batch_size)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Pretraining on the Vertebrata name pretrain
    generator.train(pretrain, loss_fn, optimizer, epochs=args.pretrain_epochs)

    # Training on the Dinosaur name pretrain
    generator.train(dataset, loss_fn, optimizer, epochs=args.epochs)

    generator.save_weights(args.save_path)

    return 0


if __name__ == "__main__":
    main()
