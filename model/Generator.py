import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tqdm import tqdm

from utils import *


@tf.keras.saving.register_keras_serializable()
class Generator(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_units):
        """
            Define the Text Generator instance.

        :param vocab_size: number of unique characters in vocabulary
        :param embedding_size: dimensionality of embedding layer
        :param hidden_units: dimensionality of the output
        """
        super(Generator, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embedding_size)
        self.lstm = layers.LSTM(units=hidden_units,
                                return_sequences=True,
                                return_state=True)
        self.dense = layers.Dense(units=vocab_size,
                                  activation="softmax")

    def __call__(self,
                 inputs,
                 **kwargs):
        """
            Generate a new character.

        :param inputs: inputs
        :return: model prediction
        """
        embed_inputs = self.embedding(inputs)
        mask = tf.not_equal(inputs, 0)
        states, h, c = self.lstm(embed_inputs,
                                 mask=mask)
        outputs = self.dense(states)

        return outputs

    def train(self,
              dataset,
              loss_fn,
              optimizer,
              epochs=5,
              val_set=None):
        """
            Train the model.

        :param dataset: training set
        :param loss_fn: loss function
        :param optimizer: optimizer
        :param epochs: number of epochs
        :param val_set: validation set
        """
        for epoch in range(epochs):
            loss_sum = 0
            val_loss_sum = 0

            for step, (context, target) in enumerate(tqdm(dataset)):
                tokenized_context = ids_from_chars(context).to_tensor()
                tokenized_target = ids_from_chars(target).to_tensor()

                with tf.GradientTape() as tape:
                    outputs = self(tokenized_context)
                    loss = loss_fn(tokenized_target, outputs)
                    loss_sum += loss

                gradients = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.trainable_weights))

            if val_set is not None:
                for step, (context, target) in enumerate(tqdm(val_set)):
                    tokenized_context = ids_from_chars(context).to_tensor()
                    tokenized_target = ids_from_chars(target).to_tensor()

                    outputs = self(tokenized_context)
                    loss = loss_fn(tokenized_target, outputs)
                    val_loss_sum += loss

                print(f"Epoch: {epoch + 1}, loss = {loss_sum}, val_loss = {val_loss_sum}")
            else:
                print(f"Epoch: {epoch + 1}, loss = {loss_sum}")

    def predict(self,
                next_char):
        """
            Generate a name
        """
        result = [next_char]

        tokenized_char = ids_from_chars(next_char)
        while tokenized_char.ndim != 2:
            tokenized_char = tf.expand_dims(tokenized_char, axis=-1)

        embed_char = self.embedding(tokenized_char)
        states, h, c = self.lstm(embed_char)

        outputs = self.dense(states)
        next_idx = tf.argmax(outputs, axis=-1)
        next_char = chars_from_ids(next_idx)

        for ith in range(30):
            if next_char != "\n":
                tokenized_char = ids_from_chars(next_char)

                while tokenized_char.ndim != 2:
                    tokenized_char = tf.expand_dims(tokenized_char, axis=-1)

                embed_char = self.embedding(tokenized_char)
                states, h, c = self.lstm(embed_char, initial_state=[h, c])

                outputs = self.dense(states)

                dist = outputs.numpy().squeeze()
                next_idx = np.random.choice(range(len(vocab)), p=dist)

                # next_idx = tf.argmax(outputs, axis=-1)
                next_char = chars_from_ids(next_idx)

                if next_char == "[UNK]":
                    continue

                result.append(next_char)
            else:
                break

        return result
