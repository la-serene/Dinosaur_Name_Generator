import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from utils import *


@tf.keras.utils.register_keras_serializable()
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
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.embedding = layers.Embedding(input_dim=vocab_size + 1,
                                          output_dim=embedding_size,
                                          mask_zero=True)
        self.lstm = layers.LSTM(units=hidden_units,
                                return_sequences=True,
                                return_state=True)
        self.dense = layers.Dense(units=vocab_size)

    def call(self,
             x,
             training=False,
             return_state=False,
             **kwargs):
        """
            Call the model

        :param x: Input character/ string
        :param training:
        :param return_state:
        """
        x = self.embedding(x)
        states, h, c = self.lstm(x, training=training,
                                 **kwargs)
        if return_state:
            return states, h, c
        else:
            return states

    def get_initial_state(self, batch_size=1):
        zeros_tensor = tf.ones((batch_size, self.hidden_units))
        return self.lstm.get_initial_state(zeros_tensor)

    def predict(self,
                next_char,
                maxlen=40):
        """
            Generate a name from a given character/ string.
        :param next_char: input
        :param maxlen: maximum length of the output
        :return: generated name
        """
        def sampling(states):
            logits = self.dense(states)
            probs = tf.nn.softmax(logits)
            dist = probs.numpy().squeeze()
            idx = np.random.choice(range(len(vocab)), p=dist)

            return idx

        next_char = list(next_char)
        result = next_char.copy()
        h, c = self.get_initial_state()

        for i in range(maxlen):
            if next_char != "\n":
                next_char = tokenizer(next_char)

                while np.ndim(next_char) != 2:
                    next_char = tf.expand_dims(next_char, axis=0)

                states, h, c = self(next_char,
                                    return_state=True,
                                    initial_state=[h, c])

                # Only take the last state from inner LSTM
                next_idx = sampling(states[:, -1])
                next_char = chars_from_ids(next_idx)

                if next_char == "[UNK]":
                    continue

                # Retrieve value from tensor and decode from byte to ASCII
                result.append(next_char.numpy().decode('ascii'))
            else:
                break

        result = "".join(result)

        return result
