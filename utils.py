import tensorflow as tf
import tensorflow.keras.layers as layers


vocab = ['\n', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
         'v', 'w', 'x', 'y', 'z', ' ']

tokenizer = layers.StringLookup(vocabulary=list(vocab))
chars_from_ids = layers.StringLookup(
    vocabulary=list(vocab), invert=True)
