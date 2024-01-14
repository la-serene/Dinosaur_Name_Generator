import tensorflow as tf
import tensorflow.layers as layers


vocab = ['\n', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
         'v', 'w', 'x', 'y', 'z']

ids_from_chars = layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

chars_from_ids = layers.StringLookup(
    vocabulary=list(vocab), invert=True, mask_token=None)


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
