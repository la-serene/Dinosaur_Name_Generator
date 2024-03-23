from utils import *
import tensorflow.keras.utils.pad_sequences as pad_sequences


def tokenize_by_character(dataset):
    return list(map(lambda x: tokenizer(list(x)), dataset))


def padding_to_maxlen(dataset, maxlen):
    return pad_sequences(dataset, maxlen, padding="post")


def tokenize_and_padding(context, target):
    context = tokenize_by_character(context)
    target = tokenize_by_character(target)

    maxlen = max(len(i) for i in context)
    context = padding_to_maxlen(context, maxlen)
    target = padding_to_maxlen(target, maxlen)

    return context, target
