import os
import json
import tensorflow as tf


def get_pretrain(data_path, BUFFER_SIZE=1024, BATCH_SIZE=64):
    files = os.listdir(data_path)

    pretrain_context = []
    pretrain_target = []

    for file_name in files:
        file_name = os.path.join(data_path, file_name)
        with open(file_name, "r") as f:
            data = json.load(f)

        for obj in data:
            pretrain_context.append(obj["scientific_name"])
            pretrain_target.append(obj["scientific_name"][1:] + "\n")

    pretrain = transform_to_dataset(pretrain_context, pretrain_target, BUFFER_SIZE, BATCH_SIZE)

    return pretrain


def get_training(file_path, BUFFER_SIZE=1024, BATCH_SIZE=64):
    with open(file_path, "r") as f:
        raw = f.read().lower()

    context = []
    target = []

    for name in raw.split("\n"):
        context.append(name)
        target.append(name[1:] + "\n")

    dataset = transform_to_dataset(context, target, BUFFER_SIZE, BATCH_SIZE)

    return dataset


def transform_to_dataset(context, target, BUFFER_SIZE, BATCH_SIZE):
    context = tf.strings.unicode_split(context, "UTF-8")
    target = tf.strings.unicode_split(target, "UTF-8")

    return (
        tf.data.Dataset
        .from_tensor_slices((context, target))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
