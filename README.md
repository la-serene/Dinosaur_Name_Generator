# Dinosaur Name Generator

[![Goolge Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qjPV4NrwVqKPJxhlYQn4RGiaA40Z2AS7?usp=sharing)

## Introduction

A dinosaurs name generator project implemented in Tensorflow.

## Dataset

Training dataset is downloaded from
this [Kaggle Notebook](https://www.kaggle.com/code/mruanova/dinosaurs-random-name-generator).

Due to the shortage of data in training dataset, model will be pretrained on another dataset consisting Latin name of
animals belong to 2 classes of Vertebrate: Aves and Mammalia from
this [repo](https://github.com/species-names/dataset.git).

## Getting started

### Training

Remind: Since a previously trained model weight has been saved in weights folder, a training step is optional. 

By default, simply run the following command to train the model.

```
python train.py
```

Here is a description on the command-line arguments:

```
--training: Path to the training data file. Default: ./data/dino.txt
--pretrain: Path to the pretraining data directory. Default: ./data/pretrain
--embedding_size: Size of the embedding layer. Default: 256
--hidden_units: Number of hidden units in the model. Default: 128
--epochs: Number of epochs for training. Default: 20
--pretrain_epochs: Number of epochs for pretraining. Default: 20
--batch_size: Batch size for training. Default: 64
--learning_rate: Learning rate for the optimizer. Default: 0.001
--save_path: Path to save the trained model. Default: ./weight/model_v2.h5
```

### Prediction

The command below requires an argument `start` as the starting of the name. `start` can be a character or a string.

```
python predict.py --start ___
```

Here is a description on the command-line arguments:

```
--start: starting of the name. Required.
--embedding_size: Size of the embedding layer. Default: 256
--hidden_units: Number of hidden units in the model. Default: 128
--save_path: Path to save the trained model. Default: ./weight/model_v2.h5
```
