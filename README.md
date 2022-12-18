English | [简体中文](https://github.com/tju-sun-lab/eeg-arnn/blob/main/README_CN.md)

## Description
Source code for the paper: [Sun, Biao, et al. "Graph Convolution Neural Network based End-to-end Channel Selection and Classification for Motor Imagery Brain-computer Interfaces." IEEE Transactions on Industrial Informatics (2022).](https://ieeexplore.ieee.org/abstract/document/9976236/). This paper is **open access**, so you don't need to pay to download it.

In the current repository, we provide data and labels for subject number 17 in the tju dataset and subject number 7 in BCI Competition IV 2a. The data of other subjects in the tju dataset in the paper are not open source yet, please contact the authors if you need them. You can also easily use this code on your own dataset.

We trained and tested EEG-ARNN on subject tju_17 and subject BCI2a_7. The training process and results were saved in training_tju.log and training_2a.log files, respectively.

The code to run EEG-ARNN is train_tju.py and train_2a.py.

## How to run the EEG-ARNN code

Run the train_tju.py file to train the tju dataset.

Run the train_2a.py file to train the BCI2a dataset.

## Description of each file in the repository

model_save: save the network model after each fold of training in the ten-fold cross-validation.

data_2a: holds the data and labels for subject #7.

data_tju: store the data and labels of subject 17.

init_adj_2a.xlsx: the initialized adjacency matrix for BCI2a dataset.

init_adj_tju.xlsx: initialized adjacency matrix for tju dataset.

train_2a.py:the main function to train the network on BCI2a dataset.

train_tju.py: the main function for training the network on tju dataset.

nnModelST_pytorch.py: network model definition.

gcnModelST_pytorch.py: implementation of the graph convolution operation.

training_2a.log: training process and results of BCI2a subject No.7 data on EEG-ARNN model.

training_tju.log:the training process and results of tju subject No.17 data on EEG-ARNN model.