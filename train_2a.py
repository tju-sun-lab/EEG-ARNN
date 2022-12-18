import gc
import os
import logging
import torch.optim as optim
import datetime
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import scipy.io as sio
from nnModelST_pytorch import zhnn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename='training_2a.log', encoding='UTF-8', mode='w')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:0')
    # device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

logger.info(f"device: {device}")

def datanorm(x):
    for i in range(np.shape(x)[0]):
        x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
    return x


def normalize_adj(adj):
    d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def preprocess_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj)
    return adj


df = pd.read_excel('init_adj_2a.xlsx')
Abf = df.iloc[:, 1:].values
A = preprocess_adj(Abf)
# A = np.ones((60,60))
A = np.float32(A)
A = torch.from_numpy(A)
label = np.array([0, 1]).squeeze()

start_time = datetime.datetime.now()
# ----------------------CNN------------------------
for p in range(7, 8):
    acc_kappa_list = list()
    Test_index = list()
    Test_index.append(p)
    dataName = 'SA0' + str(p)
    datapath_T = r'./data_2a/{}T.mat'.format(dataName)
    datapath_E = r'./data_2a/{}E.mat'.format(dataName)

    data_T = sio.loadmat(datapath_T)
    data_E = sio.loadmat(datapath_E)
    mydata_T = data_T['data_resampled']
    mydata_E = data_E['data_resampled']
    Y_T = data_T['label'].squeeze()
    Y_E = data_E['label'].squeeze()

    # 0:left 1:right 2:feet 3:tongue
    index_T0 = np.where(Y_T == 0)[0]
    index_T1 = np.where(Y_T == 1)[0]
    index_T = np.concatenate((index_T0, index_T1), axis=0)
    Y_1 = Y_T[index_T]
    data_1 = mydata_T[index_T]


    index_E0 = list(np.where(Y_E == 0))[0]
    index_E1 = list(np.where(Y_E == 1))[0]
    index_E = np.concatenate((index_E0, index_E1), axis=0)
    Y_2 = Y_E[index_E]
    data_2 = mydata_E[index_E]
    # mydata = mydata[30:,:,:,:]

    mydata = np.concatenate((data_1, data_2), axis=0)
    mydata = np.expand_dims(mydata, axis=1)
    Y = np.concatenate((Y_1, Y_2), axis=0)

    logger.info(Y)
    # Y = Y[30:]
    X = datanorm(mydata)

    del mydata
    del mydata_T
    del mydata_E
    gc.collect()

    skf = KFold(n_splits=10, shuffle=True)
    model_acc = list()
    count = 0

    for train_index, test_index in skf.split(X, Y):
        acc_bf = 0
        # Y = np.eye(2)[Y]

        count = count + 1
        X_train, X_test = X[train_index].astype(np.float32), X[test_index].astype(np.float32)
        y_train, y_test = Y[train_index].astype(np.int8), Y[test_index].astype(np.int8)
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)

        logger.info("number of training examples = " + str(X_train.shape[0]))
        logger.info("number of test examples = " + str(X_test.shape[0]))

        data_train = Data.TensorDataset(X_train, y_train)
        trainloader = DataLoader(data_train, batch_size=20, shuffle=True, num_workers=0)

        input_shape = np.shape(X_train)
        net = zhnn((input_shape[2], input_shape[3]), A)
        net.to(device)
        # Define the loss function and method of optimization
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # train network
        acclist = list()
        for epoch in range(500):
            running_loss = 0.0
            c = 0
            correct = 0
            total = 0

            # net.train()
            for i, data in enumerate(trainloader, 0):
                # get the input
                inputs, labels = data
                inputs = inputs.to(device)  # GPU
                labels = labels.to(device)
                # zeros the paramster gradients

                optimizer.zero_grad()  #
                # forward + backward + optimize
                outputs = net(inputs)

                # print(outputs, labels)
                loss = criterion(outputs, labels)  # Calculating loss
                _, pred = torch.max(outputs, 1)  # Calculating training accuracy
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                acc_tr = float(correct) / total
                loss.backward()
                A = net.A

                with torch.no_grad():
                    W_grad = A.grad
                    A = (1 - 0.001) * A - 0.001 * W_grad
                A = nn.Parameter(A, requires_grad=False)
                optimizer.step()  # Updating Parameters
                A = nn.Parameter(A)
                net.A = A

                # print statistics
                running_loss += loss.item()
                c = i
            logger.info('>>>sub [%d], cross [%d], epoch [%d], Train Loss: %.3f  Train Acc: %.3f' %
                  (p, count, epoch + 1, running_loss / c, acc_tr))  # Output average loss

            correct = 0
            total = 0

            # net.eval()
            with torch.no_grad():
                # forward
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                out = net(X_test)
                _, pred = torch.max(out, 1)
                y_score = out[:, -1]
                correct += (pred == y_test).sum().item()
                total += y_test.size(0)
            # Acc
            acc = float(correct) / total
            logger.info('Val Acc = {:.5f}'.format(acc))
            filepath = './model_save/2a_sub{}_cross{}.pth'.format(p, count)
            if acc > acc_bf:
                torch.save(obj=net.state_dict(), f=filepath)
                acc_bf = acc
                logger.info("model save")
            acclist.append(acc)

        accuracy = max(acclist)
        logger.info(f'test accuracy: {accuracy}')
        model_acc.append(accuracy)
        logger.info(f'model_acc: {model_acc}')

    model_acc = np.array(model_acc)
    acc_kappa_list.append(p)
    acc_kappa_list.append(np.min(model_acc))
    acc_kappa_list.append(np.max(model_acc))
    acc_kappa_list.append(np.mean(model_acc))
    acc_kappa_list.append(np.std(model_acc))

    del X, Y
    gc.collect()
    logger.info(f'min: {np.min(model_acc)}')
    logger.info(f'max: {np.max(model_acc)}')
    logger.info(f'mean: {np.mean(model_acc)}')
    logger.info(f'std: {np.std(model_acc)}')

end_time = datetime.datetime.now()
logger.info(f'program time: {end_time - start_time}')
logger.info('Fineshed!')