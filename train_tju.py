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
from nnModelST_pytorch import zhnn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename='training_tju.log', encoding='UTF-8', mode='w')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:0')
    #device = torch.device('cuda:1')
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

df = pd.read_excel('init_adj_tju.xlsx')
Abf = df.iloc[:, 1:].values
A = preprocess_adj(Abf)
# A = np.ones((60,60))
A = np.float32(A)
A = torch.from_numpy(A)
label = np.array([0, 1]).squeeze()


start_time = datetime.datetime.now()
# ----------------------CNN------------------------

for p in range(17, 18):
    acc_kappa_list = list()
    Test_index = list()
    Test_index.append(p)
    dataName = 'data_' + str(p)
    labelName = 'label_' + str(p)
    datapath = r'./data/{}.npy'.format(dataName)
    labelpath = r'./data/{}.npy'.format(labelName)

    mydata = np.load(datapath)
    # mydata = mydata[30:,:,:,:]
    Y = np.load(labelpath) - 1
    logger.info(Y)
    # Y = Y[30:]
    X = datanorm(mydata)

    del mydata
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

                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)

                # print(outputs, labels)
                loss = criterion(outputs, labels)  # Calculating loss
                _, pred = torch.max(outputs, 1) # Calculating training accuracy
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                acc_tr = float(correct) / total
                loss.backward()

                A = net.A
                with torch.no_grad():
                    W_grad = A.grad
                    A = (1-0.001)*A - 0.001*W_grad
                A = nn.Parameter(A, requires_grad=False)
                optimizer.step()    # Updating Parameters
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
                correct += (pred == y_test).sum().item()
                total += y_test.size(0)
            # Acc
            acc = float(correct) / total
            logger.info('Val Acc = {:.5f}'.format(acc))
            if acc > acc_bf:
                acc_bf = acc
                save_dict = net.state_dict()
            acclist.append(acc)
        filepath = './model_save/tju_sub{}_cross{}.pth'.format(p, count)
        torch.save(obj=save_dict, f=filepath)
        logger.info("model save")


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
logger.info('Fineshed！')