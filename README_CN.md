[English](https://github.com/tju-sun-lab/eeg-arnn/blob/main/README.md) | 简体中文

## 代码说明

论文源代码: [Sun, Biao, et al. "Graph Convolution Neural Network based End-to-end Channel Selection and Classification for Motor Imagery Brain-computer Interfaces." IEEE Transactions on Industrial Informatics (2022).](https://ieeexplore.ieee.org/abstract/document/9976236/). 该论文是开放获取(Open Access)的，因此不需要付费就可以下载。

在当前的仓库中，我们提供了tju dataset中的17号被试的数据和标签以及BCI Competition IV 2a中的7号被试的数据和标签。论文中tju dataset的其他被试数据暂未开源，如有需要请联系作者。你也可以很方便的在自己的数据集上使用本代码。

我们在tju_17号被试和BCI2a_7号被试上对EEG-ARNN进行了训练和测试，其训练过程和结果分别保存在了training_tju.log和training_2a.log文件中。

运行EEG-ARNN的代码为train_tju.py和train_2a.py。

## 运行EEG-ARNN代码
运行train_tju.py文件以训练tju dataset数据。

运行train_2a.py文件以训练BCI2a dataset数据。

## 文件夹中各文件说明

model_save:在十折交叉验证中保存每一折训练后的网络模型。

data_2a：存放7号被试的数据和标签。

data_tju:存放17号被试的数据和标签。

init_adj_2a.xlsx:初始化的邻接矩阵。

init_adj_tju.xlsx:初始化的邻接矩阵。

train_2a.py:在BCI2a数据集上训练网络的主函数。

train_tju.py:在tju数据集上训练网络的主函数。

nnModelST_pytorch.py:网络模型定义。

gcnModelST_pytorch.py:图卷积操作的实现。

training_2a.log:BCI2a7号被试数据在EEG-ARNN模型上的训练过程和结果。

training_tju.log:tju 17号被试数据在EEG-ARNN模型上的训练过程和结果。
