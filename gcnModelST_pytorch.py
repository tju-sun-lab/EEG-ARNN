import torch
import torch.nn as nn


class GCN_layer(nn.Module):

    def __init__(self, signal_shape, bias=False):
        super(GCN_layer, self).__init__()

        # input_shape=(node,timestep)
        self.W = nn.Parameter(torch.ones(signal_shape[0], signal_shape[0]), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(signal_shape[1]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([1, 1, 1, signal_shape[1]]), requires_grad=True)
        self.bias = bias

        # self.params = nn.ParameterDict({
        #         'W': nn.Parameter(torch.rand(signal_shape[0], signal_shape[0]), requires_grad=True),
        #         'theta': nn.Parameter(torch.rand(signal_shape[1]), requires_grad=True)
        # })

    def forward(self, Adj_matrix, input_features):

        # G = torch.from_numpy(Adj_matrix).type(torch.FloatTensor)

        hadamard = Adj_matrix

        aggregate = torch.einsum("ce,abed->abcd", hadamard, input_features)
        output = torch.einsum("abcd,d->abcd", aggregate, self.theta)

        if self.bias == True:
            output = output + self.b

        return output

# net=GCN_layer((60,512))
# print(net)