import torch
import torch.nn as nn

OPS = {
       # 'none': lambda out_feature: Zero(),
       'LinearReLU_2': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=2),
       'LinearReLU_4': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=4),
       'LinearReLU_8': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=8),
       'LinearReLU_16': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=16),
       # 'LinearReLU_32': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=32),
       # 'LinearReLU_64': lambda in_feature: LinearReLU(in_feature=in_feature, out_feature=64),
       'LinearTanh_2': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=2),
       'LinearTanh_4': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=4),
       'LinearTanh_8': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=8),
       'LinearTanh_16': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=16),
       # 'LinearTanh_32': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=32),
       # 'LinearTanh_64': lambda in_feature: LinearTanh(in_feature=in_feature, out_feature=64),
       # 'LinearSigmoid_2': lambda in_feature: LinearSigmoid(in_feature=in_feature, out_feature=2),
       'Identity': lambda in_feature: Identity()
      }

class LinearReLU(nn.Module):

    def __init__(self, in_feature,out_feature):

        super(LinearReLU,self).__init__()
        self.cell = nn.Sequential(
                                  nn.Linear(in_feature,out_feature),
                                  nn.ReLU()
                                 )

    def forward(self, x):
        return self.cell(x)


class LinearTanh(nn.Module):

    def __init__(self, in_feature, out_feature):

        super(LinearTanh,self).__init__()
        self.cell = nn.Sequential(
                                  nn.Linear(in_feature,out_feature),
                                  nn.Tanh()
                                 )

    def forward(self, x):
        return self.cell(x)


class LinearSigmoid(nn.Module):

    def __init__(self, in_feature, out_feature):

        super(LinearSigmoid,self).__init__()
        self.cell = nn.Sequential(
                                  nn.Linear(in_feature,out_feature),
                                  nn.Sigmoid()
                                 )

    def forward(self, x):
        return self.cell(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)
