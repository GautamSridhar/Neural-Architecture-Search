"""Module containing the shared RNN model."""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils
from operations import *
from genotypes import PRIMITIVES

import time

class Node(nn.Module):

    def __init__(self, in_feature, max_width):
        super(Node, self).__init__()
        self._in_feature = in_feature
        self._max_width = max_width

        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](self._in_feature)
            self._ops.append(op)

    def forward(self, x, dag_node):
        output = self._ops[PRIMITIVES.index(dag_node.name_op)](x)
        if output.shape[1] != self._max_width:
            pd = (0,self._max_width - output.shape[1])
            output = F.pad(output,pd,'constant')
        return output

class Network(nn.Module):
    """Shared model."""
    def __init__(self, args):
        super(Network, self).__init__()

        self.args = args
        self.ops = nn.ModuleList()

        self._ops = nn.ModuleList()

        for i in range(self.args._max_depth):
            if i == 0:
                op = Node(self.args._network_inputsize, self.args._max_width)
                self._ops.append(op)
            else:
                op = Node(self.args._max_width, self.args._max_width)
                self._ops.append(op)

        self.postprocess = nn.Linear(self.args._max_width, self.args._network_outputsize)

    def forward(self, inputs, dag):

        interm = inputs
        for i,op in enumerate(self._ops):
            interm = op(interm, dag[i])

        outputs = self.postprocess(interm)
        return outputs
