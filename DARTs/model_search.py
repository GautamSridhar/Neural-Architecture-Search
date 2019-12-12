import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class Node(nn.Module):


    def __init__(self, in_feature):
        super(Node, self).__init__()
        self.in_feature = in_feature

        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](self.in_feature)
            self._ops.append(op)


    def forward(self, x, weights):

        weighted_sum = 0 
        for w, op in zip(weights, self._ops):
            output = op(x)
            if output.shape[1] != self.in_feature:
                pd = (0,self.in_feature - output.shape[1])
                output = F.pad(output,pd,'constant')
            weighted_sum += w*output
        print(op)
        print(weighted_sum.shape)

        return weighted_sum


class Network(nn.Module):


    def __init__(self, network_inputsize, network_outputsize, max_depth, max_width, criterion):
        super(Network, self).__init__()
        self._network_inputsize = network_inputsize
        self._network_outputsize = network_outputsize 
        self._max_depth = max_depth
        self._max_width = max_width
        self._criterion = criterion

        self._ops = nn.ModuleList()

        for i in range(self._max_depth):
            if i == 0:
                op = Node(self._network_inputsize)
                self._ops.append(op)
            else:
                op = Node(self._max_width)
                self._ops.append(op)

        self.postprocess = nn.Linear(self._max_width,self._network_outputsize)

        self._initialize_alphas()


    def forward(self, x, weights):
        weights = F.softmax(self.w_alpha, dim=-1)

        h = x
        for i in range(len(self._net)):
            h = self._ops[i](h, weights[i])

        h = self.postprocess(h)

        return h 


    def _initialize_alphas(self):
        num_op   = len(PRIMITIVES)
        num_node = self._max_depth
        self.register_parameter('w_alpha', nn.Parameter(torch.rand((num_node, num_op))))

        self._arch_parameters = [self.w_alpha]


    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 


    def arch_parameters(self):
        return self._arch_parameters


    def genotype(self):

        def _parse(weights):
            gene = []
            for i in range(self._max_depth):
                for k in range(len(W[i])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[i][k] > W[i][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], i))
            return gene

    gene_normal = _parse(F.softmax(self.w_alphas, dim=-1).data.cpu().numpy())

    genotype = Genotype(gene=gene)
    return genotype

# if __name__ == '__main__':
#     n = Node(in_feature=16)
#     x = torch.zeros((1,16))
#     weights = torch.zeros((9,))
#     y = n(x,weights)

