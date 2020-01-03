"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
from utils import Gene
import time


def _construct_dags(operations, func_names):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.
    Args:
        operations: operations sampled from the policy
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.
    Returns:
        A list of DAGs defined by the inputs.
    RNN cell DAGs are represented in the following way:
    1. Each element (node) in a DAG is a list of `Node`s.
    2. The `Node`s in the list dag[i] correspond to the subsequent nodes
       that take the output from node i as their own input.
    3. dag[-1] is the node that takes input from x^{(t)} and h^{(t - 1)}.
       dag[-1] always feeds dag[0].
       dag[-1] acts as if `w_xc`, `w_hc`, `w_xh` and `w_hh` are its
       weights.
    4. dag[N - 1] is the node that produces the hidden state passed to
       the next timestep. dag[N - 1] is also always a leaf node, and therefore
       is always averaged with the other leaf nodes and fed to the output
       decoder.
    """
    dags = []
    for ops in operations:
        dag = []
        for i,func_ids in enumerate(ops):
            # add first node
            dag.append(Gene(i, func_names[func_ids]))
        dags.append(dag)

    return dags


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        self.num_tokens = [] 

        for idx in range(self.args._max_depth):
            self.num_tokens += [len(args.operations)]
        
        self.op_names = args.operations

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return torch.zeros(key, self.args.controller_hid)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if is_embed == False:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, sample_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an operation.
        """

        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size] 

        operations = []
        entropies = []
        log_probs = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for snums in range(sample_size):
            for block_idx in range(self.args._max_depth):
                logits, hidden = self.forward(inputs, hidden, block_idx, is_embed=(block_idx == 0))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                # TODO(brendan): .mean() for entropy?
                entropy = -(log_prob * probs).sum(1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(1, action)

                # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
                # .view()? Same below with `action`.
                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                inputs = action[:, 0] + self.num_tokens[0]

                operations.append(action[:, 0])

        operations = torch.stack(operations).transpose(0, 1)

        dags = _construct_dags(operations, self.op_names)
        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        # this needs to be checked properly
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (zeros,zeros.clone())