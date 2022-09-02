import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import logging
from torch_geometric.nn import global_add_pool


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdy = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdy, stdy)
        if self.bias is not None:
            self.bias.data.uniform_(-stdy, stdy)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # def __repr__(self):


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc1 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN_server(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_server, self).__init__()

        # self.gc1 = GraphConv(nfeat, nhid)
        self.gc1 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN_client(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_client, self).__init__()

        # self.gc1 = GraphConv(nfeat, nhid)
        self.gc1 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        return x


def directed_mp(message, edge_index):
    m = []
    for k, (i, j) in enumerate(zip(*edge_index)):
        edge_to_i = (edge_index[1] == i)
        edge_from_j = (edge_index[0] == j)
        nei_message = message[edge_to_i]
        rev_message = message[edge_to_i & edge_from_j]
        m.append(torch.sum(nei_message, dim=0) - rev_message)
    return torch.vstack(m)


def aggregate_at_nodes(x, message, edge_index):
    out_message = []
    for i in range(x.shape[0]):
        edge_to_i = (edge_index[1] == i)
        out_message.append(torch.sum(message[edge_to_i], dim=0))
    return torch.vstack(out_message)


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.initialize_weights(self)
        # self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=False)
        # self.depth = depth

    def initialize_weights(self, model: nn.Module) -> None:
        """
        Initializes the weights of a model in place.
        :param model: An PyTorch model.
        """
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)


    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # d = dict()
        # d["x"] = x
        # d["edge_index"] = edge_index
        # d["edge_attr"] = edge_attr
        # d["batch"] = batch
        # logging.info("{} {} {} {}".format(x.shape, edge_index.shape, edge_attr.shape, batch))
        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        input = self.W1(init_msg)
        h0 = self.act_func(input)
        data.edge_attr = h0
        return data
        # directed message passing over edges
        # h = h0
        # for _ in range(self.depth - 1):
        #     m = directed_mp(h, edge_index)
        #     h = self.act_func(h0 + self.W2(m))
        #
        # # aggregate in-edge messages at nodes
        # v_msg = aggregate_at_nodes(x, h, edge_index)
        #
        # z = torch.cat([x, v_msg], dim=1)
        # node_attr = self.act_func(self.W3(z))
        #
        # # readout: pyg global sum pooling
        # return global_add_pool(node_attr, batch)


class DMPNNEncoder_head(nn.Module):

    def __init__(self, hidden_size, out_dim, node_fdim, edge_fdim, depth=3, bias=True):
        super(DMPNNEncoder_head, self).__init__()
        self.act_func = nn.ReLU()
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=False)
        self.depth = depth

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim, bias=True),
        )
        self.initialize_weights(self)

    def initialize_weights(self, model: nn.Module) -> None:
        """
        Initializes the weights of a model in place.
        :param model: An PyTorch model.
        """
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def forward(self, data):
        # directed message passing over edges
        x, edge_index, h0, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # logging.info("x: {}".format(x.shape))
        # logging.info("h: {}".format(h0.shape))

        h = h0

        for i in range(self.depth - 1):
            logging.info("i: {}".format(i))
            m = directed_mp(h, edge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(x, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global sum pooling
        x = global_add_pool(node_attr, batch)
        x = self.head(x)
        logging.info("x: {}".format(x.shape))
        return x
