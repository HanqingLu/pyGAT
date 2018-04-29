import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGraphGatedAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphGatedAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.act = nn.Sigmoid()
        self.bnlayer = nn.BatchNorm1d(out_features)
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)

    def forward(self, input, adj):
        '''
        :param input: 2708, 1433
        :param adj: 2708, 2708
        :return:
        '''
        h = torch.mm(input, self.W)  # 2708, 8
        N = h.size()[0]
        idx = adj._indices()
        val = adj._values()
        M = val.size(0)

        h1 = torch.mm(h, self.a1)  # 2708, 8
        h2 = torch.mm(h, self.a2)  # 2708, 8

        h_prime = torch.zeros(N, self.out_features).cuda()
        for k in range(M):
            h_ij = self.act(h1[idx[0, k], :] + h2[idx[1, k], :]) * h[idx[1, k], :]
            h_prime[idx[0, k], :] += val[k] * h_ij

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphGatedAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphGatedAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.act = nn.Sigmoid()
        self.bnlayer = nn.BatchNorm1d(out_features)
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.at = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        # self.att1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, 1).type(
        #     torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
        #                        requires_grad=True)
        # self.att2 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, 1).type(
        #     torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
        #                        requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        '''
        :param input: 2708, 1433
        :param adj: 2708, 2708
        :return:
        '''
        h = torch.mm(input, self.W)  # 2708, 8
        N = h.size()[0]

        h1 = torch.mm(h, self.a1)  # 2708, 8
        h2 = torch.mm(h, self.a2)  # 2708, 8

        e_input = (h1.repeat(1, N).view(N * N, -1) + h2.repeat(N, 1)).view(N, -1, self.out_features)
        # generate all combinations for h1_i and h2_j

        e = self.act(e_input)  # 2708, 2708, 8

        # print(e)
        logits = e * h.unsqueeze(dim=0).repeat(N, 1, 1)  # 2708, 2708, 8

        # edges = torch.mm(logits.view(N * N, -1), self.at).view(N, N)  # 2708, 2708
        # zero_vec = -9e15 * torch.ones_like(edges)
        # attention = torch.where(adj > 0, edges, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.bmm(attention.unsqueeze(dim=1), logits).squeeze(dim=1)  # 2708, 8

        h_prime = torch.bmm(adj.unsqueeze(dim=1), logits).squeeze(dim=1)  # 2708, 8

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        '''
        :param input: 2708, 1433
        :param adj: 2708, 2708
        :return:
        '''
        h = torch.mm(input, self.W)  # 2708, 8
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
