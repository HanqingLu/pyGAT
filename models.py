import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphGatedAttentionLayer, SparseGraphGatedAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # self.MLP = nn.Linear(nhid * nheads, nclass)
        # self.out_att = GraphGatedAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # self.MLP = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # x = self.MLP(x)
        return F.log_softmax(x, dim=1)



