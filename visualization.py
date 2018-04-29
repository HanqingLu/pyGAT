import matplotlib
matplotlib.use('Agg')
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import load_data, accuracy
from models import GAT
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from ggplot import *
# import pandas as pd
from utils_tf import load_data_gat


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test, f1_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1 score= {:.4f}".format(f1_test))


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data_gat(path='data/', dataset_str='citeseer')

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    model = GAT(nfeat=features.shape[1], nhid=12, nclass=int(labels.max()) + 1, dropout=0.6, nheads=8, alpha=0.2)

    model.load_state_dict(torch.load('693.pkl'))
    compute_test()

    # x = torch.cat([att(features, adj) for att in model.attentions], dim=1)
    # embeddings = x.cpu().data.numpy()
    # time_start = time.time()
    #
    # pca_50 = PCA(n_components=50)
    # pca_result_50 = pca_50.fit_transform(embeddings)
    #
    # tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    # tsne_results = tsne.fit_transform(pca_result_50)
    #
    # feat_cols = ['tsne-one', 'tsne-two']
    # df = pd.DataFrame(tsne_results, columns=feat_cols)
    # df['labels'] = labels.cpu().data.numpy()
    # df['labels'] = df['labels'].apply(lambda i: str(i))
    # print(df.shape)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #
    # chart = ggplot(df, aes(x='tsne-one', y='tsne-two', color='labels')) \
    #         + geom_point(size=75, alpha=0.8) \
    #         + ggtitle("First and Second TSNE Components colored by digit")
    #
    # chart.save(filename='plot.png')
