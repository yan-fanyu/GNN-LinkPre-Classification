
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from collections import Counter

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings("ignore")

def make_deterministic(random_seed = 123):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

make_deterministic()

"""# 1. Loading Data
We are using [KarateClub dataset]) for the following GCN implementations. 
Through this notebook, we are using [PyG (Pytorch Geometric)](https://www.pyg.org/) to implement GCN which is one of the popular GNN libraries. 
"""

#from torch_geometric.datasets import KarateClub
#dataset = KarateClub()

from torch_geometric.datasets import Airports
dataset = Airports(root='./data/Airports', name='USA')

def show_dataset_stats(dataset):
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of node classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")

show_dataset_stats(dataset)

def show_graph_stats(graph):
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of node features: {graph.x.shape[1]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")

graph = dataset[0]
show_graph_stats(graph)

#The node features and the edge information look like below.

#print("graph.x:\n",graph.x)
#print("graph.edge_index.T:\n", graph.edge_index.T)

"""Each node has one of 4 classes """

print("Class Distribution:")
print(sorted(Counter(graph.y.tolist()).items()))

"""## Visualizing Graph
The graph data can be visualized using [NetworkX](https://networkx.org/) library. 
The node colors represent the node classes."""


def convert_to_networkx(graph, n_sample=None):

    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        print("sample_nodes:\n", sampled_nodes)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y


def plot_graph(g, y):

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.show()


g, y = convert_to_networkx(graph, n_sample=500)
plot_graph(g, y)

from torch_geometric.nn import GCNConv
"""# 3. Link prediction
Link prediction is trickier than node classification as we need some tweaks to make predictions on edges using node embeddings. The prediction steps are described below:

1.   An encoder creates node embeddings by processing the graph with two
convolution layers.
2.   We randomly add negative links to the original graph. This makes the model task a binary classification with the positive links from the original edges and the negative links from the added edges.
3.   A decoder makes link predictions (i.e. binary classifications) on all the edges including the negative links using node embeddings. It calculates a dot product of the node embeddings from pair of nodes on each edge. Then, it aggregates the values across the embedding dimension and creates a single value on every edge that represents the probability of edge existence.

This setup is from [the original link prediction implementation in Variational Graph Auto-Encoders](https://github.com/tkipf/gae). The code looks like something below. This is adapted from [the code example in PyG repo](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py) which is based on the Graph Auto-Encoders implementation.
"""


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs=100
):

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

"""For this link prediction task, we want to randomly split links/edges into train, valid, and test data. We can use the `RandomLinkSplit` module from PyG to do that."""

split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph)

"""The output data looks like something below."""

print('train_data:', train_data)
print('val_data:', val_data)
print('test_data:', test_data)
#print("train_data.edge_label:\n", train_data.edge_label)

"""There are several things to note about this output data.

First, the split is performed on `edge_index` such that the training and the validation splits do not include the edges from the validation and the test split (i.e. only have the edges from the training split), and the test split does not include the edges from the test split. This is because `edge_index` (and `x`) is used for the encoder to create node embeddings, and this setup ensures that there are no target leaks on the node embeddings when it makes predictions on the validation/test data.

Second, two new attributes (`edge_label` and `edge_label_index`) are added to each split data. They are the edge labels and the edge indices corresponding to each split. `edge_label_index` will be used for the decoder to make predictions and `edge_label` will be used for model evaluation.

Third, negative links are added to both `val_data` and `test_data` with the same number as the positive links (`neg_sampling_ratio=1.0`). They are added to `edge_label` and `edge_label_index` attributes, but not added to `edge_index` as we do not want to use the negative links on the encoder (or node embedding creation). And also, we are not adding negative links to the training set here (with `add_negative_train_samples=False`) as we will add them during the training loop in `train_link_predictor` above. This randomization during training makes the model more robust.

The image below summarizes how this edge split is performed for the encoder and the decoder (the colored edges are used in each stage).


We can now train and evaluate the model with the following code.
"""
device = torch.device('cpu')
model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")


def visualize_link_prediction_result(model, data):

    with torch.no_grad():
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
#       print("out:\n", out.cpu().numpy())
#       print("edge_label:\n", data.edge_label.cpu().numpy())
#       print("data.edge_label_index.T:\n",data.edge_label_index.T)
    edge_label_index_t = data.edge_label_index.T.cpu().numpy()
#   print("edge_label_index_T:\n", edge_label_index_T)
    edge_label_tuples = [tuple(row) for row in edge_label_index_t]
#   print("edge_label_tuples:\n", edge_label_tuples)

    out_int = [int(round(x)) for x in out.cpu().numpy()]
    print("out_int:\n", out_int)
    edge_label_int = [int(x) for x in data.edge_label.cpu().numpy()]
    print("edge_label_int:\n", edge_label_int)
#   print("data_index:\n", data.edge_index)

    # 构造测试图
    G = nx.Graph()
    # 添加边到图中
    for edge in edge_label_tuples:
        G.add_edge(edge[0], edge[1])
    # 打印图的一些基本信息
    print("节点数: ", G.number_of_nodes())
    print("边数: ", G.number_of_edges())
    print("G.nodes:\n", G.nodes)
    data_y = data.y.numpy()
    y = data_y[G.nodes]
    print("节点颜色数: ", len(y))
    # 标注预测结果的颜色：True Positive--red, True Negative--pink, False Positive--blue, False Negative--green
    color_table = [['pink', 'green'], ['blue', 'red']]
    edge_label_colors = dict()
    for i in range(len(out_int)):
        edge_label_colors[edge_label_tuples[i]] = color_table[out_int[i]][edge_label_int[i]]
        edge_label_colors[(edge_label_tuples[i][1], edge_label_tuples[i][0])] = color_table[out_int[i]][edge_label_int[i]]
    print("edge_label_colors:\n", edge_label_colors)
    print("red and pink edge: correct \n blue and green edge: wrong")
#   print("data.edge_index:\n", data.edge_index)
    #print("G.nodes:\n", G.nodes)
    #print("G.edges:\n", G.edges)
    G.add_edges_from(edge_label_tuples)
    #print("G.edges:\n", G.edges)

    plot_graph_edge(G, edge_label_colors, y)


def plot_graph_edge(g, edge_label_colors, y):

    plt.figure(figsize=(9, 7))
    nx.draw(g, node_size=30, node_color=y, edge_color=[edge_label_colors.get(edge, 'gray') for edge in g.edges()])
    plt.show()


visualize_link_prediction_result(model, test_data)



