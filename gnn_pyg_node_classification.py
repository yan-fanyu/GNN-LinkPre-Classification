# -*- coding: utf-8 -*-
"""gnn_pyg_implementations.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ksca_p4XrZjeN0A6jT5aYN6ARvwFVSbY

# Graph Neural Networks with PyG on Node Classification, Link Prediction, and Anomaly Detection

In this notebook, we will review PyG code implementations on major graph problems including node classification, link prediction, and anomaly detection.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
import torch
version = torch.__version__
i = version.find('+')
version = version[:i-1] + '0' + version[i:]
#url = 'https://data.pyg.org/whl/torch-' + version + '.html'
print("torch version:", version)
#print("url:", url)

#!pip install torch-scatter -f $url
# !pip install torch-sparse -f $url
# !pip install torch-geometric
# !pip install torch-cluster -f $url
# !pip install pygod
# !pip install --upgrade scipy

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
We are using [Cora dataset](https://paperswithcode.com/dataset/cora) for the following GCN implementations. The Cora dataset is a paper citation network data that consists of 2,708 scientific publications. Each node in the graph represents each publication and a pair of nodes is connected with an edge if one paper cites the other.

Through this notebook, we are using [PyG (Pytorch Geometric)](https://www.pyg.org/) to implement GCN which is one of the popular GNN libraries. The Cora dataset can also be loaded using PyG module.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
#
from torch_geometric.datasets import Planetoid
#
dataset = Planetoid(root='./data/Cora', name='Cora')

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

"""The node features and the edge information look like below. The node features are 1433 word vectors indicating the absence (0) or the presence (1) of the words in each publication. The edges are represented in adjacency lists."""

graph.x

graph.edge_index.T

"""Each node has one of seven classes which is going to be our model target/label."""

"""
Class Definition
0: Theory
1: Reinforcement_Learning
2: Genetic_Algorithms
3: Neural_Networks
4: Probabilistic_Methods
5: Case_Based
6: Rule_Learning
"""

print("Class Distribution:")
print(sorted(Counter(graph.y.tolist()).items()))

"""## Visualizing Graph
The graph data can be visualized using [NetworkX](https://networkx.org/) library. The node colors represent the node classes.
"""

def convert_to_networkx(graph, n_sample=None):

    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.cpu().numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y


def plot_graph(g, y):

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.show()

g, y = convert_to_networkx(graph, n_sample=1000)
plot_graph(g, y)

"""# 2. Node Classification

For the node classification problem, we are splitting the nodes into train, valid, and test using the `RandomNodeSplit` 
module from PyG (we are replacing the original split masks in the data as it has a too small train set).
"""

split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)  # resetting data split
print("graph:\n", graph)
print("graph.train_mask.shape, train_mask:\n", graph.train_mask.shape, graph.train_mask)

print(
    f"train: {int(graph.train_mask.sum())}, ",
    f"val: {int(graph.val_mask.sum())}, ",
    f"test: {int(graph.test_mask.sum())}",
)

"""Please note the data splits are written into `mask` attributes in the graph object instead of splitting the graph itself. 
Those masks are only used for training loss calculation and model evaluation, and graph convolutions use entire graph data.

## 2-1. Baseline MLP model
Before we build GCN, we are training MLP (multi-layer perceptron, i.e. feed-forward neural nets) only using node features to set a baseline performance. The model ignores the node connections (or the graph structure) and tries to classify the node labels only using the word vectors. The model class looks like below. It has two hidden layers (`Linear`) with ReLU activations followed by an output layer.
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(dataset.num_node_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, dataset.num_classes)
        )

    def forward(self, data):
        x = data.x  # only using node features (x)
        output = self.layers(x)
        return output

"""We are defining training and evaluation functions with a normal Pytorch train/eval setup."""

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = graph.to(device)
mlp = MLP().to(device)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

mlp = train_node_classifier(mlp, graph, optimizer_mlp, criterion, n_epochs=150)

test_acc = eval_node_classifier(mlp, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}')

"""## 2-2. GCN
Next, we are training GCN and comparing its performance to MLP. We are using a very simple model having two graph convolution layers and ReLU activation between them. This setup is the same as [the original paper](https://arxiv.org/pdf/1609.02907.pdf) (equation 9).

"""

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index) #输入图数据：节点+边
        x = F.relu(x)
        output = self.conv2(x, edge_index) #输入节点嵌入x+边

        return output

gcn = GCN().to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)

test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}')

"""We achieved around 15% accuracy improvement from MLP.

### Visualizing classification result
"""

def visualize_classification_result(model, graph):

    model.eval()
    pred = model(graph).argmax(dim=1)
    corrects = (pred[graph.test_mask] == graph.y[graph.test_mask]).cpu().numpy().astype(int)
    test_index = np.arange(len(graph.x))[graph.test_mask.cpu().numpy()]
    g, y = convert_to_networkx(graph)
    g_test = g.subgraph(test_index)

    print("yellow node: correct \npurple node: wrong")
    plot_graph(g_test, corrects)

visualize_classification_result(gcn, graph)
