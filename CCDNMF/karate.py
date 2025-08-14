import os
import random
import numpy as np
import linecache
import torch
from time import perf_counter as t
from Utils.evaluate import clusterscores
from Dataset.dataset import Dataset
from Model.my_model import Model
from Pretrainer.pretrainer import PreTrainer
import networkx as nx   


def train(model: Model, graph, optimizer):
    optimizer.zero_grad()
    V = model()

    loss, loss1, loss2, loss3, loss4, loss5 = model.loss(graph)
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(V.detach().cpu().numpy(), axis=0)
    y_true = graph.L.detach().cpu().numpy()
    # print(y_pred)
    scores = clusterscores(y_pred, y_true)

    return loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), scores

def find_clique( graph, node):
        A = graph.A.cpu().numpy() 
        G = nx.from_numpy_array(A)
        cliques = list(nx.find_cliques(G))
        node_cliques = [clique for clique in cliques if node in clique and 3<= len(clique) <= 5]
        if node_cliques:
            return max(node_cliques, key=len)
        else:
            return []
        
if __name__=='__main__':

    random.seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_config = {'graph_file': './Database/karate/edge.txt',
                      'label_file': './Database/karate/group.txt',
                      'device': device}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [20, 10, 2],
        'net_input_dim': graph.num_nodes,
        'seed': 42,
        'pre_iterations': 100,
        'pretrain_params_path': './Log/karate/pretrain_params.pkl'}

    model_config = {
        'device': device,
        'net_shape': [20, 10, 2],
        'net_input_dim': graph.num_nodes,
        'is_init': True,
        'pretrain_params_path': './Log/karate/pretrain_params.pkl',
        'tau': 1.3,
        'conc': 5,
        'negc': 400,
        'rec': 1,
        'mi': 20,
        'me': 1,
        'ma': 1,
        'r': 3,
        'learning_rate': 0.01,
        'weight_decay': 0.00001,
        'epoch': 600,
        'run': 20,
        'model_path': './Log/cora/cora_model.pkl',
        'neighbor': [graph.A[i].nonzero().squeeze() for i in range(graph.num_nodes)]
    }

    # 'Pre-training stage'
    # pretrainer = PreTrainer(pretrain_config)
    # pretrainer.pre_training(graph.A.detach().cpu().numpy(), 'net')
    # # pretrainer.pre_training(graph.X.t().detach().cpu().numpy(), 'att')

    learning_rate = model_config['learning_rate']
    weight_decay = model_config['weight_decay']


    start = t()
    prev = start

    M = []
    N = []
    P = []
    F = []
    # 'Fine-tuning stage'
    for i in range(1):

        model = Model(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, model_config['epoch']):
            loss, loss1, loss2, loss3, loss4, loss5, scores = train(model, graph, optimizer)

            now = t()
            prev = now

        M.append(scores['ACC'])
        N.append(scores['NMI'])
        P.append(scores['PUR'])
        F.append(scores['F_score'])
        
    print('t = ' , now - start)
    print("F-score : %f±%f" %(np.mean(F) , np.var(F)))
    print("acc : %f±%f" %(np.mean(M) , np.var(M)))
    print("NMI : %f±%f" %(np.mean(N) , np.var(N)))
    print("PUR : %f±%f" %(np.mean(P) , np.var(P)))
    print("=== Final ===")
