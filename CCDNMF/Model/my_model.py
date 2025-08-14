
import os
import pickle
import random
import torch
import gc
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from itertools import combinations
import networkx as nx



class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.net_shape = config['net_shape']
        self.net_input_dim = config['net_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']
        self.tau = config['tau'] #相似度稀释权重
        self.conc = config['conc']
        self.negc = config['negc']
        self.rec = config['rec']
        self.r = config['r']
        self.mi = config['mi']
        self.me = config['me']
        self.ma = config['ma']
        self.neighbor = config['neighbor']
        self.similarity_matrix = torch.zeros((self.net_input_dim, self.net_input_dim),device = self.device)
        

        self.fc1 = torch.nn.Linear(self.net_shape[-1], self.net_shape[1])
        self.fc2 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.U = torch.nn.ParameterDict({})
        self.V = torch.nn.ParameterDict({})
        self.clique = []
        
        if os.path.isfile(self.pretrain_params_path):
            with open(self.pretrain_params_path, 'rb') as handle:
                self.U_init, self.V_init, self.clique, self.similarity_matrix, self.threshold= pickle.load(handle)

        # self.threshold = config['threshold']

        if self.is_init:
            module = 'net'
            # print(len(self.net_shape))
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))
        else:
            module = 'net'
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name], dtype=torch.float32)))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name], dtype=torch.float32)))

    def projection1(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z.t()))
        return self.fc2(z)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        self.index_net = torch.argmax(self.V1, dim=0).long()
        O_net = F.one_hot(self.index_net, self.net_shape[-1]).float()
        S_net = torch.mm(O_net, O_net.t())
        refl_pos = refl_sim * S_net
        return -torch.log((between_sim.diag()) / (refl_sim.sum(1) - refl_pos.sum(1) + between_sim.diag()))
    
    def loss_micro(self, graph, z1: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        loss_micro = torch.tensor(0.0, device=self.device)
        cliques = self.clique

        clique_mask = torch.zeros(self.net_input_dim, self.net_input_dim, device=self.device)
        for i, list_clique in enumerate(cliques):
            if list_clique:
                clique_mask[i, list_clique] = 1
        pos_sim = refl_sim * clique_mask

        neg_sim = refl_sim * (1-clique_mask)

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)
        contrastiveloss = pos_sum / (pos_sum + neg_sum)
        contrastiveloss = torch.where(contrastiveloss == 0, torch.tensor(1.0), contrastiveloss)
        loss_micro = torch.log(contrastiveloss).sum()

        return -loss_micro/self.net_input_dim

    def loss_meso(self, graph, z1: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        loss_meso = torch.tensor(0.0, device=self.device)

        neighbors = self.neighbor
        neighbors = [neigh.unsqueeze(0) if neigh.dim() == 0 else neigh for neigh in neighbors] 

        neighbor_mask = torch.zeros(self.net_input_dim, self.net_input_dim, device=self.device)
        for i, neigh in enumerate(neighbors):
            neighbor_mask[i, neigh] = 1

        pos_sim = refl_sim * neighbor_mask
        neg_sim = refl_sim * (1 - neighbor_mask)

        # 计算损失
        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)
        contrastiveloss = pos_sum / (pos_sum + neg_sum)
        contrastiveloss = torch.where(contrastiveloss == 0, torch.tensor(1.0), contrastiveloss)
        loss_meso = torch.log(contrastiveloss).sum()

        return - loss_meso/self.net_input_dim

    def loss_macro(self, z1: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        
        self.index_net = torch.argmax(self.V1, dim=0).long()
        O_net = F.one_hot(self.index_net, self.net_shape[-1]).float()
        S_net = torch.matmul(O_net, O_net.t())
        
        neg_sim = refl_sim * (1 - S_net)
        
        pos_sim = refl_sim * S_net
        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        loss_macro = torch.log((pos_sum / (pos_sum + neg_sum))).sum()
        
        return -loss_macro/self.net_input_dim
    
    def loss_sim(self, z1: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        similarity_matrix = self.similarity_matrix
        max_sim = torch.where(similarity_matrix > self.threshold[2], torch.ones_like(similarity_matrix), 
                              torch.zeros_like(similarity_matrix)).to(self.device)
        pos_sim = refl_sim * max_sim
        neg_sim = refl_sim * (1 - max_sim)

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        contrastiveloss = pos_sum / (pos_sum + neg_sum)
        contrastiveloss = torch.where(contrastiveloss == 0, torch.tensor(1.0), contrastiveloss)
        loss_sim = torch.log(contrastiveloss).sum()


        return -loss_sim/self.net_input_dim
    def loss_rand(self, z1: torch.Tensor):   
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        diag = torch.eye(self.net_input_dim, device=self.device)
        pos_sim = refl_sim * diag
        one = torch.ones(self.net_input_dim, self.net_input_dim, device=self.device)
        neg_sim = refl_sim * (one - diag)

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        contrastiveloss = pos_sum / (pos_sum + neg_sum)
        contrastiveloss = torch.where(contrastiveloss == 0, torch.tensor(1.0), contrastiveloss)
        loss_sim = torch.log(contrastiveloss).sum()

        return -loss_sim/self.net_input_dim
    def contra_loss1(self, graph, z1: torch.Tensor):
    
        h1 = self.projection1(z1)
        ret = self.loss_micro(graph, h1)

        return ret
    def contra_loss2(self, graph, z1: torch.Tensor):
        h1 = self.projection1(z1)

        ret = self.loss_meso(graph, h1)

        return ret

    def contra_loss3(self, z1: torch.Tensor):
        h1 = self.projection1(z1)

        ret = self.loss_sim(h1)

        return ret 

    def contra_lossrand(self, z1: torch.Tensor):
        h1 = self.projection1(z1)
        ret = self.loss_macro(h1)

        return ret
    
    def forward(self):
        self.V1 = self.V['net' + str(len(self.net_shape) - 1)]
        return self.V1
    
    def loss(self, graph):
        A = graph.A
        P1 = torch.eye(self.net_input_dim, device=self.device)
       
        for i in range(len(self.net_shape)):
            P1 = torch.mm(P1, self.U['net' + str(i)])
        i = len(self.net_shape) - 1
        P1 = torch.mm(P1, self.V['net' + str(i)])
        loss1 = torch.square(torch.norm(A - P1))
        loss2 = self.contra_loss1(graph, self.V1)
        loss3 = self.contra_loss2(graph, self.V1)
        loss4 = self.contra_loss3(self.V1)

        loss5 = 0
        for i in range(len(self.net_shape)):
            zero1 = torch.zeros_like(self.U['net' + str(i)])
            X1 = torch.where(self.U['net' + str(i)] > 0, zero1, self.U['net' + str(i)])
            loss5 = loss5 + torch.square(torch.norm(X1))
        zero1 = torch.zeros_like(self.V['net' + str(i)])
        X1 = torch.where(self.V['net' + str(i)] > 0, zero1, self.V['net' + str(i)])
        loss5 = loss5 + torch.square(torch.norm(X1))

        loss = self.rec*loss1 + self.mi*loss2 + self.me*loss3 + self.ma*loss4 + self.negc*loss5
        return loss,loss1,loss2,loss3,loss4,loss5

        












