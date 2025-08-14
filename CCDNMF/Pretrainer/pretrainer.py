from sklearn.decomposition import NMF
from tqdm import tqdm
import pickle
import networkx as nx
import torch



class PreTrainer(object):

    def __init__(self, config):
        self.config = config
        self.net_shape = config['net_shape']
        self.pretrain_params_path = config['pretrain_params_path']
        self.seed = config['seed']
        self.pre_iterations = config['pre_iterations']

        self.U_init = {}
        self.V_init = {}
        self.clique = []
        self.neighbor = []

    def setup_z(self, i, modal):
            """
            Setup target matrix for pre-training process.
            """
            if i == 0:
                self.Z = self.A
            else:
                self.Z = self.V_init[modal + str(i-1)]
    def find_clique(self, graph, node):
        A = graph
        G = nx.from_numpy_array(A)
        cliques = list(nx.find_cliques(G))
        node_cliques = [clique for clique in cliques if node in clique and 3<= len(clique)]
        if node_cliques:
            all_nodes = set()
            for clique in node_cliques:
                all_nodes.update(clique)
            return list(all_nodes)
        else:
            return []
        
    def sklearn_pretrain(self, i):
            """
            Pretraining a single layer of the model with sklearn.
            :param i: Layer index.
            """
            nmf_model = NMF(n_components=self.layers[i],
                            init="random",
                            random_state=self.seed,
                            max_iter=self.pre_iterations)

            U = nmf_model.fit_transform(self.Z)
            V = nmf_model.components_
            return U, V
    def jaccard_similarity(self, set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) != 0 else 0.0
    def get_neighbors(self, A, i):
        neighbors = torch.nonzero(A[i], as_tuple=True)[0].tolist()
        return neighbors
    def jaccard_similarity_matrix(self, data):
        num_nodes = data.shape[0]
        similarity_matrix = torch.zeros(num_nodes, num_nodes)
            
        for i in range(num_nodes):
            neighbors_i = set(self.get_neighbors(data, i))
            for j in range(i, num_nodes):
                neighbors_j = set(self.get_neighbors(data, j))
                similarity = self.jaccard_similarity(neighbors_i, neighbors_j)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
            
        return similarity_matrix
    def threshold(self, tensor):
        """根据阈值进行相似度矩阵的阈值化处理"""
        # 将二维张量展平为一维
        flat_tensor = tensor.flatten()
    
        # 对展平后的张量进行降序排序
        threshold_value = []
        sorted_tensor, _ = torch.sort(flat_tensor, descending=True)
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:

            num_elements = int(threshold * len(sorted_tensor))
            threshold = sorted_tensor[num_elements]
            threshold_value.append(threshold)

        return threshold_value


    def pre_training(self, data, module):
        self.A = data
        self.layers = self.net_shape
        print("\nLayer pre-training started. \n")   

        for i in tqdm(range(len(self.layers)), desc="Layers trained: ", leave=True):
                self.setup_z(i, module)
                U, V = self.sklearn_pretrain(i)
                name = module + str(i)
                self.U_init[name] = U
                self.V_init[name] = V

        self.clique = [self.find_clique(self.A, i) for i in range(self.A.shape[0])]
        self.similarity_matrix = self.jaccard_similarity_matrix(torch.from_numpy(data))
        self.threshold_value = self.threshold(self.similarity_matrix)

        with open(self.pretrain_params_path, 'wb') as handle:
            pickle.dump([self.U_init, self.V_init, self.clique, self.similarity_matrix, self.threshold_value], handle, protocol=pickle.HIGHEST_PROTOCOL)
