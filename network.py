import numpy as np
from scipy.special import softmax

def RemoveFrom(l, xs):
    for x in xs:
        try:
            l.remove(x)
        except:
            raise Exception(x)

class Network:
    def __init__(self, edges, dists = None, symmetry = False, gamma=1., softmax=False, polynomial=False):
        """
        edges dict: 
            key: index of node, int
            val: indices of linked nodes (incoming), list of ints
        dists dict:
            key: index of node, int
            val: dists of linked nodes, array of floats
        neighbours_dict dict:
            key: index of node, int
            val: list of lists of ints
        symmetry bool:
            Enforce symmetry on the network if not. 
        gamma float:
            Dist to weights transformation parameter
        softmax bool:
            Whether to use softmax normalization for weights
        """
        self.nodes = list(edges.keys())
        self.size = len(self.nodes)
        self.gamma = gamma
        self.softmax = softmax
        self.polynomial = polynomial
        self.edges = edges
        if not (dists is None):
            self.dists = dists
        else:
            # init equal dists
            self.dists = {key: np.array([1 for _ in value]) for key, value in edges.items()}
        
        if symmetry:
            for key in self.edges.keys():
                for i in range(len(self.edges[key])):
                    node = self.edges[key][i]
                    if not node in self.edges.keys():
                        self.edges[node]=[]
                        self.dists[node] = np.array([])
                    if key not in self.edges[node]:
                        self.edges[node].append(key)
                        self.dists[node] = np.append(self.dists[node], self.dists[key][i])
                        
        self.neighbours_dict, self.neighbour_weights_dict, self.neighbour_dists_dict = {},{},{}
        for key, value in edges.items():
            neighbours, neighbour_weights, neighbour_dists = self.ComputeNeighbours(key)
            self.neighbours_dict[key] = neighbours
            self.neighbour_weights_dict[key] = neighbour_weights
            self.neighbour_dists_dict[key] = neighbour_dists
        self.adj_mat = self.AdjMatrix()
        self.w_mats = self.WeightMatrix()
        self.d_mats = self.DistMatrix()
            
    def AdjMatrix(self):
        """
        return the adjacency matrix
        """
        mat = np.zeros((self.size,self.size))
        for i in range(self.size):
            mat[([j for j in self.edges[i]],[i for j in self.edges[i]])] = 1
        return mat
        
        
    def ComputeNeighbours(self, node):
        neighbours = [self.edges[node]]
        neighbour_weights = [self.DistToWeights(self.dists[node])]
        neighbour_dists = [self.dists[node]]
        nodes_left = self.nodes[:] # nodes that are not yet connected to this node
        RemoveFrom(nodes_left,neighbours[-1]+[node])
        prev_dists = self.dists[node]
        while True:
            stage_neighbours = []
            stage_dists = {}
            for i in range(len(neighbours[-1])):
                # for each stage_r-1 neighbour, find its neighbours
                prev_neighbour = neighbours[-1][i]
                for j in range(len(self.edges[prev_neighbour])):
                    stage_neighbour = self.edges[prev_neighbour][j]
                    if (stage_neighbour in nodes_left) & (stage_neighbour!=node):
                    # if the stage-r neighbour is not in previous stage(still left unconnected) and is not self
                        if not (stage_neighbour in stage_neighbours):
                        # if stage neighbour not added to stage-r neighbourhood, add to neighbourhood 
                        # and add dist(node to stage neighbour) = min_dist(node to prev neighbour)+dist(prev neighbour to stage neighbour)
                            stage_neighbours.append(stage_neighbour)
                            stage_dists[stage_neighbour] = [prev_dists[i]+self.dists[prev_neighbour][j]]
                        else:
                        # add dist(...) = ....
                            stage_dists[stage_neighbour].append(prev_dists[i]+self.dists[prev_neighbour][j])
            if len(stage_neighbours) == 0:
                # if no new neighbours detected, end and return
                return neighbours, neighbour_weights, neighbour_dists
            # for each stage-r neighbour, find the min_dist(node to stage neighbour)
            stage_dists = np.array([min(stage_dists[stage_neighbour]) for stage_neighbour in stage_neighbours])
            RemoveFrom(nodes_left, stage_neighbours) # update nodes not connected
            neighbours.append(stage_neighbours)
            neighbour_weights.append(self.DistToWeights(stage_dists))
            neighbour_dists.append(stage_dists)
            prev_dists = stage_dists

    def DistToWeights(self, dists):
        if self.softmax:
            if len(dists) != 0:
                weights = softmax(-self.gamma*dists)
            else:
                weights = np.exp(dists)
    
        elif self.polynomial:
            if len(dists) != 0:
                dists = dists / np.max(dists)
                weights = 1 - dists**self.gamma + 1e-6
                weights = weights / sum(weights)
            else:
                weights = 1 - dists
    
        else:
            if len(dists) == 0:
                return dists
    
            if np.any(dists <= 0):
                raise ValueError("Distances must be strictly positive.")
    
            if self.gamma < 50:
                # fast original version
                weights = dists**(-self.gamma)
                weights = weights / sum(weights)
            else:
                # numerically stable version for large gamma
                log_weights = -self.gamma * np.log(dists)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                weights = weights / sum(weights)
    
        return weights

    def WeightMatrix(self):
        max_stage = max([len(self.neighbours_dict[node]) for node in self.nodes])
        w_mats = [np.identity(self.size)]
        for r in range(max_stage):
            mat = np.zeros((self.size,self.size))
            for i in range(self.size):
                if len(self.neighbours_dict[i]) > r:
                    mat[([j for j in self.neighbours_dict[i][r]],[i for j in self.neighbours_dict[i][r]])] = self.neighbour_weights_dict[i][r]
            w_mats.append(mat)
        return np.array(w_mats)

    def DistMatrix(self):
        max_stage = max([len(self.neighbours_dict[node]) for node in self.nodes])
        d_mats = []
        for r in range(max_stage):
            mat = np.zeros((self.size,self.size))
            for i in range(self.size):
                if len(self.neighbours_dict[i]) > r:
                    mat[([j for j in self.neighbours_dict[i][r]],[i for j in self.neighbours_dict[i][r]])] = self.neighbour_dists_dict[i][r]
            d_mats.append(mat)
        return np.array(d_mats)

    def UpdateGamma(self,gamma):
        self.gamma = gamma
        for key in self.neighbour_weights_dict.keys():
            for r in range(len(self.neighbour_weights_dict[key])):
                self.neighbour_weights_dict[key][r] = self.DistToWeights(self.neighbour_dists_dict[key][r])
        self.w_mats = self.WeightMatrix()
                

def adj_mat_to_dict(adj_mat):
    dict = {}
    for i in range(adj_mat.shape[0]):
        dict[i] = np.argwhere(adj_mat[i]).reshape(-1).tolist()
    return dict

import random

def full_network(N):
    edges = {}
    indices = [i for i in range(N)]
    for i in range(N):
        indices.remove(i)
        edges[i] = indices[:]
        indices.append(i)
    return Network(edges,symmetry=False)

def random_network(N):
    edges = {}
    indices = [i for i in range(N)]
    for i in range(N):
        indices.remove(i)
        edges[i] = random.sample(indices,random.randint(int(N/4),int(N/4*3)))
        indices.append(i)
    return Network(edges,symmetry=False)

def random_network_2(N,n):
    edges = {}
    indices = [i for i in range(N)]
    for i in range(N):
        indices.remove(i)
        edges[i] = random.sample(indices,n)
        indices.append(i)
    return Network(edges,symmetry=False)

def tridiagonal_network(N):
    edges = {}
    edges[0] = [1,N-1]
    edges[N-1] = [N-2,0]
    for i in range(1,N-1):
        edges[i] = [i-1,i+1]
    return Network(edges,symmetry=True)

def Adj_compare(A_true, A_pred):
    n,_ = A_true.shape
    TT, TF, FT, FF = 0,0,0,0
    for i in range(n):
        for j in range(i+1,n):
            if A_true[i,j] == 1:
                if A_pred[i,j] == 1 or A_pred[j,i]==1:
                    TT += 1
                else:
                    TF += 1
            else:
                if A_pred[i,j] == 1 or A_pred[j,i]==1:
                    FT += 1
                else:
                    FF += 1
    return [TT,TF,FT,FF]
    
def get_edges_dists(dist_mat,dist_thresh):
    N = len(dist_mat)
    dists= {}
    edges= {}
    for i in range(N):
        edges[i] = []
        dists[i] = []
        for j in range(N):
            if i==j:
                continue
            if dist_mat[i,j] <= dist_thresh:
                edges[i].append(j)
                dists[i].append(dist_mat[i,j])
        dists[i] = np.array(dists[i])
    return edges,dists
    
#edges = np.genfromtxt("C:\\Users\\LHX\\Documents\\Network Time Series\\Network count time series\\Network.csv",delimiter=",")[1:,1:]
#edges = {key: (edges[key,:][~np.isnan(edges[key,])]-1).astype(int).tolist() for key in range(edges.shape[0])}

#network = Network(edges,symmetry=True)
#vts = np.genfromtxt("C:\\Users\\LHX\\Documents\\Network Time Series\\Network count time series\\vts.csv",delimiter=",")[1:,1:]
#vts = vts[~np.isnan(vts).any(axis=1), :]
#vts_train = vts[:700]
#vts_test = vts[700:]

n8_14=Network({0:[1,3,4],1:[0,6],2:[3,4],3:[2,4],4:[0,2],5:[0,1,4],6:[3],7:[4,6,2]},symmetry=True)
n8_7=Network({0:[1],1:[0],2:[4],3:[4],4:[2],5:[0,1],6:[3],7:[6]},symmetry=True)
n8_7_2=Network({0:[6,7],1:[3,4],2:[5],3:[1,5],4:[1,6],5:[2,3],6:[0,4],7:[0]},symmetry=True)