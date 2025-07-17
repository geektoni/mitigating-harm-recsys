import pickle
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx
import pandas as pd

class GFormerDataLoader:
    def __init__(self, train_file, test_file, anchor_set_num=32, device="cpu", batch_size=512):
        self.device = device

        self.trnfile = train_file
        self.tstfile = test_file

        self.num_users = 0
        self.num_items = 0
        self.anchor_set_num = anchor_set_num

        self.batch_size = batch_size

        self.LoadData()

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff=None)
        return dists_dict

    def get_random_anchorset(self):
        n = self.num_nodes
        annchorset_id = np.random.choice(n, size=self.anchor_set_num, replace=False)
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(self.num_users + self.num_items))

        rows = self.allOneAdj._indices()[0, :]
        cols = self.allOneAdj._indices()[1, :]

        rows = np.array(rows.cpu())
        cols = np.array(cols.cpu())

        edge_pair = list(zip(rows, cols))
        graph.add_edges_from(edge_pair)
        dists_array = np.zeros((len(annchorset_id), self.num_nodes))

        dicts_dict = self.single_source_shortest_path_length_range(graph, annchorset_id, None)
        for i, node_i in enumerate(annchorset_id):
            shortest_dist = dicts_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[i, j] = 1 / (dist + 1)
        self.dists_array = dists_array
        self.anchorset_id = annchorset_id #

    def preSelect_anchor_set(self):
        self.num_nodes = self.num_users + self.num_items
        self.get_random_anchorset()

    def loadOneFile(self, filename_train, filename_test):

        ret_train = pd.read_table(
            filename_train, header=None, sep=' '
        )
        ret_test = pd.read_table(
            filename_test, header=None, sep=' '
        )

        # Get unique items and users
        # Assume they are indexed from 0
        # We do this before filtering, otherwise we have
        # less items in the end
        unique_users = max(
            ret_train[0].max()+1,
            ret_test[0].max()+1)
        unique_items = max(
            ret_train[1].max()+1,
            ret_test[1].max()+1)

        ret_train = ret_train[ret_train[2] == 0]
        ret_test = ret_test[ret_test[2] == 0] # Pick only positive edges

        if type(ret_train) != coo_matrix:
            ret_train = sp.coo_matrix(
                ( [1] * len(ret_train), (ret_train[0], ret_train[1]) ), 
            shape=(unique_users, unique_items)
            ).astype(np.float32)
            ret_test = sp.coo_matrix(
                ( [1] * len(ret_test), (ret_test[0], ret_test[1]) ), 
            shape=(unique_users, unique_items)
            ).astype(np.float32)
            #ret = sp.coo_matrix(ret)
        return ret_train, ret_test, unique_users, unique_items

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, trainMat):

        a = sp.csr_matrix((self.num_users, self.num_users))
        b = sp.csr_matrix((self.num_items, self.num_items))
        mat = sp.vstack([sp.hstack([a, trainMat]), sp.hstack([trainMat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        #return t.sparse.FloatTensor(idxs, vals, shape).cuda()
        return t.sparse_coo_tensor(idxs, vals, shape).to(self.device)

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = t.ones_like(torchAdj._values())
        shape = torchAdj.shape
        #return t.sparse.FloatTensor(idxs, vals, shape).to(self.device)
        return t.sparse_coo_tensor(idxs, vals, shape).to(self.device)

    def LoadData(self):
        trnMat, tstMat, self.num_users, self.num_items = self.loadOneFile(self.trnfile, self.tstfile)

        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)
        trnData = TrnData(trnMat, num_items=self.num_items)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=self.batch_size, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=self.batch_size, shuffle=False, num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, coomat, num_items):
        self.num_items = num_items
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.num_items)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
