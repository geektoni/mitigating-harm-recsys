import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class LightGCLDataLoader:

    def __init__(self,
                 train_path,
                 test_path,
                 q=5,
                 batch_size=1024,
                 device="cpu"):
        
        # import pickle
        # f = open('raw/LightGCL/data/gowalla/trnMat.pkl','rb')
        # train = pickle.load(f)
        # print(type(train))
        # exit()
        
        self.q = q
        self.batch_size=batch_size
        self.device=device
        
        self.train_data = pd.read_table(
            train_path, header=None, sep=' '
        )
        self.test_data = pd.read_table(
            test_path, header=None, sep=' '
        )

        self.num_users = max(
            self.train_data[0].max(),
            self.test_data[0].max(),
        )+1
        self.num_items = max(
            self.train_data[1].max(),
            self.test_data[1].max(),
        )+1

        # Consider only positive elements
        self.train_data = self.train_data[self.train_data[2] == 0]
        
        # Build a CSR matrix from the training data
        rows = self.train_data[0].values
        cols = self.train_data[1].values
        values = [1] * len(self.train_data)  # Default binary presence

        self.train_ajdm = csr_matrix((values, (rows, cols)), 
                            shape=(self.num_users, self.num_items)).tocoo()
        del rows, cols, values

        self.train_csr = (self.train_ajdm!=0).astype(np.float32)

        # Perform other operations
        self.normalize_adj_matrix()
        self.train_ajdm = self.train_ajdm.tocoo()

        # Build the dataloader
        self.train_loader = data.DataLoader(
            TrnData(self.train_ajdm),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)
        
        self.adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(self.train_ajdm)
        self.adj_norm = self.adj_norm.coalesce().to(self.device)
        print('Adj matrix normalized.')

        # perform svd reconstruction
        adj = scipy_sparse_mat_to_torch_sparse_tensor(self.train_ajdm).coalesce().to(self.device)
        print('Performing SVD...')
        self.svd_u, s, self.svd_v = torch.svd_lowrank(adj, q=self.q)
        self.u_mul_s = self.svd_u @ (torch.diag(s))
        self.v_mul_s = self.svd_v @ (torch.diag(s))
        del s
        print('SVD done.')



    def normalize_adj_matrix(self):

        rowD = np.array(self.train_ajdm.sum(1)).squeeze()
        colD = np.array(self.train_ajdm.sum(0)).squeeze()
        for i in range(len(self.train_ajdm.data)):
            self.train_ajdm.data[i] = self.train_ajdm.data[i] / pow(rowD[self.train_ajdm.row[i]]*colD[self.train_ajdm.col[i]], 0.5)


        
        

        