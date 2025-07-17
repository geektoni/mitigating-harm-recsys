import pandas as pd
import numpy as np
import torch
from tqdm import tqdm 

from torch.utils.data import Dataset

def deg_dist(train, num_v):
    uni, cou = np.unique(train[1].values,return_counts=True)
    cou = cou**(0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)

class SiReNDataset(Dataset):

    def __init__(self, train_file, test_file, offset=1, K=40, device="cpu"):
        
        self.train_data = pd.read_table(train_file, header=None, sep=' ')
        self.test_data = pd.read_table(test_file, header=None, sep=' ')

        self.train_data = self.train_data.astype({0:'int64', 1:'int64'})

        # Get how many unique users and items there are
        # It expects the items and users to be numbered starting from 1
        self.num_users = np.unique(
            self.train_data[0].values.tolist()+self.test_data[0].values.tolist()
        ).max()+1
        self.num_items = np.unique(
            self.train_data[1].values.tolist()+self.test_data[1].values.tolist()
        ).max()+1

        # Generate negative candidates
        self.neg_dist = deg_dist(self.train_data,
                                 self.num_items)

        # Generate information for training
        self.edge_1 = torch.tensor(self.train_data[0].values)
        self.edge_2 = torch.tensor(self.train_data[1].values) + self.num_users
        self.edge_3 = torch.tensor(self.train_data[2].values) - offset
        self.K = K
        self.total_items = np.arange(self.num_items)

        self.device=device
    
    def negs_gen_(self):

        print('negative sampling...')
        
        self.edge_4 = torch.empty((len(self.edge_1),self.K),dtype=torch.long)
        
        for j in set(self.train_data[0].values):
            pos=self.train_data[self.train_data[0]==j][1].values
            neg = np.setdiff1d(self.total_items,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_users).long()
            self.edge_4[self.edge_1==j]=temp.view(int(len(temp)/self.K),self.K)
        
        self.edge_4 = torch.tensor(self.edge_4).long()
        
        print('complete !')
    
    def negs_gen_EP(self,epoch):

        self.edge_4_tot = torch.empty((len(self.edge_1),self.K,epoch),dtype=torch.long)
        
        for j in tqdm(set(self.train_data[0].values)):
            pos=self.train_data[self.train_data[0]==j][1].values
            neg = np.setdiff1d(self.total_items,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K*epoch,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_users).long()
            self.edge_4_tot[self.edge_1==j]=temp.view(int(len(temp)/self.K/epoch),self.K,epoch)
        
        #self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        self.edge_4_tot = self.edge_4_tot.long()

    def __len__(self):
        return len(self.edge_1)
    
    def __getitem__(self,idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u,v,w,negs
