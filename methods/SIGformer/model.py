import torchsparsegradutils
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import structured_negative_sampling

import pickle

# Implementation of a sparse softmax
def sum_norm(indices, values, n):
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.] = 1.
    return values/s[indices[0]]

def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)

class Attention(nn.Module):
    def __init__(self, model="eig+path", sample_hop=3, hidden_dim=64, eigs_dim=64, device="cpu"):

        self.sample_hop = sample_hop
        self.hidden_dim = hidden_dim
        self.eigs_dim = eigs_dim
        self.model_type = model
        self.device = device

        super(Attention, self).__init__()
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.path_emb = nn.Embedding(2**(self.sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
        self.sqrt_dim = 1./torch.sqrt(torch.tensor(self.hidden_dim))
        self.sqrt_eig = 1./torch.sqrt(torch.tensor(self.eigs_dim))
        self.my_parameters = [
            {'params': self.lambda0, 'weight_decay': 1e-2},
            {'params': self.path_emb.parameters()},
        ]

    def forward(self, q, k, v,  indices, eigs, path_type):
        ni, nx, ny, nz = [], [], [], []
        for i, pt in zip(indices, path_type):
            x = torch.mul(q[i[0]], k[i[1]]).sum(dim=-1)*self.sqrt_dim
            nx.append(x)
            if 'eig' in self.model_type:
                if self.eigs_dim == 0:
                    y = torch.zeros(i.shape[1]).to(self.device)
                else:
                    y = torch.mul(eigs[i[0]], eigs[i[1]]).sum(dim=-1)
                ny.append(y)
            if 'path' in self.model_type:
                z = self.path_emb(pt).view(-1)
                nz.append(z)
            ni.append(i)
        i = torch.concat(ni, dim=-1)
        s = []
        s.append(torch.concat(nx, dim=-1))
        if 'eig' in self.model_type:
            s[0] = s[0]+torch.exp(self.lambda0)*torch.concat(ny, dim=-1)
        if 'path' in self.model_type:
            s.append(torch.concat(nz, dim=-1))
        s = [sparse_softmax(i, _, q.shape[0]) for _ in s]
        s = torch.stack(s, dim=1).mean(dim=1)
        return torchsparsegradutils.sparse_mm(torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])), v)


class Encoder(nn.Module):
    def __init__(self, model="eig+path", sample_hop=3, hidden_dim=64, eigs_dim=64):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.self_attention = Attention(model, sample_hop, hidden_dim, eigs_dim)
        self.my_parameters = self.self_attention.my_parameters

    def forward(self, x, indices, eigs, path_type):
        y = F.layer_norm(x, normalized_shape=(self.hidden_dim,))
        y = self.self_attention(
            y, y, y,
            indices,
            eigs,
            path_type)
        return y


class SIGformer(nn.Module):
    def __init__(self, dataset, num_users, num_items,
                 hidden_dim=64, n_layers=5, learning_rate=1e-4,
                 beta=1, lambda_reg=1e-4,
                 device="cpu"):
        super(SIGformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_users = num_users
        self.num_items = num_items
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.device = device
        self.embedding_user = nn.Embedding(self.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.hidden_dim)
        self.learning_rate = learning_rate
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]
        self.layers = []
        for i in range(self.n_layers):
            layer = Encoder(hidden_dim=self.hidden_dim,
                            eigs_dim=self.hidden_dim).to(self.device)
            self.layers.append(layer)
            self.my_parameters.extend(layer.my_parameters)
        
        self._users, self._items = None, None

        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=self.learning_rate)

        self.train_pos_list = dataset.train_pos_list
        self.train_neg_list = dataset.train_neg_list

        self.to(self.device)
        self.computer(dataset)

    def computer(self, dataset):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            indices, paths = dataset.sample()
            all_emb = self.layers[i](all_emb,
                                     indices,
                                     dataset.L_eigs,
                                     paths)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self._users, self._items = torch.split(light_out, [self.num_users, self.num_items])
    
    def predict(self, user_ids, item_ids, genre, gamma, device, batch_size=128):
        self.eval()
        user_emb, item_emb = self._users, self._items
        all_ratings = []

        with torch.no_grad():
            for i in range(0, user_ids.shape[0], batch_size):
                batch_users = user_ids[i:i+batch_size]
                user_e = user_emb[batch_users]
                rating = torch.mm(user_e, item_emb.t()).cpu().numpy(force=True)
                for j, u in enumerate(batch_users):
                    rating[j, self.train_pos_list[u]] = -(1 << 10)
                    rating[j, self.train_neg_list[u]] = -(1 << 10)
                
                # Manually extend the ratings
                all_ratings += [
                    rating[k, idx_item] for k, idx_item in enumerate(item_ids[i:i+batch_size])
                ]
                
        return torch.tensor(all_ratings), []

    def train_func(self, dataset, epoch):
        self.train()
        pos_u = dataset.train_pos_user
        pos_i = dataset.train_pos_item
        indices = torch.randperm(dataset.train_neg_user.shape[0])
        neg_u = dataset.train_neg_user[indices]
        neg_i = dataset.train_neg_item[indices]
        all_j = structured_negative_sampling(
                torch.concat([torch.stack([pos_u, pos_i]), torch.stack([neg_u, neg_i])], dim=1),
                num_nodes=self.num_items)[2]
        pos_j, neg_j = torch.split(all_j, [pos_u.shape[0], neg_u.shape[0]])
        loss = self.loss_one_batch(pos_u, pos_i, pos_j, neg_u, neg_i, neg_j, dataset)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def loss_one_batch(self, pos_u, pos_i, pos_j, neg_u, neg_i, neg_j, dataset):
        self.computer(dataset)
        all_user, all_item = self._users, self._items
        pos_u_emb0, pos_u_emb = self.embedding_user(pos_u), all_user[pos_u]
        pos_i_emb0, pos_i_emb = self.embedding_item(pos_i), all_item[pos_i]
        pos_j_emb0, pos_j_emb = self.embedding_item(pos_j), all_item[pos_j]
        neg_u_emb0, neg_u_emb = self.embedding_user(neg_u), all_user[neg_u]
        neg_i_emb0, neg_i_emb = self.embedding_item(neg_i), all_item[neg_i]
        neg_j_emb0, neg_j_emb = self.embedding_item(neg_j), all_item[neg_j]
        pos_scores_ui = torch.sum(torch.mul(pos_u_emb, pos_i_emb), dim=-1)
        pos_scores_uj = torch.sum(torch.mul(pos_u_emb, pos_j_emb), dim=-1)
        neg_scores_ui = torch.sum(torch.mul(neg_u_emb, neg_i_emb), dim=-1)
        neg_scores_uj = torch.sum(torch.mul(neg_u_emb, neg_j_emb), dim=-1)
        if self.beta == 0:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2))/float(pos_u.shape[0])
            scores = pos_scores_uj-pos_scores_ui
        else:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2) +
                              neg_u_emb0.norm(2).pow(2) +
                              neg_i_emb0.norm(2).pow(2) +
                              neg_j_emb0.norm(2).pow(2))/float(pos_u.shape[0]+neg_u.shape[0])
            scores = torch.concat([pos_scores_uj-pos_scores_ui, self.beta*(neg_scores_uj-neg_scores_ui)], dim=0)
        loss = torch.mean(F.softplus(scores))
        return loss+self.lambda_reg*reg_loss

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model