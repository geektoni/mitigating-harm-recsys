
import torch
from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch.utils.data import DataLoader

from torch import optim

from tqdm import tqdm
    
class LightGConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self,x,edge_index):
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
    def update(self,inputs: Tensor) -> Tensor:
        return inputs

class SiReN(nn.Module):
    def __init__(self, train, num_users, num_items, offset=1, num_layers = 2, MLP_layers=2, dim = 64, reg=1e-4,
                 learning_rate=5e-3, batch_size=1024, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SiReN,self).__init__()

        self.M = num_users
        self.N = num_items
        self.num_layers = num_layers
        self.MLP_layers = MLP_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim
        self.batch_size = batch_size

        edge_user = torch.tensor(train[train[2]>=offset][0].values)
        edge_item = torch.tensor(train[train[2]>=offset][1].values)+self.M
        
        edge_ = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)
        self.data_p=Data(edge_index=edge_)
        
        # For the graph with positive edges (LightGCN)
        self.E = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E.data)
        
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LightGConv()) 

        
        # For the graph with negative edges
        self.E2 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E2.data)

        for _ in range(MLP_layers):
            self.mlps.append(nn.Linear(dim,dim,bias=True))
            nn.init.xavier_normal_(self.mlps[-1].weight.data)
        
        # Attntion model
        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)

        self.data_p.to(self.device)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
        
    def aggregate(self):
        # Generate embeddings z_p
        B=[]; B.append(self.E)
        x = self.convs[0](self.E,self.data_p.edge_index)

        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x,self.data_p.edge_index)

            B.append(x)
        z_p = sum(B)/len(B) 

        # Generate embeddings z_n
        C = []; C.append(self.E2)
        x = F.dropout(F.relu(self.mlps[0](self.E2)),p=0.5,training=self.training)
        for i in range(1,self.MLP_layers):
            x = self.mlps[i](x);
            x = F.relu(x)
            x = F.dropout(x,p=0.5,training=self.training)
            C.append(x)
        z_n = C[-1]
        
        # Attntion for final embeddings Z
        w_p = self.q(F.dropout(torch.tanh((self.attn(z_p))),p=0.5,training=self.training))
        w_n = self.q(F.dropout(torch.tanh((self.attn(z_n))),p=0.5,training=self.training))
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))
        Z = alpha_[:,0].view(len(z_p),1) * z_p + alpha_[:,1].view(len(z_p),1) * z_n
        return Z
    
    def forward(self,u,v,w,n,device):
        emb = self.aggregate().to(self.device)
        u_ = emb[u]
        v_ = emb[v]
        n_ = emb[n]
        w_ = w.to(self.device)
        positivebatch = torch.mul(u_ , v_ ); 
        negativebatch = torch.mul(u_.view(len(u_),1,self.embed_dim),n_)  
        sBPR_loss =  F.logsigmoid((((-1/2*torch.sign(w_)+3/2)).view(len(u_),1) * (positivebatch.sum(dim=1).view(len(u_),1))) - negativebatch.sum(dim=2)).sum(dim=1) # weight
        reg_loss = u_.norm(dim=1).pow(2).sum() + v_.norm(dim=1).pow(2).sum() + n_.norm(dim=2).pow(2).sum() 
        return -torch.sum(sBPR_loss) + self.reg * reg_loss
    
    #def predict(self, test_user_ids, batch_size=None):
    def predict(self, user_ids, item_ids, genre, gamma, device, batch_size=128):
        self.eval()
        emb = self.aggregate()
        emb_u, emb_v = torch.split(emb,[self.M, self.N])
        emb_u = emb_u.cpu().detach(); emb_v = emb_v.cpu().detach()
        r_hat = emb_u.mm(emb_v.t()).numpy(force=True)

        return torch.tensor([r_hat[user_id, item_id] for user_id, item_id in zip(user_ids, item_ids)]), []
        #return r_hat.numpy(force=True) # Return full users
        #return r_hat[test_user_ids, :]

    def train_func(self, dataset, epoch):
    
        self.train()

        if epoch%20-1==0:
            dataset.negs_gen_EP(20)
            
        training_loss = 0
        dataset.edge_4 = dataset.edge_4_tot[:,:,epoch%20-1]
        
        ds = DataLoader(dataset,
                        batch_size=self.batch_size,
                        shuffle=True)
        q=0
        
        for u,v,w,negs in tqdm(ds):   
            q+=len(u)
            self.optimizer.zero_grad()
            loss = self(u,v,w,negs,self.device) # original
            loss.backward()                
            self.optimizer.step()
            training_loss += loss.item() * 1/len(ds)

        return training_loss          
