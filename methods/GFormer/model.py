import torch
import torch as t
from torch import nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GFormer(nn.Module):
    def __init__(self, gtLayer, user, items, dataset,
                 emb_dim=32, gn_layer=2, pnn_layer=1,
                 gtw=0.1, lr=1e-3, device="cpu"):
        super(GFormer, self).__init__()

        self.num_users = user
        self.num_items = items
        self.emb_dim = emb_dim
        self.gn_layer = gn_layer
        self.pnn_layer = pnn_layer
        self.gtw = gtw

        self.uEmbeds = nn.Parameter(init(t.empty(self.num_users, self.emb_dim)))
        self.iEmbeds = nn.Parameter(init(t.empty(self.num_items, self.emb_dim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gn_layer)])
        self.gcnLayer = GCNLayer()
        self.gtLayers = gtLayer
        self.pnnLayers = nn.Sequential(*[PNNLayer(device=device) for i in range(self.pnn_layer)])

        self.device = device

        self.opt = t.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(self.num_users, self.num_items, device=self.device)
        self.sampler = LocalGraph(gtLayer, self.num_users, self.num_items, device=self.device)

        self.dataset = dataset

    def getEgoEmbeds(self):
        return t.cat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, handler, is_test, sub, cmp, encoderAdj, decoderAdj=None):
        embeds = t.cat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        emb, _ = self.gtLayers(cmp, embeds)
        cList = [embeds, self.gtw * emb]
        emb, _ = self.gtLayers(sub, embeds)
        subList = [embeds, self.gtw * emb]

        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoderAdj, embedsLst[-1])
            embeds2 = gcn(sub, embedsLst[-1])
            embeds3 = gcn(cmp, embedsLst[-1])
            subList.append(embeds2)
            embedsLst.append(embeds)
            cList.append(embeds3)
        if is_test is False:
            for i, pnn in enumerate(self.pnnLayers):
                embeds = pnn(handler, embedsLst[-1])
                embedsLst.append(embeds)
        if decoderAdj is not None:
            embeds, _ = self.gtLayers(decoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        cList = sum(cList)
        subList = sum(subList)

        return embeds[:self.num_users], embeds[self.num_users:], cList, subList

    def train_func(self, dataset, epoch):

        def innerProduct(usrEmbeds, itmEmbeds):
            return t.sum(usrEmbeds * itmEmbeds, dim=-1)

        def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
            return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

        # Improved: Always use batched processing to compute contrast scores.
        def contrast(nodes, allEmbeds, allEmbeds2=None, batch_size=256):
            if allEmbeds2 is not None:
                pckEmbeds = allEmbeds[nodes]
                scores_list = []
                for i in range(0, pckEmbeds.size(0), batch_size):
                    batch = pckEmbeds[i:i+batch_size]
                    dot = batch @ allEmbeds2.T
                    scores_list.append(t.logsumexp(dot, dim=-1))
                scores = t.cat(scores_list, dim=0).mean()
            else:
                uniqNodes = t.unique(nodes)
                pckEmbeds = allEmbeds[uniqNodes]
                # Batch the dot-product here as well:
                scores_list = []
                for i in range(0, pckEmbeds.size(0), batch_size):
                    batch = pckEmbeds[i:i+batch_size]
                    dot = batch @ allEmbeds.T
                    scores_list.append(t.logsumexp(dot, dim=-1))
                scores = t.cat(scores_list, dim=0).mean()
            return scores

        def contrastNCE(nodes, allEmbeds, allEmbeds2=None, batch_size=256):
            if allEmbeds2 is not None:
                pckEmbeds = allEmbeds[nodes]
                pckEmbeds2 = allEmbeds2[nodes]
                scores_list = []
                for i in range(0, pckEmbeds.size(0), batch_size):
                    batch1 = pckEmbeds[i:i+batch_size]
                    batch2 = pckEmbeds2[i:i+batch_size]
                    scores_list.append(t.logsumexp(batch1 * batch2, dim=-1))
                scores = t.cat(scores_list, dim=0).mean()
            else:
                scores = None
            return scores

        def calcRegLoss(model):
            ret = 0
            for W in model.parameters():
                ret += W.norm(2).square()
            return ret

        trnLoader = dataset.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // dataset.batch_size
        dataset.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % 100 == 0:
                att_edge, add_adj = self.sampler(dataset.torchBiAdj, self.getEgoEmbeds(), dataset)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs = ancs.long().to(self.device)
            poss = poss.long().to(self.device)
            negs = negs.long().to(self.device)

            usrEmbeds, itmEmbeds, cList, subLst = self(dataset, False, sub, cmp, encoderAdj, decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            usrEmbeds2 = subLst[:dataset.num_users]
            itmEmbeds2 = subLst[dataset.num_users:]
            ancEmbeds2 = usrEmbeds2[ancs]
            posEmbeds2 = itmEmbeds2[poss]

            bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
            bprLoss2 = - (scoreDiff.sigmoid().log().sum()) / dataset.batch_size

            regLoss = calcRegLoss(self) * 1e-4
            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) + \
                           contrast(ancs, usrEmbeds, itmEmbeds) + 0.001 * contrastNCE(ancs, subLst, cList)
            loss = bprLoss + regLoss + contrastLoss + bprLoss2

            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            #print(f"Step {i}/{steps}:", loss.item())
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret['Loss']
    
    def predict(self, user_ids, item_ids, genre, gamma, device, batch_size=128):
        self.eval()
        all_ratings = []
        usrEmbeds, itmEmbeds, _, _ = self(self.dataset, True, self.dataset.torchBiAdj, self.dataset.torchBiAdj,
                                                          self.dataset.torchBiAdj)
        with torch.no_grad():
            for i in range(0, user_ids.shape[0], batch_size):
                batch_users = user_ids[i:i+batch_size]
                user_e = usrEmbeds[batch_users]
                rating = torch.mm(user_e,  t.transpose(itmEmbeds, 1, 0)).cpu().numpy(force=True)

                # Manually extend the ratings
                all_ratings += [
                    rating[k, idx_item] for k, idx_item in enumerate(item_ids[i:i+batch_size])
                ]
        
        return torch.tensor(all_ratings), []
    
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class PNNLayer(nn.Module):
    def __init__(self, emb_dim=32, anchor_set_num=32, device="cpu"):
        super(PNNLayer, self).__init__()
        self.emb_dim = emb_dim
        self.anchor_set_num = anchor_set_num

        self.linear_out_position = nn.Linear(self.emb_dim, 1)
        self.linear_out = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_hidden = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.act = nn.ReLU()
        self.device = device

    def forward(self, handler, embeds):
        anchor_set_id = handler.anchorset_id  # Expected to be a list/tensor of length `anchor_set_num`
        # Convert dists_array to tensor on device.
        dists_array = t.tensor(handler.dists_array, dtype=t.float32, device=self.device)
        # Ensure dists_array is in shape (num_nodes, anchor_set_num).
        if dists_array.shape[0] != embeds.shape[0]:
            dists_array = dists_array.T

        # Get anchor embeddings; shape: (anchor_set_num, emb_dim)
        set_ids_emb = embeds[anchor_set_id]

        # Expand anchors to all nodes: shape becomes (num_nodes, anchor_set_num, emb_dim)
        num_nodes = embeds.shape[0]
        set_ids_expanded = set_ids_emb.unsqueeze(0).expand(num_nodes, -1, -1)
        
        # Reshape dists_array to (num_nodes, anchor_set_num, 1)
        dists_array_emb = dists_array.unsqueeze(2)
        
        # Element-wise multiplication: result shape (num_nodes, anchor_set_num, emb_dim)
        messages = set_ids_expanded * dists_array_emb

        # Expand each node's embedding to match the anchor dimension:
        # Result shape: (num_nodes, anchor_set_num, emb_dim)
        self_feature = embeds.unsqueeze(1).expand(-1, self.anchor_set_num, -1)
        
        # Concatenate along the last dimension: shape becomes (num_nodes, anchor_set_num, 2 * emb_dim)
        messages = t.cat((messages, self_feature), dim=-1)
        
        # Process through a hidden layer, then aggregate over the anchors.
        messages = self.linear_hidden(messages)
        outposition1 = messages.mean(dim=1)
        return outposition1


class GTLayer(nn.Module):
    def __init__(self, emb_dim=32, head=4, device="cpu"):
        super(GTLayer, self).__init__()

        self.emb_dim = emb_dim
        self.head = head

        self.qTrans = nn.Parameter(init(t.empty(self.emb_dim, self.emb_dim)))
        self.kTrans = nn.Parameter(init(t.empty(self.emb_dim, self.emb_dim)))
        self.vTrans = nn.Parameter(init(t.empty(self.emb_dim, self.emb_dim)))

        self.device = device

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).to(self.device)
        noise = -t.log(-t.log(noise))
        return scores + 0.01 * noise

    def forward(self, adj, embeds, flag=False):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head, self.emb_dim // self.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head, self.emb_dim // self.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head, self.emb_dim // self.head])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], self.head]).to(self.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.emb_dim])
        tem = t.zeros([adj.shape[0], self.emb_dim]).to(self.device)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)
        return resEmbeds, att

class LocalGraph(nn.Module):

    def __init__(self, gtLayer, user, item, add_rate=0.01, device="cpu"):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer
        self.sft = t.nn.Softmax(0)
        self.device = device
        self.num_users = user
        self.num_items = item
        self.add_rate = add_rate
        self.pnn = PNNLayer(device=device).to(self.device)

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).to(self.device)
        noise = -t.log(-t.log(noise))
        return scores + noise

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        return dists_dict

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)
        num_workers = 1  # windows
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)
        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def precompute_dist_data(self, edge_index, num_nodes, approximate=0):
        graph = nx.Graph()
        graph.add_edges_from(edge_index)
        n = num_nodes
        dists_dict = self.all_pairs_shortest_path_length_parallel(graph,
                                                                  cutoff=approximate if approximate > 0 else None)
        dists_array = np.zeros((n, n), dtype=np.int8)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array

    def forward(self, adj, embeds, handler):
        # Run PNN on CPU or GPU as appropriate; here we run on the same device.
        embeds = self.pnn(handler, embeds)
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        # Instead of converting tensors repeatedly, perform CPU work then convert once.
        rows_cpu = rows.cpu().numpy()
        cols_cpu = cols.cpu().numpy()

        tmp_rows = np.random.choice(rows_cpu, size=int(len(rows_cpu) * self.add_rate))
        tmp_cols = np.random.choice(cols_cpu, size=int(len(cols_cpu) * self.add_rate))

        add_cols = t.tensor(tmp_cols, device=self.device)
        add_rows = t.tensor(tmp_rows, device=self.device)

        newRows = t.cat([add_rows, add_cols, t.arange(self.num_users + self.num_items, device=self.device), rows])
        newCols = t.cat([add_cols, add_rows, t.arange(self.num_users + self.num_items, device=self.device), cols])

        ratings_keep = np.ones(newRows.shape[0], dtype=np.float32)
        # Build sparse matrix on CPU and then convert.
        adj_mat = sp.csr_matrix((ratings_keep, (newRows.cpu().numpy(), newCols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))
        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        embeds_l2, atten = self.gt_layer(add_adj, embeds)
        att_edge = t.sum(atten, dim=-1)

        return att_edge, add_adj

class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items, sub=0.1, keep_rate=0.9, ext=0.5, re_rate=0.8, device="cpu"):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.sub = sub
        self.keep_rate = keep_rate
        self.ext = ext
        self.re_rate = re_rate
        self.device = device
        self.sft = t.nn.Softmax(1)

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def create_sub_adj(self, adj, att_edge, flag):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        if flag:
            att_edge = att_edge.detach().cpu().numpy() + 0.001
        else:
            att_f = att_edge
            att_f = t.clamp(att_f, max=3)
            att_edge = 1.0 / (np.exp(att_f.detach().cpu().numpy() + 1e-8))
        att_f = att_edge / att_edge.sum()
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.sub),
                                      replace=False, p=att_f)

        keep_index.sort()

        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(self.num_users + self.num_items, device=self.device), rows])
        cols = t.cat([t.arange(self.num_users + self.num_items, device=self.device), cols])

        ratings_keep = np.ones(rows.shape[0], dtype=np.float32)
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu().numpy(), cols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = t.clamp(att_edge, max=3)
        att_f = 1.0 / (np.exp(att_f.detach().cpu().numpy() + 1e-8))
        att_f1 = att_f / att_f.sum()

        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.keep_rate),
                                          replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(self.num_users + self.num_items, device=self.device), rows])
        cols = t.cat([t.arange(self.num_users + self.num_items, device=self.device), cols])
        ratings_keep = np.ones(rows.shape[0], dtype=np.float32)
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu().numpy(), cols.cpu().numpy())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        # Additional operations for decoderAdj are handled similarly on CPU then moved.
        drop_row_ids = users_up[~np.in1d(np.arange(len(users_up.cpu())), keep_index)]
        drop_col_ids = items_up[~np.in1d(np.arange(len(items_up.cpu())), keep_index)]

        ext_rows = np.random.choice(rows.cpu().numpy(), size=int(len(drop_row_ids) * self.ext))
        ext_cols = np.random.choice(cols.cpu().numpy(), size=int(len(drop_col_ids) * self.ext))

        ext_cols = t.tensor(ext_cols, device=self.device)
        ext_rows = t.tensor(ext_rows, device=self.device)

        tmp_rows = t.cat([ext_rows, drop_row_ids])
        tmp_cols = t.cat([ext_cols, drop_col_ids])

        new_rows = np.random.choice(tmp_rows.cpu().numpy(), size=int(adj._values().shape[0] * self.re_rate))
        new_cols = np.random.choice(tmp_cols.cpu().numpy(), size=int(adj._values().shape[0] * self.re_rate))

        new_rows = t.tensor(new_rows, device=self.device)
        new_cols = t.tensor(new_cols, device=self.device)

        newRows = t.cat([new_rows, new_cols, t.arange(self.num_users + self.num_items, device=self.device), rows])
        newCols = t.cat([new_cols, new_rows, t.arange(self.num_users + self.num_items, device=self.device), cols])

        hashVal = newRows * (self.num_users + self.num_items) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (self.num_users + self.num_items)
        newRows = ((hashVal - newCols) / (self.num_users + self.num_items)).long()

        decoderAdj = t.sparse_coo_tensor(
            t.stack([newRows, newCols], dim=0), t.ones_like(newRows, dtype=t.float32, device=self.device),
            adj.shape
        )

        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp
