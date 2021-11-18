
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
import time


class LGNGuard(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(LGNGuard, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)

        self.__init_weight()
        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        
        self.beta_ = 0.5
        self.pruning_W = torch.nn.Linear(2, 1)
        self.pruning_W.reset_parameters()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.emb_size)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_item, embedding_dim=self.emb_size)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        # print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def attention_coef(self, embedding_k, adj, memory):
        """
        Use torch instead of numpy to calculate cosine similarity, 60% faster than the original paper
        """
        t = time.time()
        size = memory.shape
        # cosine similarity
        emb_copy = embedding_k.cpu().data.numpy()
        coef_matrix = torch.nn.functional.cosine_similarity(X=emb_copy, Y=emb_copy)
        row, col = adj.nonzero()

        coef = lil_matrix(size, dtype=np.float32)
        coef[row, col] = coef_matrix[row, col] # type - scipy.sparse.lil.lil_matrix
        # row, col = row.cpu().data, col.cpu().data
        # print(f"generating coef: \t\t{time.time()-t:.8f}")

        # normalization, make the sum of each row is 1 (LightGCN no self-loop!
        coef = normalize(coef, axis=1, norm='l1')  # --> scipy.sparse.csr.csr_matrix
        # print(f"normalize coef: \t\t{time.time()-t:.8f}")

        # print(f'normalized coef: {coef}')
        # print(f'===' * 10)
        # edge pruning (learnable parameter W)
        characteristic_v = np.vstack((coef[row, col].data, coef[col, row].data))
        characteristic_v = torch.from_numpy(characteristic_v.T).cuda()
        prune_pro = self.pruning_W(characteristic_v)
        prune_pro = torch.sigmoid(prune_pro)  # do not use softmax since we only have one element
        th_1 = torch.nn.Threshold(0.5, 0)
        prune_pro = th_1(prune_pro)
        th_2 = torch.nn.Threshold(-0.49, 1)
        prune_pro = th_2(-prune_pro)
        prune_decision = prune_pro.clone().requires_grad_()
        prune_matrix = lil_matrix(size, dtype=np.float32)
        prune_matrix[row, col] = prune_decision.cpu().data.numpy().squeeze(-1)
        # print(f'prune_matrix: {prune_matrix}')
        # print(f'===' * 10)
        coef = coef.multiply(prune_matrix.tocsr())
        # print(f'memory: {memory}')
        # print(f'==='*10)
        # print(f'coef: {coef}')
        print(f"edge pruning: \t\t{time.time()-t:.8f}")

        memory = self.beta_ * memory + (1 - self.beta_) * coef
        graph_ = memory.multiply(adj.tocsr())
        print(f"graph: \t\t{time.time()-t:.8f}")

        row, col = graph_.nonzero()
        ind_ = np.vstack((row, col))
        weight = graph_[row, col]
        # weight = np.exp(weight)   # exponent, kind of softmax
        weight = torch.tensor(np.array(weight)[0], dtype=torch.float32)  # .cuda()
        ind_ = torch.tensor(ind_, dtype=torch.int64)  # .cuda()

        new_adj = torch.sparse.FloatTensor(ind_, weight, size)
        print(f"generating torch.sparse graph: \t\t{time.time()-t:.8f}")
        print('==='*10)
        return new_adj.cuda(), memory
    
    def computer(self,drop_flag=False):
        """
        propagate methods for lightGCN
        """       
        A_hat = self.sparse_dropout(self.sparse_norm_adj,self.node_dropout,self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = A_hat

        adj = lil_matrix(g_droped.size(), dtype=np.float32)
        ind_ = g_droped.coalesce().indices().cpu().data
        val_ = g_droped.coalesce().values().cpu().data
        
        adj[ind_[0], ind_[1]] = val_
        memory = adj.tocsr()
        
        for layer in range(len(self.layers)):
            g_droped, memory = self.attention_coef(all_emb, adj, memory)
            
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_user, self.n_item])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(torch.tensor(users).to(self.device))
        pos_emb_ego = self.embedding_item(torch.tensor(pos_items).to(self.device))
        neg_emb_ego = self.embedding_item(torch.tensor(neg_items).to(self.device))
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def create_bpr_loss(self, users, pos, neg,drop_flag=False):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users, pos, neg)
        
#         print(userEmb0.shape, posEmb0.shape, negEmb0.shape, float(len(users)))
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        emb_loss = self.decay * reg_loss / self.batch_size
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss+emb_loss, loss, emb_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.matmul(users_emb, items_emb.t())
        gamma     = self.f(inner_pro)
        return gamma
