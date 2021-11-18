"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class ALGN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(ALGN, self).__init__()
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

        self.reg_adv = args.reg_adv
        self.eps = args.eps
        
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
        
        for layer in range(len(self.layers)):
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
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        emb_loss = self.decay * reg_loss / self.batch_size
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss+emb_loss, loss, emb_loss

    def create_apr_loss(self, users, pos, neg, drop_flag=False):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,self.node_dropout,self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        users_emb.retain_grad()
        items_emb.retain_grad()
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = A_hat
        for layer in range(len(self.layers)):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        all_users, all_items = torch.split(light_out, [self.n_user, self.n_item])
        users_fea, pos_fea, neg_fea = all_users[users], all_items[pos], all_items[neg]
        pos_scores = torch.sum(torch.mul(users_fea, pos_fea), dim=1)
        neg_scores = torch.sum(torch.mul(users_fea, neg_fea), dim=1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # Backward to get grads
        bpr_loss.backward(retain_graph=True)
        grad_users_emb = users_emb.grad
        grad_items_emb = items_emb.grad
        # Construct adversarial perturbation
        delta_u = nn.functional.normalize(grad_users_emb, p=2, dim=1) * self.eps
        delta_i = nn.functional.normalize(grad_items_emb, p=2, dim=1) * self.eps
        adv_all_emb = torch.cat([users_emb + delta_u, items_emb + delta_i])
        adv_embs = [adv_all_emb]
        for layer in range(len(self.layers)):
            adv_all_emb = torch.sparse.mm(g_droped, adv_all_emb)
            adv_embs.append(adv_all_emb)
        adv_embs = torch.stack(adv_embs, dim=1)
        adv_light_out = torch.mean(adv_embs, dim=1)
        adv_all_users, adv_all_items = torch.split(adv_light_out, [self.n_user, self.n_item])
        adv_users_fea, adv_pos_fea, adv_neg_fea = adv_all_users[users], adv_all_items[pos], adv_all_items[neg]
        adv_pos_scores = torch.sum(torch.mul(adv_users_fea, adv_pos_fea), dim=1)
        adv_neg_scores = torch.sum(torch.mul(adv_users_fea, adv_neg_fea), dim=1)
        apr_loss = torch.mean(torch.nn.functional.softplus(adv_neg_scores - adv_pos_scores))
        apr_loss = self.reg_adv*apr_loss
        # regularizer
        reg_loss = (1/2)*(users_emb[users].norm(2).pow(2) + items_emb[pos].norm(2).pow(2) + items_emb[neg].norm(2).pow(2))/float(len(users))
        emb_loss = self.decay * reg_loss / self.batch_size

        total_loss = apr_loss + emb_loss + bpr_loss
        return total_loss, bpr_loss, apr_loss

       
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
