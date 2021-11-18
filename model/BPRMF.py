
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BPRMF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(BPRMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]

        self.adv_loss = nn.MarginRankingLoss(margin=self.kappa_)

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.emb_size)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_item, embedding_dim=self.emb_size)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)


    def create_bpr_loss(self, users, pos_items, neg_items, drop_flag = False):
        users_emb, pos_items_emb, neg_items_emb = self.get_embedding(users, pos_items, neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), axis=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_items_emb), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        bpr_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(users_emb) ** 2
                       + torch.norm(pos_items_emb) ** 2
                       + torch.norm(neg_items_emb) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return bpr_loss + emb_loss,bpr_loss, emb_loss

    # def rating(self, u_g_embeddings, pos_i_g_embeddings):
    #     return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def get_embedding(self, users, pos_items, neg_items):
        """
        return users,pos_items,neg_items' embedding
        """
        users = torch.Tensor(users).long().to(self.device)
        pos_items = torch.Tensor(pos_items).long().to(self.device)
        neg_items = torch.Tensor(neg_items).long().to(self.device)

        users_emb = self.embedding_user(users)
        pos_items_emb = self.embedding_item(pos_items)
        neg_items_emb = self.embedding_item(neg_items)
        
        return users_emb, pos_items_emb, neg_items_emb

    def forward(self, users, pos_items, drop_flag = False):
        """
        return recommendation socre for u,i
        """
        users = torch.Tensor(users).long().to(self.device)
        pos_items = torch.Tensor(pos_items).long().to(self.device)

        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos_items)

        score = torch.matmul(users_emb, pos_emb.t())

        return score