import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(MF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]
        self.neg_num = args.neg_ratio
        self._negative_weight = 0.9

        self.kappa_ = 1

        self.adv_loss = nn.MarginRankingLoss(margin=self.kappa_)

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

    def create_bce_loss(self, users, pos_items, neg_items, drop_flag = False):
        users_emb, pos_items_emb, neg_items_emb = self.get_embedding(users, pos_items, neg_items)
        pos_label = torch.ones(len(pos_items_emb)).to(self.device)
        neg_num = self.neg_num
        batch_size = users_emb.shape[0]
        dim = users_emb.shape[1]

        # pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), axis=1)
        # neg_scores = torch.sum(torch.mul(users_emb.view(batch_size,1,dim).repeat(1,neg_num,1).view(batch_size*neg_num,dim), neg_items_emb), dim=1)
        pos_scores = (users_emb * pos_items_emb).sum(dim=-1)
        users_emb = users_emb.unsqueeze(1)
        neg_scores = (users_emb * neg_items_emb).sum(dim=-1)
        # neg_label = torch.zeros(len(neg_scores)).to(self.device)
        neg_label = torch.zeros(neg_scores.size()).to(self.device)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_label, reduction="sum")
        
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_label, reduction="sum")
        loss = pos_loss+neg_loss
        
        # pos_loss = torch.relu(1 - pos_scores)
        # neg_loss = torch.relu(neg_scores - 0)
        # if self._negative_weight:
        #     loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        # else:
        #     loss = pos_loss + neg_loss.sum(dim=-1)

        regularizer = (torch.norm(users_emb) ** 2
                       + torch.norm(pos_items_emb) ** 2
                       + torch.norm(neg_items_emb) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return loss + emb_loss, loss, emb_loss

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