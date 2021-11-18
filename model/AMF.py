
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.modules.activation import LogSigmoid
from utility.helper import set_seed
set_seed(2021)

class AMF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(AMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.__init_weight()

        self.reg_adv = args.reg_adv
        self.eps = args.eps

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

    def create_apr_loss(self, users, pos_items, neg_items, drop_flag = False):

        users_emb, pos_items_emb, neg_items_emb = self.get_embedding(users, pos_items, neg_items)
        # optimizer.zero_grad()
        users_emb.retain_grad()
        u_clone = users_emb.data.clone()
        pos_items_emb.retain_grad()
        i_clone = pos_items_emb.data.clone()
        neg_items_emb.retain_grad()
        j_clone = neg_items_emb.data.clone()

        pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), axis=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_items_emb), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        bpr_loss = -1 * torch.mean(maxi)

        # Backward to get grads
        bpr_loss.backward(retain_graph=True)
        grad_user_emb = users_emb.grad
        grad_pos_emb = pos_items_emb.grad
        grad_neg_emb = neg_items_emb.grad

        # Construct adversarial perturbation
        delta_u = nn.functional.normalize(grad_user_emb, p=2, dim=1) * self.eps
        delta_i = nn.functional.normalize(grad_pos_emb, p=2, dim=1) * self.eps
        delta_j = nn.functional.normalize(grad_neg_emb, p=2, dim=1) * self.eps

        # Add adversarial perturbation to embeddings
        adv_pos_scores = torch.sum(torch.mul(users_emb + delta_u, pos_items_emb + delta_i), axis=1)
        adv_neg_scores = torch.sum(torch.mul(users_emb + delta_u, neg_items_emb + delta_j), axis=1)
        adv_maxi = nn.LogSigmoid()(adv_pos_scores - adv_neg_scores)
        apr_loss = -1 * torch.mean(adv_maxi)
        apr_loss = self.reg_adv*apr_loss

        regularizer = (torch.norm(users_emb) ** 2
                       + torch.norm(pos_items_emb) ** 2
                       + torch.norm(neg_items_emb) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        total_loss = apr_loss + emb_loss + bpr_loss
        # total_loss.backward()
        # optimizer.step()

        return total_loss, bpr_loss, apr_loss


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