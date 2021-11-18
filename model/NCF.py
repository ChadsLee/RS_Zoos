import math
import numpy as np
from operator import neg
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
	def __init__(self, n_user, n_item, norm_adj, args, model='Pre-NeuMF', GMF_model=None, MLP_model=None):
		super(NCF, self).__init__()
		# shared parameter
		self.n_user = n_user
		self.n_item = n_item
		self.device = args.device
		self.emb_size = args.embed_size
		self.batch_size = args.batch_size
		self.decay = eval(args.regs)[0]
		# NCF parameter
		self.dropout = args.dropout_rate # dropout rate
		self.num_layers = args.NCF_layers
		self.MLP_emb_size = self.emb_size * (2 ** (self.num_layers - 1))
		self.model = model # [NeuMF, GMF, MLP]
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model
		# model parameter
		self.embed_user_GMF = nn.Embedding(self.n_user, self.emb_size)
		self.embed_item_GMF = nn.Embedding(self.n_item, self.emb_size)
		self.embed_user_MLP = nn.Embedding(
				self.n_user, self.MLP_emb_size)
		self.embed_item_MLP = nn.Embedding(
				self.n_item, self.MLP_emb_size)

		MLP_modules = []
		for i in range(self.num_layers):
			input_size = self.emb_size * (2 ** (self.num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		# predict_size
		if self.model in ['MLP', 'GMF']:
			predict_size = self.emb_size 
		else:
			predict_size = self.emb_size * 2

		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		if self.model in ['GMF', 'MLP', 'NeuMF']:
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight,
									a=1, nonlinearity='sigmoid')
			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			#load pre-trained MLP, GMF for 'Pre-NeuMF'
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

		

	def _create_bce_loss(self, users, pos_items, neg_items, drop_flag=False):
		loss_function = nn.BCEWithLogitsLoss()
		pos_label = torch.ones(len(pos_items)).to(self.device)
		neg_label = torch.zeros(len(neg_items)).to(self.device)

		#list 2 tensor
		# neg_users = torch.Tensor([u for u in users for i in range(4)]).long().to(self.device)
		users = torch.Tensor(users).long().to(self.device)
		pos_items = torch.Tensor(pos_items).long().to(self.device)
		neg_items = torch.Tensor(neg_items).long().to(self.device)
		# user embeddings
		embed_user_GMF = self.embed_user_GMF(users)
		embed_user_MLP = self.embed_user_MLP(users)

		# pos_items embeddings
		embed_pos_item_GMF = self.embed_item_GMF(pos_items)
		embed_pos_item_MLP = self.embed_item_MLP(pos_items)
		# neg_items embeddings
		embed_neg_item_GMF = self.embed_item_GMF(neg_items)
		embed_neg_item_MLP = self.embed_item_MLP(neg_items)

		# positive items recommendation score
		pos_output_GMF = embed_user_GMF * embed_pos_item_GMF
		pos_interaction = torch.cat((embed_user_MLP, embed_pos_item_MLP), -1)
		pos_output_MLP = self.MLP_layers(pos_interaction)
		pos_concat = torch.cat((pos_output_GMF, pos_output_MLP), -1)
		pos_prediction = self.predict_layer(pos_concat)
		#flatten
		pos_scores = pos_prediction.view(-1)
		# negtive items recommendation score
		neg_output_GMF = embed_user_GMF * embed_neg_item_GMF
		neg_interaction = torch.cat((embed_user_MLP, embed_neg_item_MLP), -1)
		neg_output_MLP = self.MLP_layers(neg_interaction)
		neg_concat = torch.cat((neg_output_GMF, neg_output_MLP), -1)
		neg_prediction = self.predict_layer(neg_concat)
		# flatten
		neg_scores = neg_prediction.view(-1)

		pos_bce_loss = loss_function(pos_scores, pos_label)
		neg_bce_loss = loss_function(neg_scores, neg_label)
		bce_loss = pos_bce_loss + neg_bce_loss
		# regularizer loss
		regularizer = (torch.norm(embed_user_GMF) ** 2 + torch.norm(embed_user_MLP) ** 2
						+ torch.norm(embed_pos_item_GMF) ** 2 + torch.norm(embed_pos_item_MLP) ** 2
						+ torch.norm(embed_neg_item_GMF) ** 2 + torch.norm(embed_neg_item_MLP) ** 2) / 2
		emb_loss = self.decay * regularizer / self.batch_size

		return bce_loss+emb_loss, bce_loss, emb_loss
	
	def _create_bpr_loss(self, users, pos_items, neg_items, drop_flag=False):
		""" real bpr loss """
		#list 2 tensor
		users = torch.Tensor(users).long().to(self.device)
		pos_items = torch.Tensor(pos_items).long().to(self.device)
		neg_items = torch.Tensor(neg_items).long().to(self.device)
		# user embeddings
		embed_user_GMF = self.embed_user_GMF(users)
		embed_user_MLP = self.embed_user_MLP(users)
		# pos_items embeddings
		embed_pos_item_GMF = self.embed_item_GMF(pos_items)
		embed_pos_item_MLP = self.embed_item_MLP(pos_items)
		# neg_items embeddings
		embed_neg_item_GMF = self.embed_item_GMF(neg_items)
		embed_neg_item_MLP = self.embed_item_MLP(neg_items)

		# positive items recommendation score
		pos_output_GMF = embed_user_GMF * embed_pos_item_GMF
		pos_interaction = torch.cat((embed_user_MLP, embed_pos_item_MLP), -1)
		pos_output_MLP = self.MLP_layers(pos_interaction)
		pos_concat = torch.cat((pos_output_GMF, pos_output_MLP), -1)
		pos_prediction = self.predict_layer(pos_concat)
		pos_scores = pos_prediction.view(-1)

		# negtive items recommendation score
		neg_output_GMF = embed_user_GMF * embed_neg_item_GMF
		neg_interaction = torch.cat((embed_user_MLP, embed_neg_item_MLP), -1)
		neg_output_MLP = self.MLP_layers(neg_interaction)
		neg_concat = torch.cat((neg_output_GMF, neg_output_MLP), -1)
		neg_prediction = self.predict_layer(neg_concat)
		# 这里需要检查一下尺寸
		neg_scores = neg_prediction.view(-1)
		# bpr loss
		maxi = nn.LogSigmoid()(pos_scores - neg_scores)
		bpr_loss = -1 * torch.mean(maxi)

		# regularizer loss
		regularizer = (torch.norm(embed_user_GMF) ** 2 + torch.norm(embed_user_MLP) ** 2
						+ torch.norm(embed_pos_item_GMF) ** 2 + torch.norm(embed_pos_item_MLP) ** 2
						+ torch.norm(embed_neg_item_GMF) ** 2 + torch.norm(embed_neg_item_MLP) ** 2) / 2
		emb_loss = self.decay * regularizer / self.batch_size

		return bpr_loss + emb_loss,bpr_loss, emb_loss

	def forward(self, user, item, drop_flag = False):
		user = torch.Tensor(user).long().to(self.device)
		item = torch.Tensor(item).long().to(self.device)
		if not self.model == 'MLP':
			# GMF output
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			# 先把user_emb repeat n_items次，然后reshape，(batch_size, n_items, emb_size)
			output_GMF = (torch.reshape(embed_user_GMF.repeat(1,item.shape[0]), (user.shape[0],item.shape[0],self.emb_size))) * embed_item_GMF
			# output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			# MLP output
			embed_user_MLP = self.embed_user_MLP(user)
			embed_user_MLP = (torch.reshape(embed_user_MLP.repeat(1,item.shape[0]), (user.shape[0],item.shape[0],self.MLP_emb_size)))
			embed_item_MLP = self.embed_item_MLP(item)
			embed_item_MLP = embed_item_MLP.expand(user.shape[0], item.shape[0], self.MLP_emb_size)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return torch.squeeze(prediction)
	def get_rating(self, user, item):
		user = torch.Tensor(user).long().to(self.device)
		item = torch.Tensor(item).long().to(self.device)
		if not self.model == 'MLP':
			# GMF output
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			# MLP output
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return torch.squeeze(prediction)

	def create_bpr_loss(self, users, pos_items, neg_items, drop_flag=False):
		"""
		actually BCE loss, easy to write code
		"""
		loss_function = nn.BCEWithLogitsLoss()
		pos_label = torch.ones(len(pos_items)).to(self.device)
		neg_label = torch.zeros(len(neg_items)).to(self.device)
		# scores
		pos_scores = self.get_rating(users, pos_items)
		neg_scores = self.get_rating(users, neg_items)
		pos_bce_loss = loss_function(pos_scores, pos_label)
		neg_bce_loss = loss_function(neg_scores, neg_label)
		bce_loss = pos_bce_loss + neg_bce_loss

		return bce_loss, pos_bce_loss, neg_bce_loss

	