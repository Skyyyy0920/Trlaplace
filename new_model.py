#! -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import transformers

from torchtext.vocab import GloVe, FastText
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.nn.functional import normalize

from scipy.stats import bernoulli


class GaussianDP(nn.Module):

    def __init__(self, eps, embs_dim=300, C=0.005, device='cuda:0'):
        super().__init__()
        # self.eps = eps
        self.embs_dim = embs_dim
        # self.C = C
        d = 2 * C * np.sqrt(embs_dim)
        alpha = eps / d
        self.scale = 1 / alpha
        self.device = device

    def forward(self, embs):
        embs = embs + torch.Tensor(np.random.normal(0, self.scale, self.embs_dim)).to(self.device)
        return embs


class LaplacianDP(nn.Module):

    def __init__(self, eps, embs_dim=300, C=0.005, device='cuda:0'):
        super().__init__()
        # self.eps = eps
        self.embs_dim = embs_dim
        # self.C = C
        d = 2 * C * np.sqrt(embs_dim)
        alpha = eps / d
        self.scale = 1 / alpha
        self.device = device

    def forward(self, embs):
        embs = embs + torch.Tensor(np.random.laplace(0, self.scale, self.embs_dim)).to(self.device)

        return embs


class TrLaplacianDP(nn.Module):

    def __init__(self, eps, embs_dim=300, C=0.005, device='cuda:0'):
        super().__init__()
        self.embs_dim = embs_dim
        d = 2 * C * np.sqrt(embs_dim)
        alpha = eps / d
        self.scale = 1 / alpha
        self.A = -self.scale * np.log(1 - (2 * eps / np.sqrt(embs_dim)))
        self.mean = 0
        self.sgn = 2 * bernoulli.rvs(1, 0.5, 1000000) - 1
        self.device = device

    def forward(self, embs):
        r = self.mean - self.scale * self.sgn * np.log(np.random.uniform(0, 1, 1000000))
        mask = (r >= -self.A) & (r <= self.A)
        r_tr = r[mask]
        out = r_tr[:300]
        embs = embs + torch.Tensor(out).to(self.device)

        return embs


class MDP(nn.Module):

    def __init__(self, eps, embs_dim=300, device='cuda:0'):
        super().__init__()
        self.embs_dim = embs_dim
        self.eps = eps
        self.device = device

    def forward(self, embs):
        v = MultivariateNormal(torch.zeros(self.embs_dim), torch.eye(self.embs_dim)).sample()
        v1 = normalize(v, p=2.0, dim=0)
        l = Gamma(torch.tensor([float(self.embs_dim)]), torch.tensor([self.eps])).sample()
        z = l * v1
        embs = embs + z.to(self.device)

        return embs


class MahaDP(nn.Module):

    def __init__(self, eps, embs_dim=300, lamb=0.5, device='cuda:0'):
        super().__init__()
        self.embs_dim = embs_dim
        self.lamb = lamb
        self.eps = eps
        self.device = device

    def forward(self, embs):
        v = MultivariateNormal(torch.zeros(self.embs_dim), torch.eye(self.embs_dim)).sample().to(self.device)
        v1 = normalize(v, p=2.0, dim=0)
        gamma = Gamma(torch.tensor([float(self.embs_dim)]), torch.tensor([self.eps]))
        # batch_cov = torch.zeros((embs.shape[0], embs.shape[2], embs.shape[2])).to(self.device)

        # for i in range(embs.shape[0]):
        #     batch_cov[i] = torch.cov(embs[i].t())
        embs_cov = embs.reshape(-1, self.embs_dim)
        embs_cov = torch.mm(embs_cov.t(), embs_cov) / embs_cov.shape[0]

        a = self.lamb * embs_cov + (1 - self.lamb) * torch.eye(self.embs_dim).to(self.device)
        a1 = torch.mm(v1, a.t())

        l = gamma.sample([embs.shape[0] * embs.shape[1]]).to(self.device)
        z = l * a1
        z = z.reshape(embs.shape[0], embs.shape[1], embs.shape[2]).to(self.device)
        embs = embs + z

        return embs


class PMBDP(nn.Module):

    def __init__(self, eps, embs_dim=300, beta=0.5, delta=1e-6, device='cuda:0'):
        super().__init__()
        self.para = 1
        m1 = np.sqrt(np.log(embs_dim)) + np.sqrt(np.log(1 / delta))
        self.m = int(self.para * np.power(m1, 2) / (beta * beta))
        self.embs_dim = embs_dim
        self.beta = beta
        self.eps = eps
        self.delta = delta
        self.device = device

    def forward(self, embs):
        s = torch.from_numpy(np.random.normal(0, 1 / self.m, self.m * self.embs_dim))
        fi = torch.reshape(s, (self.m, self.embs_dim))
        fi = fi.float().to(self.device)

        v = MultivariateNormal(torch.zeros(self.m), torch.eye(self.m)).sample()
        v1 = normalize(v, p=2.0, dim=0)
        l = Gamma(torch.tensor([float(self.m)]), torch.tensor([self.eps / (1 + self.beta)])).sample()
        z = l * v1
        w = torch.einsum('md, bld ->blm', fi, embs)
        w = w + torch.Tensor(z).to(self.device)
        embs = torch.einsum('dm, blm ->bld', fi.t(), w)

        return embs


class BiLSTM(nn.Module):

    def __init__(self, output_dim, input_size=300, hidden_dim=512, device='cuda:0', eps=1, emb_type='glove',
                 noise=None):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1, batch_first=True,
                                                        dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=300, hidden_size=hidden_dim)
        # self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.fc = nn.Linear(input_size, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        if emb_type == 'glove':
            self.myvec = GloVe(name='840B', dim=300).vectors.to(device)
        elif emb_type == 'fast':
            self.myvec = FastText(language='en').vectors.to(device)
        else:
            raise 'The word embedding type does not exist!'

        self.noise = True

        if noise == 'gaus':
            self.noise_layer = GaussianDP(eps=eps, embs_dim=300, C=0.005, device=device)
        elif noise == 'lap':
            self.noise_layer = LaplacianDP(eps=eps, embs_dim=300, C=0.005, device=device)
        elif noise == 'trlap':
            self.noise_layer = TrLaplacianDP(eps=eps, embs_dim=300, C=0.005, device=device)
        elif noise == 'mdp':
            self.noise_layer = MDP(eps=eps, embs_dim=300, device=device)
        elif noise == 'pmbdp':
            self.noise_layer = PMBDP(eps=eps, embs_dim=300, beta=0.5, delta=1e-6, device=device)
        elif noise == 'mahadp':
            self.noise_layer = MahaDP(eps=eps, embs_dim=300, lamb=0.5, device=device)
        else:
            self.noise = False
            self.noise_layer = None

        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight)

        for weight in self.lstm._all_weights:
            if "weight" in weight:
                init.kaiming_uniform_(getattr(self.lstm, weight))
            if "bias" in weight:
                init.kaiming_uniform_(getattr(self.lstm, weight))

    def forward(self, embs):
        embs = self.myvec[embs]
        if self.noise:
            embs = self.noise_layer(embs)
        # hidden, h_n = self.lstm(embs)
        hidden = self.transformer_encoder(embs)
        hidden = hidden[:, -1, :]
        hidden = F.relu(hidden)
        hidden = self.fc(hidden)
        hidden = F.relu(hidden)
        output = self.fc_2(hidden)
        return output


class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, embs):
        # ids = [batch size, seq len]
        embs = embs.squeeze(dim=1)
        output = self.transformer(inputs_embeds=embs, output_attentions=True)
        # output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction


class Transformer_id(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, embs):
        # ids = [batch size, seq len]
        embs = embs.squeeze(dim=1)
        output = self.transformer(inputs_embeds=embs, output_attentions=True)
        # output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction
