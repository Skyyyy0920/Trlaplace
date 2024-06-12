import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.nn.functional import normalize
from scipy.stats import bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Baseline
# ----------------------------------------------------------------------------------------------------------------------------#
# Local Metric DP
def ori_metric_dp(batch_embs, eps, embs_dim=300):
    # https://dl.acm.org/doi/pdf/10.1145/3336191.3371856   
    # Original Metric DP
    # MCMC Sampling
    v = MultivariateNormal(torch.zeros(embs_dim), torch.eye(embs_dim)).sample()
    v1 = normalize(v, p=2.0, dim=0)
    l = Gamma(torch.tensor([float(embs_dim)]), torch.tensor([float(eps)])).sample()
    z = l * v1
    embs = batch_embs + z.to(device)
    return embs


def mahalanobis(batch_embs, eps, embs_dim=300, lamb=1):
    # https://arxiv.org/pdf/2010.11947.pdf 
    # Mahalanobis Mechanism, lamb is a tuning parameter

    # MCMC Sampling
    v = MultivariateNormal(torch.zeros(embs_dim), torch.eye(embs_dim)).sample(
        [batch_embs.shape[0] * batch_embs.shape[1]]).to(device)
    v1 = normalize(v, p=2.0, dim=0)
    gamma = Gamma(torch.tensor([float(embs_dim)]), torch.tensor([float(eps)]))

    emb_cov = torch.reshape(batch_embs, (-1, batch_embs.shape[2]))
    cov = torch.mm(emb_cov.t(), emb_cov) / (emb_cov.shape[0])
    a = lamb * cov + (1 - lamb) * torch.eye(embs_dim).to(device)
    a1 = torch.mm(v1, a.t())

    l = gamma.sample([batch_embs.shape[0] * batch_embs.shape[1]]).to(device)
    z = l * a1

    z = torch.reshape(z, (batch_embs.shape[0], batch_embs.shape[1], batch_embs.shape[2])).to(device)
    batch_embs = batch_embs + z

    return batch_embs


def vickrey(batch_embs, embs):
    # https://arxiv.org/pdf/2104.11838.pdf 
    # Vickrey Mechanism

    # The noise is the same with Original Metric DP
    # Only difference is the post-processing of word embedding to word

    # In case our method is denoise, I think it is not neccessary to compare this mechanism

    return embs


def privemb(batch_embs, eps, embs_dim=300, beta=0.5, delta=0.1 ** 6):
    # https://aclanthology.org/2021.trustnlp-1.3.pdf
    # PRIVEMB, para and beta are tuning parameters
    para = 1
    m1 = np.sqrt(np.log(embs_dim)) + np.sqrt(np.log(1 / delta))
    m = int(para * np.power(m1, 2) / (beta * beta))

    s = torch.from_numpy(np.random.normal(0, 1 / m, m * embs_dim))
    fi = torch.reshape(s, (m, embs_dim))
    fi = fi.float().to(device)

    # sampling
    v = MultivariateNormal(torch.zeros(m), torch.eye(m)).sample()
    v1 = normalize(v, p=2.0, dim=0)
    l = Gamma(torch.tensor([float(m)]), torch.tensor([eps / (1 + beta)])).sample()
    z = l * v1

    w = torch.einsum('md, bld ->blm', fi, batch_embs)
    w = w + torch.Tensor(z).to(device)

    # map back
    embs = torch.einsum('dm, blm ->bld', fi.t(), w)

    return embs


# LDP
def Gaussian(batch_embs, eps, embs_dim=300, C=0.005):
    # calculate sensitivity
    # C is a tuning parameter
    C = 0.005
    d = 2 * C * np.sqrt(embs_dim)
    alpha = eps / d
    scale = 1 / alpha

    embs = batch_embs + torch.Tensor(np.random.normal(0, scale, embs_dim)).to(device)

    return embs


def Laplacian(batch_embs, eps, embs_dim=300, C=0.005):
    # calculate sensitivity
    # C is a tuning parameter
    d = 2 * C * np.sqrt(embs_dim)
    alpha = eps / d
    scale = 1 / alpha

    embs = batch_embs + torch.Tensor(np.random.laplace(0, scale, 300)).to(device)

    return embs


def TrLaplacian(batch_embs, eps, embs_dim=300, C=0.005):
    # embs_dim = 1700
    # embs_dim = 500
    ## MCMC sampling Trancated Laplacian
    # C is a tuning parameter
    # ε≤4∆1∆∞
    d1 = 2 * C * np.sqrt(embs_dim)
    dinf = 2 * C
    alpha = eps / d1
    scale = 1.0 / alpha
    A = -scale * np.log(1 - (2 * eps / np.sqrt(embs_dim)))
    mean = 0
    # MCMC
    r = []
    out = []
    lower = -A
    upper = A

    sgn = 2 * bernoulli.rvs(1, 0.5, 1000000) - 1
    r = mean - scale * sgn * np.log(np.random.uniform(0, 1, 1000000))
    mask1 = (r >= lower)
    mask2 = (r <= upper)
    mask = (mask1) & (mask2)
    r_tr = r[mask]
    out = r_tr[0:300]

    embs = batch_embs + torch.Tensor(out).to(device)

    return embs
