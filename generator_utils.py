import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def test_loss_function(recon_x, x, mu, logvar, variational_beta=.000005):
    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False
    BCE = reconstruction_function(recon_x, x)
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + variational_beta * KLD

def vae_loss(recon_x, x, mu, logvar, variational_beta=1):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # NOTE if ever get problems here, check that inputs are between 0 and 1
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')  
    # print(recon_loss)
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(kldivergence)
    return recon_loss + variational_beta * kldivergence


def mse_loss_function(recons, input, mu, log_var, variational_beta=0.00025):
    """ For Celeba
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    recons_loss = F.mse_loss(recons, input, reduction="mean")
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss = recons_loss + variational_beta * kld_loss
    return loss # {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

def bce_loss_function(recons, input, mu, log_var, variational_beta=1):
    # For SVHN
    recons = torch.clamp(recons, min=0, max=1)
    input = torch.clamp(input, min=0, max=1)
    CE = F.mse_loss(recons, input, reduction="sum")
    var_x = log_var.exp()
    log_var = torch.clip(log_var, min=-100, max=100)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var_x)  # log_var.exp()

    loss = CE + variational_beta * KLD
    return loss

def to_img(x):
    x = x.clamp(0, 1)
    return x

def save_image(img, path):
    img = to_img(img)
    npimg = img.numpy()
    np_transposed = np.transpose(npimg, (1, 2, 0))
    plt.imshow(np_transposed)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.savefig(path, dpi=400)


