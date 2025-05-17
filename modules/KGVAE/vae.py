import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3 / 20, 3 / 4, 1 / 10]):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        # 加入 epsilon 以避免 norm 為 0
        norm = x.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
        x = x / norm[:, None]
        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    # feed['users'] = pairs[:, 0]
    # feed['pos_items'] = pairs[:, 1]
    # feed['neg_items'] = pairs[:, 2]
    # feed['batch_start'] = start
    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.2, calculate_loss=True):
        # print(user_ratings.shape)
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        # print(x_pred.shape)
        if calculate_loss:

            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo

        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


class RecVAETrainer:
    def __init__(
            self,
            recvae: VAE,
            user_item_matrix: torch.Tensor,
            lr: float,
            n_enc_epochs: int,
            n_dec_epochs: int,
            beta=None,
            gamma=1,
            device="cpu",
    ):
        """
        recvae:            你的 VAE 實例 (modules.KGVAE.kgvae.VAE)
        user_item_matrix:  [n_users, n_items] 的二元交互矩陣
        lr:                學習率
        n_enc_epochs:      每輪預訓練中，僅做 Encoder 階段的迭代次數
        n_dec_epochs:      每輪預訓練中，僅做 Decoder 階段的迭代次數
        beta, gamma:       RecVAE 的 KL 權重參數
        """
        # 將 RecVAE 和交互矩陣搬到指定裝置
        self.recvae = recvae.to(device)
        self.R = user_item_matrix.to(device)
        self.n_enc = n_enc_epochs
        self.n_dec = n_dec_epochs
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # 建立兩個 optimizer：一個只更新 encoder，另一個只更新 decoder
        self.opt_enc = torch.optim.Adam(self.recvae.encoder.parameters(), lr=lr)
        self.opt_dec = torch.optim.Adam(self.recvae.decoder.parameters(), lr=lr)

    def train_one_epoch(self, dropout_rate=0.5):
        # 1) Encoder-only (有 dropout)
        total_enc_loss = 0.0
        self.recvae.train()
        for _ in range(self.n_enc):
            self.opt_enc.zero_grad()
            # recvae_forward 會傳回 ( (mll, kld), loss )
            _, loss_enc = self.recvae(
                self.R,
                beta=self.beta,
                gamma=self.gamma,
                dropout_rate=dropout_rate,
                calculate_loss=True
            )
            loss_enc.backward()
            self.opt_enc.step()
            total_enc_loss += loss_enc.item()
        avg_enc = total_enc_loss / self.n_enc

        # 更新 CompositePrior 裡的 encoder_old
        self.recvae.update_prior()

        # 2) Decoder-only (no dropout)
        total_dec_loss = 0.0
        self.recvae.train()
        for _ in range(self.n_dec):
            self.opt_dec.zero_grad()
            _, loss_dec = self.recvae(
                self.R,
                beta=self.beta,
                gamma=self.gamma,
                dropout_rate=0.0,
                calculate_loss=True
            )
            loss_dec.backward()
            self.opt_dec.step()
            total_dec_loss += loss_dec.item()
        avg_dec = total_dec_loss / self.n_dec

        return avg_enc, avg_dec

    def pretrain(self, epochs, dropout_rate=0.5):
        assert epochs > 0, "Must pretrain for at least one epoch"

        avg_enc, avg_dec = 0.0, 0.0
        for epoch in range(epochs):
            avg_enc, avg_dec = self.train_one_epoch(dropout_rate=dropout_rate)
            print(f"[RecVAE 預訓練] Epoch {epoch + 1}/{epochs}  "
                  f"Encoder avg loss: {avg_enc:.4f}, Decoder avg loss: {avg_dec:.4f}")

        return avg_enc, avg_dec
