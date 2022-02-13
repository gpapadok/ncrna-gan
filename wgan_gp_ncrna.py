#!/usr/bin/env python
# coding: utf-8

from math import ceil
import os, glob, json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import torch
import torch.optim as optim
from torch import autograd

from tools.seq_gen import create_fake_fasta
from tools.models import Generator, Discriminator
from tools.utils import pad_sequence, encode_sequence


def get_sequences(tsv="./data/trainset.csv", length=200):
    ncrna_labels = [
        'mature',
        'pre-miRNA',
        'tRNA',
        'rRNA',
        'snoRNA',
        'trfs',
        'pseudo-hairpins'
    ]

    ncrna = pd.read_csv(tsv)

    labels = ncrna[ncrna_labels].values
    seqs = ncrna['Sequence'].apply(
        lambda seq: pad_sequence(encode_sequence(seq), length)
        ).values
    seqs = np.stack(seqs, axis=0)
    seqs = np.transpose(seqs, (0,2,1))

    return seqs, labels


class WGAN_GP():
    # WGAN-GP model

    def __init__(self, hidden_channels=100, batch_size=64, lr=1e-5,
                 version='test', latent_size=128, gamma=10, disc_iter=10,
                 beta1=.5, beta2=.9, residual=.3, tau=1.):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        self.batch_size = batch_size
        self.epochs_trained = 0
        self.lr = lr
        self.version = version
        self.latent_size = latent_size
        self.gamma = gamma
        self.disc_iter = disc_iter

        self._initialize_model(hidden_channels, beta1, beta2, residual, tau)

        os.makedirs(f"runs/{self.version}/samples", exist_ok=True)
        os.makedirs(f"runs/{self.version}/weights", exist_ok=True)

        return

    def _initialize_model(self, hidden_channels, beta1, beta2, residual, tau):
        self.discriminator = Discriminator(n_chars=5,
                                           seq_len=200,
                                           hidden_channels=hidden_channels,
                                           residual=residual
                                           ).to(self.device)
        self.generator = Generator(n_chars=5,
                                   seq_len=200,
                                   hidden_channels=hidden_channels,
                                   residual=residual,
                                   latent_size=self.latent_size,
                                   tau=tau
                                   ).to(self.device)

        self.optimD = optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr,
                                 betas=(beta1,beta2))
        self.optimG = optim.Adam(self.generator.parameters(),
                                 lr=self.lr,
                                 betas=(beta1,beta2))

        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimD,
                                                     step_size=50,
                                                     gamma=.5,
                                                     )

        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimG,
                                                     step_size=50,
                                                     gamma=.5,
                                                     )

        return

    def _get_next_batch(self, data, j):
        # batch generator
        return data[j*self.batch_size:(j+1)*self.batch_size]

    def save_samples(self, labels):
        self.generator.eval()
        fasta_path = f"runs/{self.version}/samples/{self.epochs_trained}.fa"
        idx = np.random.randint(len(labels), size=2000)
        create_fake_fasta(self.generator, fasta_path, labels[idx])
        self.generator.train()
        return

    def train(self, trainset, labels, epochs):
        # train function
        self._load_model()
        n_samples = trainset.size(0)
        n_iter = ceil(n_samples / self.batch_size)

        errG_history = []
        errD_real_history = []
        errD_fake_history = []

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            for j in range(n_iter):
                # TRAIN DISCRIMINATOR
                self.optimD.zero_grad()

                # Train with real
                real = self._get_next_batch(trainset, j).to(self.device)
                real_labels = self._get_next_batch(labels, j)
                current_batch_size = real.size()[0]
                out_real = self.discriminator(real, real_labels)
                errD_real = - out_real.mean()

                # Train with fake
                noise = torch.randn(current_batch_size, 1, self.latent_size,
                                    device=self.device)
                fake = self.generator(noise, real_labels, hard=True)
                out_fake = self.discriminator(fake, real_labels)
                errD_fake = out_fake.mean()

                # Gradient penalty
                epsilon = torch.rand(current_batch_size, 1, 1,
                                     device=self.device)
                epsilon = epsilon.expand_as(real)
                interpolations = epsilon * real + (1 - epsilon) * fake
                out_grad = self.discriminator(interpolations, real_labels)
                grad_outputs = torch.ones_like(out_grad, device=self.device)
                gradients = autograd.grad(outputs=out_grad,
                                          inputs=interpolations,
                                          grad_outputs=grad_outputs,
                                          create_graph=True,
                                          retain_graph=True)[0]
                grad_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + 1e-12
                    )
                grad_penalty = self.gamma * ((grad_norm - 1) ** 2).mean()

                # backpropagate loss
                loss = errD_real + errD_fake + grad_penalty
                loss.backward()

                self.optimD.step()

                errD_real_history += [errD_real.item()]
                errD_fake_history += [errD_fake.item()]

                # TRAIN GENERATOR
                if j % self.disc_iter == 0:
                    self.optimG.zero_grad()

                    noise = torch.randn(current_batch_size, 1, self.latent_size,
                                        device=self.device)
                    fake = self.generator(noise, real_labels, hard=True)
                    out_gen = self.discriminator(fake, real_labels)
                    errG = - out_gen.mean()
                    errG.backward()

                    self.optimG.step()

                    errG_history += [errG.item()]

            epoch_real_d = np.mean(errD_real_history[-n_iter:])
            epoch_fake_d = np.mean(errD_fake_history[-n_iter:])
            epoch_gen_d = np.mean(errG_history[-n_iter:])
            
            self.epochs_trained += 1
            print(f'EPOCH {self.epochs_trained:2}', end=" | ")
            print(f'disc_real_distance: {epoch_real_d:.4f}', end=' - ')
            print(f'disc_fake_distance: {epoch_fake_d:.4f}', end=' - ')
            print(f'gen_distance: {epoch_gen_d:.4f}')

            # save losses
            with open(f"runs/{self.version}/losses.csv", "a") as losses:
                losses.write(
                    f"{epoch_real_d}, {epoch_fake_d}, {epoch_gen_d}\n"
                    )
            # save samples
            self.save_samples(labels)


            if self.epochs_trained < 201:
                self.scheduler_D.step()
                self.scheduler_G.step()

            if self.epochs_trained % 10 == 0:
                self.save_model()
                self.plot_errors(errG_history,
                                 errD_real_history,
                                 errD_fake_history,
                                 n_iter
                                 )

        return errG_history, errD_real_history, errD_fake_history

    def save_model(self):
        path_g = f"runs/{self.version}/weights/G_{self.epochs_trained}.pt"
        path_d = f"runs/{self.version}/weights/D_{self.epochs_trained}.pt"
        torch.save(self.generator.state_dict(), path_g)
        torch.save(self.discriminator.state_dict(), path_d)
        return

    def _load_model(self):
        gnames = glob.glob(f"runs/{self.version}/weights/G_*.pt")
        dnames = glob.glob(f"runs/{self.version}/weights/D_*.pt")

        if len(gnames) == 0 or len(dnames) == 0:
            self.epochs_trained = 0
            print("No checkpoint found. Starting from scratch...")
            return

        find_epoch = lambda x: int(x.split('_')[-1].split('.')[0])

        gfile = max(gnames, key=find_epoch)
        dfile = max(dnames, key=find_epoch)

        self.generator.load_state_dict(torch.load(gfile))
        self.discriminator.load_state_dict(torch.load(dfile))

        self.epochs_trained = find_epoch(gfile)
        print(f"Epoch {self.epochs_trained} loaded.")
        return

    def plot_errors(self, g_err, dreal_err, dfake_err, n_iter):
        try:
            new_errG = signal.resample(g_err, len(dreal_err))
        except ValueError:
            return

        e_errG = [np.mean(new_errG[j:(j+n_iter)])
                  for j in range(0, len(dreal_err), n_iter)]
        e_errD_real = [np.mean(dreal_err[j:(j+n_iter)])
                       for j in range(0, len(dreal_err), n_iter)]
        e_errD_fake = [np.mean(dfake_err[j:(j+n_iter)])
                       for j in range(0, len(dreal_err), n_iter)]

        plt.plot(e_errG)
        plt.plot(e_errD_real)
        plt.plot(e_errD_fake)
        plt.title('distance graph')
        plt.xlabel('batch count')
        plt.ylabel('wasserstein distance')
        plt.legend(['gen', 'discr real', 'disc fake'])
        fig_path = f"runs/{self.version}/distance_{self.epochs_trained}.png"
        plt.savefig(fig_path)
        plt.close()
        return


def _main():
    params = {
        "version": "trial0",
        "lr": .0002,
        "beta1": .0,
        "beta2": .9,
        "batch_size": 256,
        "latent_size": 64,
        "gamma": 10,
        "disc_iter": 5,
        "tau": .9,
        "residual": .1,
        "resblock_channels": 512,
    }

    seq_len = 200

    seqs, labels = get_sequences(length=seq_len)

    # load params if it finds file
    if not os.path.exists(f"runs/{params['version']}/hp.json"):
        os.makedirs(f"runs/{params['version']}/", exist_ok=True)
        with open(f"runs/{params['version']}/hp.json", "w") as f:
            f.write(json.dumps(params))

    gan = WGAN_GP(
        hidden_channels=params['resblock_channels'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        version=params['version'],
        latent_size=params['latent_size'],
        gamma=params['gamma'],
        disc_iter=params['disc_iter'],
        beta1=params['beta1'],
        beta2=params['beta2'],
        tau=params['tau'],
        residual=params['residual']
        )

    errorG_history, errorD_real_history, errorD_fake_history = gan.train(
        torch.tensor(seqs, dtype=torch.float32),
        labels,
        epochs=400
        )

    return


if __name__ == '__main__':
    _main()
