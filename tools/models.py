import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, channels, residual=.3):
        super(ResBlock, self).__init__()

        self.res = residual

        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
        )

        return

    def forward(self, input):
        x = self.res_block(input)
        return input + self.res * x


class Generator(nn.Module):
    # Generator network

    def __init__(self, n_chars, seq_len, residual, hidden_channels=100,
                 latent_size=128, n_labels=7, tau=1.):
        super(Generator, self).__init__()

        self.seq_len = seq_len

        self.latent_size = latent_size

        self.resblock_channels = hidden_channels

        self.tau = tau

        self.embed = nn.Embedding(7, 7)

        self.lin = nn.Linear(self.latent_size+n_labels,
                             self.resblock_channels * self.seq_len)

        self.block = nn.Sequential(
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
        )

        self.out = nn.Sequential(
            nn.Conv1d(self.resblock_channels, n_chars, 1),
        )

        return

    def forward(self, input, labels, tau=None, hard=False):
        if tau:
            self.tau = tau

        y = torch.tensor(labels, dtype=torch.float)

        if torch.cuda.is_available():
            y = y.cuda()

        y = y.mm(self.embed.weight.data)

        x = torch.cat([input.view(-1, self.latent_size), y], dim=1)
        x = self.lin(x).view(-1, self.resblock_channels, self.seq_len)

        x = self.block(x)

        x = self.out(x)
        return F.gumbel_softmax(x, tau=self.tau, hard=hard, dim=1)


class Discriminator(nn.Module):
    # Discriminator network

    def __init__(self, n_chars, seq_len, residual, hidden_channels=100,
                 n_labels=7):
        super(Discriminator, self).__init__()

        self.res = residual

        self.seq_len = seq_len

        self.resblock_channels = hidden_channels

        self.embed = nn.Embedding(7, 7)

        self.conv = nn.Conv1d(n_chars, self.resblock_channels, 1)

        self.block = nn.Sequential(
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
            ResBlock(hidden_channels, residual),
        )
        self.res_block1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.resblock_channels+n_labels, self.resblock_channels,
                      5, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.resblock_channels, self.resblock_channels,
                      5, padding=2),
        )

        self.out = nn.Sequential(
            nn.Linear(self.resblock_channels * self.seq_len, 1),
        )

        return

    def forward(self, input, labels):
        y = torch.tensor(labels, dtype=torch.float)

        if torch.cuda.is_available():
            y = y.cuda()

        y = y.mm(self.embed.weight.data)

        x = self.conv(input)

        residual = x
        _, _, x_dim = x.size()
        labels_layer = y.view(-1, 7, 1).expand(-1, 7, x_dim)
        x = torch.cat([x, labels_layer], dim=1)

        x = self.res * self.res_block1(x)
        x += residual

        x = self.block(x)

        x = self.out(x.view(-1, self.resblock_channels * self.seq_len))
        return x
