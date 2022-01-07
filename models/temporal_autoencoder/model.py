import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.nn import init
from torch.autograd import Variable


def reparametrize(mu, logvar, mult=0, pos=None):
    std = logvar.div(2).exp()
    #new_mu = deepcopy(mu)
    eps = Variable(std.data.new(std.size()).normal_())
    #new_mu[0][pos] += mult * std[0][pos] # * eps[0][0]
    return mu + std * eps # * mult


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, model_name, z_dim=100, nc=3):
        super(BetaVAE_H, self).__init__()
        self.model_name = model_name
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv3d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            #nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            #nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            #nn.BatchNorm3d(64),
            nn.Conv3d(64, 256, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            #nn.BatchNorm3d(256),
            # nn.Conv3d(64, 64, 4, 2, 1),            # B, 256,  1,  1
            # nn.ReLU(True),
            # nn.Conv3d(64, 256, 4, 1),  # B, 256,  1,  1
            # nn.ReLU(True),
            #nn.BatchNorm2d(num_features=256),
            View((-1, 256*2*2)),                 # B, 256
            nn.Linear(256*2*2, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256*2*2),               # B, 256
            View((-1, 256, 1, 2, 2)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 64, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 32, 4, 2, 1),  # B, nc, 64, 64
            nn.ReLU(True),
            nn.ConvTranspose3d(32, nc, 4, 2, 1),
            nn.Sigmoid()
        )

        self.weight_init()
        self.to('cuda')

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, mult=None, pos=None):
        distributions = self._encode(x)
        logvar = distributions[:, self.z_dim:]
        mu = distributions[:, :self.z_dim]  # to test how values change the output images, vary mu only, not logvar
        if mult is None:

            # 1. run inference on an image
            # 2. fix all the latent variables, then traverse one across a few standard deviations
            # 3. plot each traversal node
            # 10 latent variables x 5 images per variable = 50 traversal images
            # logvar = distributions[:, self.z_dim:]

            z = reparametrize(mu, logvar)
            x_recon = self._decode(z)

            return x_recon, mu, logvar
        z = reparametrize(mu, logvar, mult, pos)
        x_recon = self._decode(z)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def save_model(self, epoch):
        model_savepath = f'./ckpt/{self.model_name}/pth/{epoch}.pth'
        torch.save(self.state_dict(), model_savepath)

    @classmethod
    def load_model(cls, model_name, epoch):
        # model name used here
        raise NotImplementedError


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class TemporalAutoencoder(nn.Module):
    def __init__(self, model_name, encoder_channels):
        super().__init__()
        self.model_name = model_name
        self.encoder_channels = encoder_channels
        self.decoder_channels = deepcopy(encoder_channels)[::-1]
        self.n_encoder_layers = len(encoder_channels) - 1
        self.n_decoder_layers = len(self.decoder_channels) - 1

        model_path = f'./ckpt/{model_name}/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Encoder connections
        encoder_stack = []
        for l in range(self.n_encoder_layers):
            inn, out = self.encoder_channels[l], self.encoder_channels[l + 1]
            encoder_stack.append(nn.Sequential(
                nn.Conv3d(in_channels=inn, out_channels=out, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=1, num_channels=out),
                nn.GELU(),
                nn.AvgPool3d(kernel_size=2, stride=2)
            ))
        self.encoder_stack = nn.ModuleList(encoder_stack)

        # Decoder connections
        decoder_stack = []
        for l in range(self.n_decoder_layers):
            inn, out = self.decoder_channels[l], self.decoder_channels[l + 1]
            if l == self.n_decoder_layers - 1:
                decoder_stack.append(nn.Sequential(
                    nn.Conv3d(in_channels=inn, out_channels=out, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=out),
                    nn.GELU(),
                    nn.ConvTranspose3d(
                        in_channels=out,
                        out_channels=out,
                        kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.Hardtanh(min_val=0.0, max_val=1.0)
                ))
            else:
                decoder_stack.append(nn.Sequential(
                    nn.Conv3d(in_channels=inn, out_channels=out, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=out),
                    nn.GELU(),
                    nn.ConvTranspose3d(
                        in_channels=out,
                        out_channels=out,
                        kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GELU()
                ))
        self.decoder_stack = nn.ModuleList(decoder_stack)

        self.to('cuda')

    def forward(self, x):
        # Encoder pass
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)
            noise = 0.1 * torch.randn(size=x.size()).to(device='cuda')
            x = x + noise

        # Decoder pass
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x)
            noise = 0.1 * torch.randn(size=x.size()).to(device='cuda')
            x = x + noise

        return x

    def save_model(self, optimizer, scheduler, train_losses, valid_losses):

        last_epoch = scheduler.last_epoch
        torch.save({
            'model_name': self.model_name,
            'model_params': self.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
            'optimizer_params': optimizer.state_dict(),
            'scheduler_params': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses},
            f'./ckpt/{self.model_name}/ckpt_{last_epoch:02}.pt')
        print('SAVED')
        plt.plot(list(range(last_epoch)), valid_losses, label='valid')
        plt.plot(list(range(last_epoch)), train_losses, label='train')
        plt.legend()
        plt.savefig(f'./ckpt/{self.model_name}/loss_plot.png')
        plt.close()

    @classmethod
    def load_model(cls, model_name, encoder_channels, epoch_to_load=None):

        ckpt_dir = f'./ckpt/{model_name}/'
        list_dir = [c for c in os.listdir(ckpt_dir) if ('decoder' not in c and '.pt' in c)]
        ckpt_path = list_dir[-1]  # take last checkpoint (default)
        for ckpt in list_dir:
            if str(epoch_to_load) in ckpt.split('_')[-1]:
                ckpt_path = ckpt
        save = torch.load(ckpt_dir + ckpt_path)
        model = cls(
            model_name=model_name, encoder_channels=encoder_channels)
        model.load_state_dict(save['model_params'])
        optimizer = save['optimizer']
        scheduler = save['scheduler']
        optimizer.load_state_dict(save['optimizer_params'])
        scheduler.load_state_dict(save['scheduler_params'])
        valid_losses = save['valid_losses']
        train_losses = save['train_losses']
        return model, optimizer, scheduler, train_losses, valid_losses