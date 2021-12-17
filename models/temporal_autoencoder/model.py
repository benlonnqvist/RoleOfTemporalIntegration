import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy


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