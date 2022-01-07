import torch
import numpy as np
import random
from roleoftemporalintegration.models.temporal_autoencoder.model import BetaVAE_H
from roleoftemporalintegration.models.temporal_autoencoder.model import TemporalAutoencoder
from roleoftemporalintegration.models.temporal_autoencoder.utils import train_fn, valid_fn
from roleoftemporalintegration.dataset_fn import get_datasets_seg, get_SQM_dataset, get_cifar_dataset
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 1. make sqm dataset
# 2. make models on the rome list
# 3. test them on sqm

def instantiate_model():
    load_model, n_epochs_run, n_epoch_save, epoch_to_load = False, 1000, 10, 300
    name = 'TemporalAutoencoderReconstruct'
    batch_size = 32
    n_layers, t_start = 4, 0
    encoder_channels = (3, 64, 128, 128)
    learning_rate = 1e-4
    lr_decay_time, lr_decay_rate = 1, 1.0 # set decay rate to 1.0 for no decay
    loss_w = {
        'img_bce': 0.0,
        'img_mae': 100.0,
        'img_mse': 0.0}
    model_name = \
          f'_TD{encoder_channels}_{name}'
    model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')

    # Dataset
    dataset_path = r'C:\Users\loennqvi\Github\seg_net_vgg\data\MOTS'
    #dataset_path = r'D:\DL\datasets\kitti\mots'
    n_samples, tr_ratio = 1000, 0.80  # n_train(valid)_samples = ((1-)tr_ratio) * n_samples
    n_frames, n_backprop_frames = 100, 1
    augmentation, remove_ground = True, False
    integration_period = 16
    n_classes = 3 if remove_ground else 4
    # train_dl, valid_dl = get_datasets_seg(
    #     dataset_path, tr_ratio, batch_size_train, batch_size_valid, n_frames,
    #     augmentation=augmentation, n_classes=n_classes, remove_ground=remove_ground)
    train_dl, valid_dl = get_cifar_dataset(batch_size=batch_size)
    sqm_dataset_path = r'C:\Users\loennqvi\Github\seg_net_vgg\data\SQM'
    #valid_dl = get_SQM_dataset(sqm_dataset_path, n_classes, remove_ground)

    # Load the model
    if not load_model:
        print(f'\nCreating model: {model_name}')
        model = BetaVAE_H(model_name=model_name)
        train_losses, valid_losses, last_epoch = [], [], 0
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            range(lr_decay_time, (n_epochs_run + 1) * 10, lr_decay_time),
            gamma=lr_decay_rate)
        train_losses, valid_losses = [], []
    else:
        print(f'\nLoading model: {model_name}')
        model, optimizer, scheduler, train_losses, valid_losses = \
            BetaVAE_H.load_model(model_name=model_name)
        last_epoch = scheduler.last_epoch

    # Train the network
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training network ({n_params} trainable parameters)')
    for epoch in range(last_epoch, last_epoch + n_epochs_run):
        print(f'\nEpoch n°{epoch}')
        train_losses.append(train_fn(
            train_dl=train_dl, model=model, optimizer=optimizer, loss_weight=loss_w, t_start=t_start,
            n_backprop_frames=n_backprop_frames, epoch=epoch, timeseries_data=False,
            num_integration_frames=integration_period, beta=4))
        valid_losses.append(valid_fn(valid_dl=valid_dl, model=model, loss_weight=loss_w, t_start=t_start,
                                     epoch=epoch, timeseries_data=False, num_integration_frames=integration_period,
                                     beta=4))
        scheduler.step()
        if (epoch + 1) % n_epoch_save == 0:
            model.save_model(epoch)


if __name__ == '__main__':
    instantiate_model()
