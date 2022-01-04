import torch
import torch.nn as nn
import imageio
import skimage.transform
import numpy as np

from ..loss_fn import FocalLoss, DiceLoss
from roleoftemporalintegration.dataset_fn import stack_input_noise

bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
dice_loss_fn = DiceLoss()


def train_fn(train_dl, model, optimizer, loss_weight,
             t_start, n_backprop_frames, epoch, timeseries_data=False, num_integration_frames=1, plot_gif=True):
    model.train()
    plot_loss_train = 0.0
    n_batches = len(train_dl)
    TA = True
    with torch.autograd.set_detect_anomaly(True):
        for batch_idx, (batch, _) in enumerate(train_dl):
            batch_loss_train = 0.0
            A_seq, P_seq = [], []
            if timeseries_data:
                num_integration_batches = batch.shape[-1] // num_integration_frames
            else:
                num_integration_batches = 1
            for t in range(num_integration_batches):
                if timeseries_data:
                    input = batch[..., t * num_integration_frames : (t + 1) * num_integration_frames].to(device='cuda')
                    A = torch.movedim(input, -1, 2)
                else:
                    input = stack_input_noise(batch, num_integration_frames).to(device='cuda')
                    A = input
                #target_batch = batch[..., t * integration_period + 10 : (t + 1) * integration_period + 10].to(device='cuda')
                #target = torch.movedim(target_batch, -1, 2)
                P = model(A)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                time_weight = float(t >= t_start)
                loss = loss_fn(A, P, time_weight, loss_weight, batch_idx, n_batches)
                #target = target.detach().cpu()
                for p in model.parameters(): p.grad = None  # equivalent to zero_grad(), but faster
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # TODO: ??!?
                optimizer.step()
                batch_loss_train += loss.detach().item() / num_integration_batches
            plot_loss_train += batch_loss_train / n_batches
            if batch_idx == 0 and plot_gif:
                A_seq = torch.cat(A_seq, axis=2).permute((0, 1, 3, 4, 2))
                P_seq = torch.cat(P_seq, axis=2).permute((0, 1, 3, 4, 2))
                plot_recons(A_seq, P_seq, epoch=epoch,
                            output_dir=f'./ckpt/{model.model_name}/')

    print(f'\r\nEpoch train loss : {plot_loss_train}')
    return plot_loss_train

def valid_fn(valid_dl, model, loss_weight, t_start, epoch, integration_period, plot_gif=True):
    model.eval()
    plot_loss_valid = 0.0
    n_batches = len(valid_dl)
    TA = True
    with torch.no_grad():
        for batch_idx, (batch, _) in enumerate(valid_dl):
            batch_loss_valid = 0.0
            A_seq, P_seq = [], []
            num_integration_batches = batch.shape[-1] // integration_period
            for t in range(num_integration_batches):
                input = batch[..., t * integration_period: (t + 1) * integration_period].to(device='cuda')
                A = torch.movedim(input, -1, 2)
                P = model(A)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                time_weight = float(t >= t_start)
                loss = loss_fn(A, P, time_weight, loss_weight, batch_idx, n_batches)
                batch_loss_valid += loss.item() / num_integration_batches
            plot_loss_valid += batch_loss_valid / n_batches
            if batch_idx == 0 and plot_gif:
                A_seq = torch.cat(A_seq, axis=2).permute((0, 1, 3, 4, 2))
                P_seq = torch.cat(P_seq, axis=2).permute((0, 1, 3, 4, 2))
                plot_recons(
                    A_seq, P_seq, epoch=epoch,
                    output_dir=f'./ckpt/{model.model_name}/',
                    mode='test' if epoch == -1 else 'valid')

    print(f'\r\nEpoch valid loss : {plot_loss_valid}')
    return plot_loss_valid


def loss_fn(A, P, time_weight, loss_weight, batch_idx, n_batches):
    # Image prediction loss (unsupervised)
    img_bce_loss = bce_loss_fn(P, A) * loss_weight['img_bce'] if loss_weight['img_bce'] else 0.0
    img_mae_loss = mae_loss_fn(P, A) * loss_weight['img_mae'] if loss_weight['img_mae'] else 0.0
    img_mse_loss = mse_loss_fn(P, A) * loss_weight['img_mse'] if loss_weight['img_mse'] else 0.0

    # Total loss
    img_loss = img_bce_loss + img_mae_loss + img_mse_loss
    print(
        f'\rBatch ({batch_idx + 1}/{n_batches})[' +
        f'image: {img_loss:.3f} (bce: {img_bce_loss:.3f}, ' +
        f'mae: {img_mae_loss:.3f}, mse: {img_mse_loss:.3f})' +
        f']', end='')
    return img_loss * time_weight


def plot_recons(A_seq, P_seq,
                epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):
    batch_size, n_channels, n_rows, n_cols, n_frames = A_seq.shape
    img_plot = A_seq.numpy() * DATASET_STD + DATASET_MEAN
    prediction_plot = P_seq.numpy() * DATASET_STD + DATASET_MEAN
    v_rect = np.ones((batch_size, n_channels, n_rows, 10, n_frames))
    data_rec = np.concatenate((img_plot, v_rect, prediction_plot), axis=3)
    out_batch = data_rec.transpose((0, 2, 3, 1, 4))
    for s_idx in sample_indexes:
        out_seq = out_batch[s_idx]
        gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
        gif_path = f'{output_dir}{mode}_epoch{epoch:02}_id{s_idx:02}'
        imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)
