import torch
import torch.nn as nn
import imageio
import skimage.transform
import numpy as np
import torch.nn.functional as F
from ..loss_fn import FocalLoss, DiceLoss
from roleoftemporalintegration.dataset_fn import stack_input_noise

bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
dice_loss_fn = DiceLoss()


def train_fn(train_dl, model, optimizer, loss_weight,
             t_start, n_backprop_frames, epoch, timeseries_data=False, num_integration_frames=1, beta=1, plot_gif=True):
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
                    input = stack_input_noise(batch, num_integration_frames, batch_size=batch.shape[0], test=False).to(device='cuda')
                    A = input
                #target_batch = batch[..., t * integration_period + 10 : (t + 1) * integration_period + 10].to(device='cuda')
                #target = torch.movedim(target_batch, -1, 2)
                for p in model.parameters(): p.grad = None  # equivalent to zero_grad(), but faster
                P, mu, logvar = model(A)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                time_weight = float(t >= t_start)
                recon_loss = reconstruction_loss(A, P, 'bernoulli')
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss = recon_loss + beta * total_kld
                #target = target.detach().cpu()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1)  # TODO: ??!?
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


def valid_fn(valid_dl, model, loss_weight, t_start, epoch, timeseries_data=False,
             num_integration_frames=1, beta=1, plot_gif=True):
    model.eval()
    plot_loss_valid = 0.0
    n_batches = len(valid_dl)
    TA = True
    with torch.no_grad():
        for batch_idx, (batch, _) in enumerate(valid_dl):
            batch_loss_valid = 0.0
            A_seq, P_seq = [], []
            if timeseries_data:
                num_integration_batches = batch.shape[-1] // num_integration_frames
            else:
                num_integration_batches = 1
            for t in range(num_integration_batches):
                if timeseries_data:
                    input = batch[..., t * num_integration_frames: (t + 1) * num_integration_frames].to(device='cuda')
                    A = torch.movedim(input, -1, 2)
                else:
                    input = stack_input_noise(batch, num_integration_frames, batch_size=batch.shape[0], test=True).to(device='cuda')
                    A = input
                P, mu, logvar = model(A)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                time_weight = float(t >= t_start)
                recon_loss = reconstruction_loss(A, P, 'bernoulli')
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss = recon_loss + beta * total_kld
                # loss = loss_fn(A, P, time_weight, loss_weight, batch_idx, n_batches)
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


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    #logvar = torch.clamp(logvar, min=-50., max=50.)
    logvarexp = logvar.exp()
    klds = -0.5*(1 + logvar - mu.pow(2) - logvarexp)
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


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
                epoch=0, sample_indexes=(0,), output_dir='./', mode='train', timeseries_data=False):
    batch_size, n_channels, n_rows, n_cols, n_frames = A_seq.shape
    img_plot = A_seq.numpy() #* DATASET_STD + DATASET_MEAN
    prediction_plot = P_seq.numpy() #* DATASET_STD + DATASET_MEAN
    v_rect = np.ones((batch_size, n_channels, n_rows, 10, n_frames))
    data_rec = np.concatenate((img_plot, v_rect, prediction_plot), axis=3)
    out_batch = data_rec.transpose((0, 2, 3, 1, 4))
    for s_idx in sample_indexes:
        out_seq = out_batch[s_idx]
        gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
        gif_path = f'{output_dir}{mode}_epoch{epoch:02}_id{s_idx:02}'
        imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)
