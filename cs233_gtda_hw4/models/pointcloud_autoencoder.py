"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
# from ..losses.chamfer import chamfer_loss

# In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
# one that is written in pure pytorch:
from ..losses.nn_distance import chamfer_loss


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        x = torch.transpose(pointclouds, 1, 2)
        return self.encoder(x).squeeze(-1)

    def __call__(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        x = torch.transpose(pointclouds, 1, 2)
        x = self.encoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.decoder(x)
        x = x.view(pointclouds.shape)
        return x

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()
        
        for b in loader:
            batch = b['point_cloud'].to(device)
            recon = self.__call__(batch)
            batch_loss = chamfer_loss(batch, recon).mean()
            loss_meter.update(batch_loss, len(batch))
            
            batch_loss.backward()
            optimizer.step()
        
        return loss_meter.avg

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: Left for students to decide
        """
        recons = []
        recon_losses = []
        for b in loader:
            batch = b['point_cloud'].to(device)
            recon = self.__call__(batch)
            recons.append(recon)
            recon_loss = chamfer_loss(batch, recon)
            recon_losses.append(recon_loss)
        recons = torch.cat(recons, dim=0)
        recon_losses = torch.cat(recon_losses, dim=0)
        return recons, recon_losses