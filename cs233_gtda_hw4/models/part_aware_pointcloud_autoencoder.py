"""
Part-Aware PC-AE.

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

class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier, part_lambda=0.005):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        """
        super(PartAwarePointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = part_classifier
        
        self.xentropy = nn.CrossEntropyLoss()
        self.part_lambda = part_lambda
        
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
        x = torch.transpose(pointclouds, 1, 2) # shape [batch_size, 3, 1024]
        x = self.encoder(x)
        x = torch.transpose(x, 1, 2) # shape [batch_size, 1, 128]
        y = torch.cat([pointclouds, x.repeat(1, 1024, 1)], dim=2) # shape [batch_size, 1024, 3+128=131]
        y = torch.transpose(y, 1, 2) # shape [batch_size, 131, 1024]
        x = self.decoder(x)
        x = x.view(pointclouds.shape)
        y = self.part_classifier(y)
        print(y.size()) # want y to have shape [batch_size, 3, 1024]
        y = torch.transpose(y, 1, 2) # at the end y has shape [batch_size, 1024, 3] as expected
        return x, y

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
            gt = b['part_mask'].to(device)
            recon, seg = self.__call__(batch)
            
            batch_loss = chamfer_loss(batch, recon).mean() + (self.part_lambda * self.xentropy(seg, gt).mean())
            loss_meter.update(batch_loss, len(batch))
            
            optimizer.zero_grad()
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
        segs = []
        for b in loader:
            batch = b['point_cloud'].to(device)
            recon, seg = self.__call__(batch)
            recons.append(recon)
            segs.append(seg)
            recon_loss = chamfer_loss(batch, recon)
            recon_losses.append(recon_loss)
        recons = torch.cat(recons, dim=0)
        segs = torch.cat(segs, dim=0)
        recon_losses = torch.cat(recon_losses, dim=0)
        return recons, recon_losses, segs
