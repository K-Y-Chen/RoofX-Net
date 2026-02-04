import torch.nn as nn
import torch 
import numpy as np
from torch import softmax
import matplotlib.pyplot as plt

class SobelLoss(nn.Module):
    def __init__(self, size_average=True, criterion='mse', trainable=False):
        super(SobelLoss, self).__init__()
        self.size_average = size_average
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.trainable = trainable
        
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = self.trainable
        
        if criterion == 'L1':
            self.criterion = nn.SmoothL1Loss()
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        
        
    def forward(self, preds, targets):
        '''
        preds:      (N, cls, H, W)
        targets:    (N, H, W)
        '''
        preds = preds.clone()
        targets = targets.clone()
        
        logic = softmax(preds, dim=1)
        pred_bkg = logic[:, :1, :, :]   # 背景概率    
        pred_fg = 1 - pred_bkg          # 前景概率

        targets[targets != 0] = 1
 
        targets = targets.unsqueeze(1).float()
        # compute edge-aware map for preds
        edge_pred = self.edge_conv(pred_fg)
        # compute edge-aware map for targets
        edge_target = self.edge_conv(targets)
        # compute loss between two maps above        
        
        edge_loss = self.criterion(edge_pred, edge_target)
        # return mean
        if self.size_average:
            return edge_loss.mean()
        else:
            return edge_loss.sum()
        
        
class LaplaceLoss(nn.Module):
    def __init__(self, size_average=True, criterion='mse', trainable=False):
        super(LaplaceLoss, self).__init__()
        self.size_average = size_average
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.trainable = trainable
        
        edge_k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        # edge_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        edge_k = torch.from_numpy(edge_k).float().view(1, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = self.trainable
        
        if criterion == 'L1':
            self.criterion = nn.SmoothL1Loss()
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        
        
    def forward(self, preds, targets):
        '''
        preds:      (N, cls, H, W)
        targets:    (N, H, W)
        '''
        preds = preds.clone()
        targets = targets.clone()
        
        logic = softmax(preds, dim=1)
        pred_bkg = logic[:, :1, :, :]   # 背景概率    
        pred_fg = 1 - pred_bkg          # 前景概率

        targets[targets != 0] = 1
 
        targets = targets.unsqueeze(1).float()
        # compute edge-aware map for preds
        edge_pred = self.edge_conv(pred_fg)
        # compute edge-aware map for targets
        edge_target = self.edge_conv(targets)
        # compute loss between two maps above
        
        edge_loss = self.criterion(edge_pred, edge_target)
        # return mean
        if self.size_average:
            return edge_loss.mean()
        else:
            return edge_loss.sum()