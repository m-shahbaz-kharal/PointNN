import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, dim, num_points):
        super(TNet, self).__init__()
        
        self.dim = dim
        
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim*dim)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.max_pool = nn.MaxPool1d(num_points)
        
    def forward(self, x):
        bs = x.shape[0]
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        
        x = self.max_pool(x).view(bs, -1)
        
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)
        
        I = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda: I = I.cuda()
        
        x = x.view(-1, self.dim, self.dim) + I
        
        return x
    
class PointNetBackbone(nn.Module):
    def __init__(self, num_points, num_global_feats, local_feat=True):
        super(PointNetBackbone, self).__init__()
        
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat
        
        self.tnet1 = TNet(dim=3, num_points=num_points)
        self.tnet2 = TNet(dim=64, num_points=num_points)
        
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)
        
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)
        
    def forward(self, x):
        bs = x.shape[0]
        
        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        
        A_feat = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        
        local_features = x.clone()
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        
        global_feautres, critical_indices = self.max_pool(x)
        global_feautres = global_feautres.view(bs, -1)
        critical_indices = critical_indices.view(bs, -1)
        
        if self.local_feat:
            features = torch.cat((local_features, global_feautres.unsqueeze(-1).repeat(1, 1, self.num_points)), dim=1)
            return features, critical_indices, A_feat
        else:
            return global_feautres, critical_indices, A_feat
        
class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average
        
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, (list, np.ndarray)): self.alpha = torch.Tensor(alpha)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        
    def forward(self, predictions, targets, A):
        bs = predictions.shape[0]
        ce_loss = self.cross_entropy_loss(predictions, targets)
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1,1)).view(-1)
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1)
            if A.is_cuda: I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2,1)))
            reg = self.reg_weight * reg / bs
        else:
            reg = 0
        loss = ((1-pn) ** self.gamma * ce_loss)
        if self.size_average: return loss.mean() + reg
        else: return loss.sum() + reg