import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNetBackbone

class BBoxEst(nn.Module):
    def __init__(self, num_points, num_global_feats):
        super(BBoxEst, self).__init__()
        
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)
        
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 16)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.linear_center = nn.Linear(16, 3)
        self.linear_dims = nn.Linear(16, 3)
        # self.linear_roty_sin = nn.Linear(16, 1)
        # self.linear_roty_cos = nn.Linear(16, 1)
        self.linear_roty = nn.Linear(16,1)
        self.linear_class = nn.Linear(16, 2)
        
    def forward(self, x):
        x, crit_idxs, A_feat = self.backbone(x)
        
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)
        
        center = self.linear_center(x)
        dims = self.linear_dims(x)
        # roty_sin = self.linear_roty_sin(x)
        # roty_cos = self.linear_roty_cos(x)
        # roty = torch.atan2(roty_sin, roty_cos)
        roty = self.linear_roty(x)
        obj_class = self.linear_class(x)
        
        return center, dims, roty, obj_class, crit_idxs, A_feat