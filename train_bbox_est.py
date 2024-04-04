import os
import time
import numpy as np

from ds_loaders.pcdet import PerObjectDataLoader
from models.bbox_est import BBoxEst
from models.pointnet import PointNetLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

dataset_path = "C:\\Users\\MShahbazKharal\\OneDrive - University of Central Florida\\urbanity - MSI-LP\\hands_on\\datasets\\carla_gabby_20ft_intersection_all_lidars\\output\\post\\per_object_pcdet_dataset"
npy_path = os.path.join(dataset_path, 'point_cloud')
label_path = os.path.join(dataset_path, 'label')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

alpha = np.ones(2)
alpha[0]=0.5
alpha[1]=0.5
gamma = 1.
reg_weight = 0.0001

lr = 0.0001
batch_size = 64
epochs = 180


def test_visualize(pcd, center, extent, ry, obj_class):
    pcd_np = pcd.detach().cpu().numpy()[0]
    center = center.detach().cpu().numpy()[0]
    extent = extent.detach().cpu().numpy()[0]
    ry = ry.detach().cpu().numpy()[0]
    obj_class = torch.argmax(obj_class[0]).detach().cpu().numpy()
    
    import open3d as o3d
    print('Class:', PerObjectDataLoader.class_id_to_key[int(obj_class)])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,ry])
    bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    bbox.color=[1,0,0]
    
    o3d.visualization.draw_geometries([pcd, bbox])
    
def train_one_epoch(loader, model, point_net_loss, bbox_loss, optim, scheduler):
    model.train(True)
    
    running_center_loss = 0.
    last_center_loss = 0.
    running_extent_loss = 0.
    last_extent_loss = 0.
    # running_roty_loss = 0.
    # last_roty_loss = 0.
    running_pn_loss = 0.
    last_pn_loss = 0.
    
    running_total_loss = 0.
    last_total_loss = 0.
    i = 0
    
    for pcd, (lbl_center, lbl_extent, lbl_orient, lbl_class) in loader:
        optim.zero_grad()
        
        pcd = pcd.transpose(2,1).to(device)
        lbl_center = lbl_center.to(device)
        lbl_extent = lbl_extent.to(device)
        # lbl_orient = lbl_orient.to(device).reshape(-1,1)
        lbl_class = lbl_class.to(device)
        
        # center, dims, roty, obj_class, crit_idxs, A_feat = model(pcd)
        center, dims, obj_class, crit_idxs, A_feat = model(pcd)
        
        center_loss = bbox_loss(center, lbl_center)
        extent_loss = bbox_loss(dims, lbl_extent)
        # roty_loss = bbox_loss(roty, lbl_orient)
        pn_loss = point_net_loss(obj_class, lbl_class, A_feat)
        
        # total_loss =  center_loss + extent_loss + roty_loss + pn_loss
        total_loss =  center_loss + extent_loss + pn_loss
        total_loss.backward()
        
        optim.step()
        scheduler.step()
        
        running_center_loss += center_loss.item()
        running_extent_loss += extent_loss.item()
        # running_roty_loss += roty_loss.item()
        running_pn_loss += pn_loss.item()
        
        running_total_loss += total_loss.item()
        
        if i % 16 == 15:
            last_center_loss = running_center_loss / 16.
            last_extent_loss = running_extent_loss / 16.
            # last_roty_loss = running_roty_loss / 16.
            last_pn_loss = running_pn_loss / 16.
            last_total_loss = running_total_loss / 16.
            
            print(f'Batch: {i+1}')
            print(f'Avg. Center Loss: {last_center_loss}')
            print(f'Avg. Extent Loss: {last_extent_loss}')
            # print(f'Avg. Rot Y Loss: {last_roty_loss}')
            print(f'Avg. PointNet Loss: {last_pn_loss}')
            print(f'Avg. Total Loss: {last_total_loss}')
            
            running_center_loss = 0.
            running_extent_loss = 0.
            # running_roty_loss = 0.
            running_pn_loss = 0.
            running_total_loss = 0
            
        i += 1
    
    return last_total_loss

def eval_one_epoch(loader, model, point_net_loss, bbox_loss):
    model.eval()
    
    running_loss = 0.
    last_loss = 0.
    i = 0
    
    with torch.no_grad():
        for pcd, (lbl_center, lbl_extent, lbl_orient, lbl_class) in loader:
            pcd = pcd.transpose(2,1).to(device)
            lbl_center = lbl_center.to(device)
            lbl_extent = lbl_extent.to(device)
            lbl_orient = lbl_orient.to(device).reshape(-1,1)
            lbl_class = lbl_class.to(device)
            
            # center, dims, roty, obj_class, crit_idxs, A_feat = model(pcd)
            center, dims, obj_class, crit_idxs, A_feat = model(pcd)
            pn_loss = point_net_loss(obj_class, lbl_class, A_feat)
            # loss = bbox_loss(center, lbl_center) + bbox_loss(dims, lbl_extent) + bbox_loss(roty, lbl_orient) + pn_loss
            loss = bbox_loss(center, lbl_center) + bbox_loss(dims, lbl_extent) + pn_loss
            
            running_loss += loss.item()
            i += 1
            
    last_loss = running_loss / i+1
    return last_loss

def main():
    train_set = PerObjectDataLoader(npy_path, label_path, True)
    val_set = PerObjectDataLoader(npy_path, label_path, False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    print("Length of Training Set:", len(train_set))
    print("Length of Validation Set:", len(val_set))
    
    # for pcd, (lbl_center, lbl_extent, lbl_orient, lbl_class) in train_loader:
    #     test_visualize(pcd, lbl_center, lbl_extent  , lbl_orient, lbl_class)
    
    model = BBoxEst(train_set.num_points, 512).to(device)
    point_net_loss = PointNetLoss(alpha, gamma, reg_weight).to(device)
    bbox_loss = nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.01, step_size_up=2000, cycle_momentum=False)
    
    best_loss = 1_000_000.
    for i in range(epochs):
        print(f'Epoch: {i} ', end='')
        
        loss = train_one_epoch(train_loader, model, point_net_loss, bbox_loss, optim, scheduler)
        print(f'Train Avg. Loss: {loss} ', end='')
        
        loss = eval_one_epoch(val_loader, model, point_net_loss, bbox_loss)
        print(f'Eval Avg. Loss: {loss}')
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')
    
if __name__ == '__main__': main()
    