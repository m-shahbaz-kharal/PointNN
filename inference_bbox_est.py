import os
import numpy as np

from .ds_loaders.pcdet import PerObjectDataLoader
from .models.bbox_est import BBoxEst
from .models.pointnet import PointNetLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

checkpoint = 'best_model.pth'
dataset_path = "C:\\Users\\mu290328\OneDrive - University of Central Florida\\urbanity - MSI-LP\\hands_on\\datasets\\carla_gabby_20ft_intersection_all_lidars\\output\\post\\per_object_pcdet_dataset"
npy_path = os.path.join(dataset_path, 'point_cloud')
label_path = os.path.join(dataset_path, 'label')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def inference_visualize(pcd, center, extent, ry, obj_class):
    pcd_np = pcd.detach().cpu().numpy()[0]
    center = center.detach().cpu().numpy()[0]
    extent = extent.detach().cpu().numpy()[0]
    ry = ry.detach().cpu().numpy()[0][0]
    obj_class = torch.argmax(obj_class[0]).detach().cpu().numpy()
    
    import open3d as o3d
    print('Class:', PerObjectDataLoader.class_id_to_key[int(obj_class)])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,ry])
    bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    bbox.color=[1,0,0]
    
    o3d.visualization.draw_geometries([pcd, bbox])

def main():
    val_set = PerObjectDataLoader(npy_path, label_path, False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print("Length of Validation Set:", len(val_set))
    
    model = BBoxEst(val_set.num_points, 512).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    
    with torch.no_grad():
        for pcd_orig, (lbl_center, lbl_extent, lbl_orient, lbl_class) in val_loader:
                pcd = pcd_orig.transpose(2,1).to(device)
                lbl_center = lbl_center.to(device)
                lbl_extent = lbl_extent.to(device)
                lbl_orient = lbl_orient.to(device).reshape(-1,1)
                lbl_class = lbl_class.to(device)
                center, dims, roty, obj_class, crit_idxs, A_feat = model(pcd)
                inference_visualize(pcd_orig, center, dims, roty, obj_class)
    
if __name__ == '__main__': main()
    