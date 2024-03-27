import os
import numpy as np

import torch
from torch.utils.data import Dataset

class PerObjectDataLoader(Dataset):
    key_to_class_id = {'Car': 0, 'Pedestrian': 1}
    class_id_to_key = {v:k for k,v in key_to_class_id.items()}
    
    def __init__(self, npy_path, label_path, train=True, num_points=128):
        self.npy_path = npy_path
        self.label_path = label_path
        self.train = train
        self.num_points = num_points
        
        self.npy_files = sorted(os.listdir(self.npy_path), key=lambda x: int(x.split('.')[0]))
        
        train_size = int(0.8 * len(self.npy_files))
        if self.train: self.npy_files = self.npy_files[: train_size]
        else: self.npy_files = self.npy_files[train_size: ]
        
    def __len__(self): return len(self.npy_files)
    
    def __getitem__(self, index):
        npy_file = os.path.join(self.npy_path, self.npy_files[index])
        base_name = os.path.basename(npy_file).split('.')[0]
        lbl_file = os.path.join(self.label_path, base_name + '.txt')
        
        pcd_npy = np.load(npy_file)[:, :3]
        if pcd_npy.shape[0] < self.num_points: pcd_npy = np.pad(pcd_npy, ((0, self.num_points-pcd_npy.shape[0]), (0,0)), mode='constant', constant_values=0)
        elif pcd_npy.shape[0] > self.num_points: pcd_npy = pcd_npy[:self.num_points]
        # range_x = np.max(pcd_npy[:, 0]) - np.min(pcd_npy[:, 0])
        # range_y = np.max(pcd_npy[:, 1]) - np.min(pcd_npy[:, 1])
        # range_z = np.max(pcd_npy[:, 2]) - np.min(pcd_npy[:, 2])
        # range_xyz = np.array([range_x, range_y, range_z])
        # pcd_npy /= range_xyz
        pcd_npy = torch.tensor(pcd_npy, dtype=torch.float32)
        
        with open(lbl_file) as f: lbl_txt = f.readline()
        lbl_data = [float(v) for v in lbl_txt.strip().split(' ')[:7]]
        lbl_center = torch.tensor(lbl_data[0:3], dtype=torch.float32)
        # lbl_center /= range_xyz
        lbl_extent = torch.tensor(lbl_data[3:6], dtype=torch.float32)
        # lbl_extent /= range_xyz
        lbl_orient = torch.tensor(lbl_data[6], dtype=torch.float32)
        lbl_class = lbl_txt.strip().split(' ')[7]
        lbl_class = torch.tensor(self.key_to_class_id[lbl_class], dtype=torch.long)
        
        return pcd_npy, (lbl_center, lbl_extent, lbl_orient, lbl_class)