import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm
import trimesh
# import open3d as o3d


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, cates, split, scale_mode, transform=None, num_points=2048):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        self.split = split

        self.ratio = 1.0
        if self.split == "train":
            self.ratio = 0.80
        elif self.split == "val" or self.split == "test":
            self.ratio = 0.20

        self.scale_mode = scale_mode
        self.transform = transform

        self.num_points = num_points

        if 'all' in cates:
            cates = cate_to_synsetid.keys()

        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()

        self.pointclouds = []
        self.stats = None

        self.load()

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_dir = os.path.join(f, synsetid)
                cate_name = synsetid_to_cate[synsetid]

                for j, pc_id in enumerate(os.listdir(cate_dir)):

                    if pc_id.startswith('.'): #if hidden file skip
                        continue

                    mesh_path = os.path.join(cate_dir, pc_id, "models/model_normalized.ply")
                    if not os.path.isfile(mesh_path):
                        continue
                    mesh = trimesh.load(mesh_path, force="mesh")
                    if not isinstance(mesh, trimesh.Trimesh):
                        continue
                    pc = mesh.sample(self.num_points)
                    yield torch.tensor(pc, dtype=torch.float), j, cate_name


        for pc, pc_id, cate_name in _enumerate_pointclouds(self.path):

            if self.scale_mode == 'global_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = self.stats['std'].reshape(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                shift = ((pc_min + pc_max) / 2).view(1, 3)
                scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            pc = (pc - shift) / scale

            self.pointclouds.append({
                'pointcloud': pc,
                'cate': cate_name,
                'id': pc_id,
                'shift': shift,
                'scale': scale
            })

        # # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

