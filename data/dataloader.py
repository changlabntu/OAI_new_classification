import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
from utils.data_utils import imagesc
import pandas as pd


def get_womac4_label():
    df = pd.read_csv('data/womac4_moaks.csv')
    right_knee = df.loc[df['SIDE'] == 'RIGHT', :]
    left_knee = df.loc[df['SIDE'] == 'LEFT', :]
    label = right_knee['V$$WOMKP#'].values > left_knee['V$$WOMKP#'].values
    return label


class Image3DDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted(glob.glob(os.path.join(root_dir[0], '*.tif')))
        self.unique_ids = self._get_unique_ids()
        self.labels = get_womac4_label()

    def _get_unique_ids(self):
        return list(set('_'.join(os.path.basename(f).split('_')[:-1]) for f in self.image_files))

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):

        pain = np.expand_dims(self.labels[idx], 0)

        id_ver = self.unique_ids[idx]
        slices = [f for f in self.image_files if f.startswith(os.path.join(self.root_dir[0], id_ver))]
        slices.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        image_list = []

        for dir in self.root_dir:
            image_3d = []
            for slice_file in slices:
                image_2d = tiff.imread(slice_file.replace(self.root_dir[0], dir))
                image_2d = image_2d / image_2d.max()
                image_3d.append(image_2d)

            image_3d = np.stack(image_3d, axis=0)
            x = torch.from_numpy(image_3d).float()  # (Z, X, Y)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            image_list.append(x)

        print(pain)

        if pain[0]:
            return (image_list[1], image_list[0]), pain  # (R, L)

        elif pain[0]:
            return (image_list[0], image_list[1]), pain  # (L, R)



# Create the dataset
root = "/media/ExtHDD01/Dataset/paired_images/womac4/full/"
root_dir = [root + 'ap/', root + 'bp/']
dataset = Image3DDataset(root_dir)

a, b = dataset.__getitem__(10)
