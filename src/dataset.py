import os, sys, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image, ImageOps
import random
from sklearn.utils import class_weight
import cv2
# from do_augmentation import augment



class AlzheimersClassficiation(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, meta, transform=None):
        """
        Args:
            root_dir (string): Path to root directory containing images
            meta_file (string): Path to csv file containing images metadata (image_id, class)

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.meta = meta
        self.transform = transform
        self.df = pd.read_csv(self.meta, index_col=False)
        self.image_paths = self.df['path'].to_list()
        self.classes = sorted(self.df['label'].unique().tolist())
        self.class_id = {i:j for i, j in enumerate(self.classes)}
        self.class_to_id = {value:key for key,value in self.class_id.items()}
        self.class_count =  self.df['label'].value_counts().to_dict()
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),
                                            ])
        

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        # print(img_path)
        image = Image.open(img_path)
        # image = cv2.imread(img_path)
        # # print(f'Image: {image.size}')
        image_tensor = self.transform(image)
        # tensor = image_tensor
        # print(f'Tensor Shape: {image_tensor.shape}')
        image_tensor = image_tensor[:3,::]
        label_id = torch.tensor(self.class_to_id[str(label)])
        return image_tensor, label_id
