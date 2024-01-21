from __future__ import print_function
import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import csv
import xlrd
from torchvision.transforms import transforms
import scipy.ndimage


def load_label(excel_path, dataformat, label_orNot=False, label_index=1):
    if dataformat == '.csv':
        data = pd.read_csv(excel_path)
    else:
        data = pd.read_excel(excel_path)
    label = []
    PIDs = list(data.values[:, 0])
    if label_orNot:
        label = list(data.values[:, label_index])  # label没有做one-hot
    return PIDs, label


class MyDataset(Dataset):
    """
    custom_path: selected train/valid/test dataset PID/path
    all_PIDlabels: All images path & corresponding label--excel/csv/txt document
    """

    def __init__(self, custom_path, all_PIDlabels, root_data_path, transform=None, label_index=1):
        root_data_path = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/npy_external'

        All_img_names, labels_list = load_label(all_PIDlabels, dataformat=os.path.splitext(all_PIDlabels)[-1],
                                                label_orNot=True, label_index=label_index)
        imgs_list = [os.path.join(root_data_path, str(i) + '.npy') for i in All_img_names]

        # self.image_files = np.array(root)
        self.image_files = imgs_list  # train/valid/test
        self.labels = labels_list
        self.transform = transform

    def __getitem__(self, index):
        # x, y = load_npy(self.image_files[index])
        x_path = self.image_files[index]
        # print(x_path)
        x = np.load(x_path, allow_pickle=True)  # Channel × H × W
        y = self.labels[index]

        # CPU tensor
        return torch.FloatTensor(x), torch.FloatTensor(y)


    def __len__(self):
        return len(self.image_files)

