import os, shutil, sys
import numpy as np
import torch, torch.nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import scipy.io as scio
from pathlib import Path
from utils import *
from glob import glob

import config


def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = torch.zeros(N * 99).float().reshape(N, 99)
    alpha_exp = torch.zeros(N * 29).float().reshape(N, 29)
    # if config.use_cuda:
    #     alpha_shp, alpha_exp = alpha_shp.to(config.device), alpha_exp.to(config.device)

    alpha_shp[:, :40] = param[:, 12:52]
    alpha_exp[:, :10] = param[:, 52:]
    return p, offset, alpha_shp, alpha_exp


class MyDataSet(Dataset):
    def __init__(
        self,
        max_number_class=100000,
        root="../",
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.catalog = root

        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")
        self.shape_mat = dict(scio.loadmat("/data2/lmd2/imgc/shape.mat"))
        self.transform_with_lms = transforms.Compose(
            [
                # transforms.RandomResizedCrop(112,scale=(0.9,1),ratio=(1, 1)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.RandomRotation(10)]),
                transforms.RandomCrop((112, 96))
            ]
        )

        file_list_path = root + "file_path_list_imgc2.txt"
        count = 0
        with open(file_list_path) as ins:
            for ori_img_path in ins:
                ori_img_path = ori_img_path[0:-1]
                ori_mat_path = ori_img_path[:-3] + "mat"
                id = os.path.split(os.path.split(ori_img_path)[0])[-1]
                if str(id) in self.shape_mat and os.path.exists(ori_mat_path):
                    self.data.append([ori_img_path, id, ori_mat_path])
                    count += 1
                    print("loaded:%d\r" % count, sys.stdout.flush())
                    if count == 128 * 40:
                        return
                        pass

    def __getitem__(self, index):
        # index = 0
        imgname, label, mat_path = self.data[index]
        img = Image.open(imgname)

        mat = scio.loadmat(mat_path)
        lms = mat["lm"].reshape(-1)

        img, lms = self.transform_with_lms(img, lms)

        lms = torch.from_numpy(lms)
        shape = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform(img)
        return img, shape, lms

    def __len__(self):
        return len(self.data)


class AFLW2000DataSet(Dataset):
    def __init__(
        self,
        max_number_class=100000,
        root="../",
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        self.root = root
        self.data = []
        self.catalog = root
        self.transform = transform
        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")

        types = ("*.jpg", "*.png")
        image_path_list = []
        for files in types:
            image_path_list.extend(glob(os.path.join(root, files)))
        # file_list_path = root  + '../file_path_list_AFLW2000_align.txt'
        print(len(image_path_list))
        # count=0

        for ori_img_path in image_path_list:
            # ori_img_path = ori_img_path[0:-1]
            ori_mat_path = ori_img_path[:-3] + "mat"
            # print(ori_img_path)
            self.data.append([ori_img_path, ori_mat_path])
            # count += 1
            # print("loaded:%d\r" % count,
            # sys.stdout.flush())
        # imgname, mat_path = self.data[0]
        # img=cv2.imread(imgname)
        # mat = scio.loadmat(mat_path)
