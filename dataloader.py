from __future__ import print_function

import random
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from torch.utils.data import DataLoader
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
import log
from PIL import Image
from imgaug import augmenters as iaa
import string
from tqdm import tqdm
from tensorboardX import SummaryWriter

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
np.random.seed(2)

class Fusionset(Data.Dataset):
    def __init__(self, io, args, root, get_patch=144, transform=None, gray=True, partition='train'):

        self.img1_path = root + '/img1'   
        self.img2_path = root + '/img2'

        self.vsm_img1_path = root + '/vsm_img1'
        self.vsm_img2_path = root + '/vsm_img2'

        self.files_img1 = glob(os.path.join(self.img1_path, '*.*')) 
        self.files_img2 = glob(os.path.join(self.img2_path, '*.*'))
        
        self.files_vsm_img1 = glob(os.path.join(self.vsm_img1_path, '*.*'))
        self.files_vsm_img2 = glob(os.path.join(self.vsm_img2_path, '*.*'))
        
        self.gray = gray
        self.patch_size = get_patch
        self._tensor = transforms.ToTensor()
        self.transform = transform
        self.args = args

        self.num_examples = len(self.files_img1)

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 9]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 9]).astype(np.int)
            np.random.shuffle(self.val_ind)
        io.cprint("number of " + partition + " examples in dataset" + ": " + str(len(self.files_img1)))

    def __len__(self):
        return len(self.files_img1)

    def __getitem__(self, index):
        img_1 = Image.open(self.files_img1[index])
        img_2 = Image.open(self.files_img2[index])
        
        # vsm
        img_vsm_1 = Image.open(self.files_vsm_img1[index])
        img_vsm_2 = Image.open(self.files_vsm_img2[index])

        if self.transform is not None:

            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        if self.gray:
            img_1 = img_1.convert('L')
            img_2 = img_2.convert('L')
            img_vsm_1 = img_vsm_1.convert('L')
            img_vsm_2 = img_vsm_2.convert('L')


        img_1 = np.array(img_1)
        img_2 = np.array(img_2)
        
        img_vsm_1 = np.array(img_vsm_1)
        img_vsm_2 = np.array(img_vsm_2)

        # get patch
        p_1, p_2, vsm_p_1, vsm_p_2 = self.get_patch(img_1, img_2, img_vsm_1, img_vsm_2)

        p_1 = self._tensor(p_1)
        p_2 = self._tensor(p_2)
        
        vsm_p_1 = self._tensor(vsm_p_1)
        vsm_p_2 = self._tensor(vsm_p_2)


        return p_1, p_2, vsm_p_1, vsm_p_2

    def get_patch(self,img1, img2, img3, img4):
        lh , lw = img1.shape[:2]
        l_stride = self.patch_size

        x = random.randint(0,lw - l_stride)
        y = random.randint(0,lh - l_stride)

        img1 = img1[y:y+l_stride,x:x+l_stride]
        img2 = img2[y:y+l_stride,x:x+l_stride]
        img3 = img3[y:y+l_stride,x:x+l_stride]
        img4 = img4[y:y+l_stride,x:x+l_stride]
        
        return img1, img2, img3, img4