import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

import logging, coloredlogs

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
        
        coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
        # logging.warning(f"[KITTILoader.py] -> init called!")  

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        
        # logging.info(f"type(self.left): {type(self.left)} type(self.left[0]): {type(self.left[0])} left[0]: {self.left[0]}")
        # logging.info(f"type(self.right): {type(self.right)} type(self.right[0]): {type(self.right[0])} right[0]: {self.right[0]}")
        # logging.info(f"type(self.disp_L): {type(self.disp_L)} type(self.disp_L[0]): {type(self.disp_L[0])} disp_L[0]: {self.disp_L[0]}")


        
    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        # logging.error(f"left_img.shape: {np.array(left_img).shape}")
        # logging.error(f"dataL.shape: {np.array(dataL).shape}")


        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size
        #    logging.error(f"type(dataL): {type(dataL)}")
        #    logging.info(f"[before] dataL.shape: {np.array(dataL).shape}")
           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
        #    logging.info(f"[after crop] dataL.shape: {np.array(dataL).shape}")
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           
        #    logging.warning(f"[before] left_img.shape: {np.array(left_img).shape}")
           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)
        #    logging.warning(f"[after] left_img.shape: {np.array(left_img).shape}")
           
           return left, right, disp_L, left_img, right_img, dataL
        #    return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
