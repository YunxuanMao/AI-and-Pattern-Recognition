import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split

import json


def get_img():
    
    # read data from json
    with open('./data/train.json','r') as obj:
        img_dict = json.load(obj)

    # get the list of images
    img_list = img_dict['annotations']
    
    # get the path and label of every image
    imgs = []
    img_labels = []
    for d in img_list:
        path = './data/'+d['filename']
        img = Image.open(path).convert('RGB')
        imgs.append(img)
        img_labels.append(d['label'])
    
    return imgs, img_labels

def split_data(X, y, val_size = 0.2):
    '''
    X: data
    y: labels
    '''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
    data = {'train': X_train, 'val': X_val}
    labels = {'train': y_train, 'val': y_val}
    return data, labels
    

class img_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, img_labels, transform):
        self.imgs = imgs
        self.img_labels = img_labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.transform(img)
        label = self.img_labels[index]
        return img, label

    def __len__(self):
        return len(self.imgs)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, img_paths, transform):
        self.imgs = imgs
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.transform(img)
        img_path = self.img_paths[index]
        return img, img_path

    def __len__(self):
        return len(self.imgs)
     