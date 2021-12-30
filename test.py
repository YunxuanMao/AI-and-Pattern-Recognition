import os
import json

import numpy
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from net import Net

from data_utils import test_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_test_img():
    img_paths = []
    imgs = []
    for root,dirs,files in os.walk(r'./data/test'):
        for file in files:
            path = os.path.join(root,file)
            img_paths.append(path[7:])
            img = Image.open(path).convert('RGB')
            imgs.append(img)
    return img_paths, imgs
    

def main():
    
    img_paths, imgs = get_test_img()
    dataset = test_dataset(imgs, img_paths, data_transform)
    test_dataloader = DataLoader(dataset, num_workers = 2)

    path_wts = './models/model_1640834180.9709306/model_best.pth'
    model = Net()
    model.eval()
    model.load_state_dict(torch.load(path_wts))
    model = model.to(device)

    annotations = []
    for inputs, paths in test_dataloader:
        inputs = inputs.to(device)
        paths = paths

        with torch.no_grad():
            outputs = model(inputs)
            #print(inputs)
            _, preds = torch.max(outputs, 1)
            annotations.append({"filename":paths[0], "label": preds.cpu().numpy().tolist()[0]})
    key = {"annotations":annotations}

    with open('./data/submit.json','w') as f:
        json.dump(key, f, sort_keys=False, indent =4)

    
    
if __name__ == "__main__":
    main()