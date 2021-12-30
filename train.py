
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from solver import Solver
from net import Net
from data_utils import img_dataset, get_img, split_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)  # relative to where you're running this script from
    parser.add_argument('--num_epochs', type=int, default=900) #relative to where you're running this script from
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    # read data
    imgs, img_labels = get_img()
    print(len(imgs))
    imgs, img_labels = split_data(imgs, img_labels, val_size = 0.2)
    datasets = {x:img_dataset(imgs[x], img_labels[x], data_transforms[x]) for x in ['train', 'val']}

    # model
    #path_wts = './models/model_1640834180.9709306/model_best.pth'
    model = Net()
    #model.load_state_dict(torch.load(path_wts))
    model = model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    print(device)

    ft_solver = Solver(model, optimizer, criterion, datasets, device, params)

    best_model = ft_solver.train()

if __name__ == "__main__":
    main()