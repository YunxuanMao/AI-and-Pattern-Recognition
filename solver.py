import time, tqdm
import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


class Solver(object):
    def __init__(self, model, optimizer, criterion, datasets, device, params, **kwargs):
        self.model = model
        self.datasets = datasets
        self.device = device

        self.batch_size = params["batch_size"]
        self.num_epochs = params["num_epochs"]
        self.num_workers = params["num_workers"]

        self.optimizer = optimizer
        self.criterion = criterion

        self.save_every = kwargs.pop("save_every", 10)

        self.dataloaders = {}
        for key, value in self.datasets.items():
            self.dataloaders[key] = DataLoader(self.datasets[key], self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.writer = SummaryWriter()
        self.model_path = './models/model_{}/'.format(time.time())
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path) 

    def train(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in tqdm.tqdm(range(self.num_epochs)):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                #if phase == 'train':
                #    scheduler.step()

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = running_corrects.double() / len(self.datasets[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                self.writer.add_scalar('{} Loss'.format(phase), epoch_loss, global_step = epoch)
                self.writer.add_scalar('{} Acc'.format(phase), epoch_acc, global_step = epoch)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), self.model_path+'model_{}.pth'.format(epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, self.model_path+'model_best.pth')
        return self.model