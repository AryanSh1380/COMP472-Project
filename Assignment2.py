import torch
import numpy as np
from torch.utils.data import Dataset
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import pandas as pd


# Function to load CIFAR10 dataset


def cifar_loader(batch_size, shuffle_test=False):
    # Normalization values for CIFAR10 dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    # Loading training dataset with data augmentation techniques
    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, 4),
                                         transforms.ToTensor(),
                                         normalize
                                     ]))
    # Loading test dataset
    test_dataset = datasets.CIFAR10('./data', train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    # Creating data loaders for training and testing
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


# Hyperparameters and settings


# class DataloaderName(Dataset):
#     def __init__(self, inputprameters):
#
#         #codes
#
#     def __getitem__(self, index):
#
#         # code
#
#         return output
#
#     def __len__(self):
#         return self.__data.shape[0]


class Pclass(Dataset):
    def __init__(self, mode):
        csv_file = 'Labeled Data.csv'

        self.mode = mode
        self.data_frame = pd.read_csv(csv_file)  # Load CSV file
        self.image_paths = self.data_frame['path'].tolist()
        self.labels = self.data_frame['label'].tolist()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(os.getcwd(), self.image_paths[idx])  # Construct full image path
        img = Image.open(img_path).convert('RGB')  # Open image
        img = self.transform(img)  # Apply transformations
        label = self.labels[idx]  # Get label
        return img, label

    # def __len__(self):
    #     return len(self.allaimges)
    #
    # def __getitem__(self, idx):
    #     Im = self.mytransform(Image.open(self.allaimges[idx]))
    #     Cls = self.clsLabel[idx]
    #
    #     return Im, Cls


class MultiLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, output_size)

        self.layer1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        self.layer5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)

        self.layer6 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 6 * 6, 4)

    def forward(self, x):
        # x = F.relu(self.fc1(x.view(x.size(0),-1)))
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(self.fc3(x), dim=1)

        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(self.Maxpool(F.leaky_relu(self.layer3(x))))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        x = self.B5((F.leaky_relu(self.layer5(x))))
        x = self.B6(self.Maxpool(F.leaky_relu(self.layer6(x))))

        return self.fc(x.view(x.size(0), -1))


if __name__ == '__main__':

    batch_size = 400
    test_batch_size = 100
    input_size = 3 * 96 * 96  # 3 channels, 32x32 image size
    hidden_size = 50  # Number of hidden units
    output_size = 10  # Number of output classes (CIFAR-10 has 10 classes)
    num_epochs = 10

    mode = 'train'
    csv_file = 'Labeled Data.csv'

    # Create an instance of the Pclass dataset
    dataset = Pclass('train')

    trainset = Pclass('train')
    Trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass('test')
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    # train_loader, _ = cifar_loader(batch_size)
    # _, test_loader = cifar_loader(test_batch_size)
    # dataloader = DataLoader(dataset=IrisDataset('iris.data'),
    #                         batch_size=10,
    #                         shuffle=True)

    epochs = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load('path'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC = 0
    for epoch in range(epochs):
        running_loss = 0
        for instances, labels in Trainloader:
            optimizer.zero_grad()

            label_map = {'Neutral': 0, 'Surprised': 1, 'Happy': 2, 'Focused': 3}
            numerical_labels = [label_map[label] for label in labels]
            tensor_labels = torch.tensor(numerical_labels).cuda()

            output = model(instances.cuda())
            loss = criterion(output, tensor_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(running_loss / len(Trainloader))

        model.eval()
        with torch.no_grad():
            allsamps = 0
            rightPred = 0

            for instances, labels in testloader:
                output = model(instances.cuda())
                predictedClass = torch.max(output, 1)
                allsamps += output.size(0)
                rightPred += (torch.max(output, 1)[1] == tensor_labels).sum()

            ACC = rightPred / allsamps
            print('Accuracy is=', ACC * 100)
            if ACC > BestACC:
                BestACC = ACC
                # torch.save(model.state_dict())
                # torch.save(model.state_dict(), 'path')
        model.train()
