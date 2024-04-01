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



class MultiLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)

        self.layer6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 6 * 6, 4)

    def forward(self, x):

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
    output_size = 4  # Number of output classes
    num_epochs = 10

    mode = 'train'
    csv_file = 'Labeled Data.csv'

    # Create an instance of the Pclass dataset
    dataset = Pclass('train')

    trainset = Pclass('train')
    Trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass('test')
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=8, drop_last=True)

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
