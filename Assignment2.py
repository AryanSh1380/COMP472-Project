from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import pandas as pd
from sklearn.model_selection import train_test_split

class Pclass(Dataset):
    def __init__(self, mode):
        csv_file = 'Labeled Data.csv'
        self.mode = mode
        data_frame = pd.read_csv(csv_file)  # Load CSV file

        # Splitting the data into train and test sets
        train_df, test_df = train_test_split(data_frame, test_size=0.15, random_state=42)

        # Select the appropriate subset based on mode
        if mode == 'train':
            self.data_frame = train_df
        elif mode == 'test':
            self.data_frame = test_df
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        self.image_paths = self.data_frame['path'].tolist()
        self.labels = self.data_frame['label'].tolist()

        # Define your transformations here
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Resize the image to 96x96
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(os.getcwd(), self.image_paths[idx])  # Construct full image path
        img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        img = self.transform(img)  # Apply transformations

        # Convert labels to numerical format
        label_map = {'Neutral': 0, 'Surprised': 1, 'Happy': 2, 'Focused': 3}
        label = label_map[self.labels[idx]]  # Convert label from string to integer

        return img, label


class MultiLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        k_size = 3

        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=k_size, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)

        self.layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=k_size, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)

        self.Maxpool = nn.MaxPool2d(2)

        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k_size, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)

        self.layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k_size, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k_size, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)

        self.layer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=k_size, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(128)

        self.layer7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k_size, padding=1, stride=1)
        self.B7 = nn.BatchNorm2d(256)

        self.layer8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=k_size, padding=1, stride=1)
        self.B8 = nn.BatchNorm2d(256)

        self.layer9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k_size, padding=1, stride=1)
        self.B9 = nn.BatchNorm2d(512)

        self.layer10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=k_size, padding=1, stride=1)
        self.B10 = nn.BatchNorm2d(512)

        self.layer11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k_size, padding=1, stride=1)
        self.B11 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(512*3*3, 4)

    def forward(self, x):
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(self.Maxpool(F.leaky_relu(self.layer3(x))))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        x = self.B5((F.leaky_relu(self.layer5(x))))
        x = self.B6(self.Maxpool(F.leaky_relu(self.layer6(x))))
        x = self.B7(F.leaky_relu(self.layer7(x)))
        x = self.B8(self.Maxpool(F.leaky_relu(self.layer8(x))))
        x = self.B9((F.leaky_relu(self.layer9(x))))

        #print(x.size())

        return self.fc(x.view(x.size(0), -1))


if __name__ == '__main__':

    batch_size = 32
    input_size = 3 * 96 * 96  # 3 channels, 96x96 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes

    mode = 'train'
    csv_file = 'Labeled Data.csv'

    # Create an instance of the Pclass dataset
    dataset = Pclass('train')

    trainset = Pclass('train')
    Trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass('test')
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    BestACC = 0
    for epoch in range(epochs):
        running_loss = 0
        for instances, labels in Trainloader:
            optimizer.zero_grad()

            tensor_labels = torch.tensor(labels).cuda()

            output = model(instances.cuda())
            loss = criterion(output, tensor_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(running_loss / len(Trainloader))

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for instances, labels in testloader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy:.2f}%')

        if accuracy > BestACC:
            BestACC = accuracy

            torch.save(model.state_dict(), 'C:/Users/saaba/Desktop/model_epoc50_k3_11Layer.pt')

        print(f'Best accuracy on the test set: {BestACC:.2f}%')

        model.train()