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
        # Initialize the dataset with a mode (train or test)
        csv_file = 'Labeled Data.csv'
        self.mode = mode
        data_frame = pd.read_csv(csv_file)  # Load CSV file

        # Splitting the data into train and test sets with fixed random state
        train_df, test_df = train_test_split(data_frame, test_size=0.15, random_state=42)
        training, validation = train_test_split(train_df, test_size=0.1765, random_state=42)

        # Select the appropriate subset based on mode
        if mode == 'train':
            self.data_frame = training
        elif mode == 'validation':
            self.data_frame = validation
        elif mode == 'test':
            self.data_frame = test_df
        else:
            raise ValueError("Mode must be 'train', 'validation', or 'test'")

        # Extract image paths and labels
        self.image_paths = self.data_frame['path'].tolist()
        self.labels = self.data_frame['label'].tolist()

        # Define transformations and convert to tensors
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Resize the image to 96x96
            transforms.ToTensor(),
        ])

    def __len__(self):
        # return number of items in dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # returns corresponding image and label given an index
        img_path = os.path.join(os.getcwd(), self.image_paths[idx])  # Construct full image path
        img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        img = self.transform(img)  # Apply transformations

        # Convert labels to numerical format
        label_map = {'Neutral': 0, 'Surprised': 1, 'Happy': 2, 'Focused': 3}
        label = label_map[self.labels[idx]]  # Convert label from string to integer

        return img, label


class MultiLayerFCNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Kernel size for layers
        k_size = 3

        # 1st convolutional block with batch normalizations
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

        # Last fully connected layer for 4 categories
        self.fc = nn.Linear(1024 * 1 * 1, 4)

    def forward(self, x):
        # Forward Pass initialization
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
        x = self.B10(self.Maxpool(F.leaky_relu(self.layer10(x))))
        x = self.B11((F.leaky_relu(self.layer11(x))))

        # print(x.size())

        return self.fc(x.view(x.size(0), -1))


if __name__ == '__main__':

    print(torch.cuda.is_available())

    batch_size = 32

    # Create an instance of the dataset for training and testing
    trainset = Pclass('train')
    Trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    validationset = Pclass('validation')
    ValidationLoader = DataLoader(validationset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass('test')
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=8, drop_last=True)

    epochs = 10  # Number of iterations
    # Set device to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = MultiLayerFCNet().to(device)
    model = nn.DataParallel(model).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    BestACC = 0

    for epoch in range(epochs):
        model.train()  # Set to training mode
        running_loss = 0
        for instances, labels in Trainloader:
            instances, labels = instances.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero parameter gradients
            output = model(instances)  # Forward pass
            loss = criterion(output, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # optimize

            running_loss += loss.item()

        # Average loss per batch
        print(running_loss / len(Trainloader))

        model.eval()  # Set model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():  # Disable gradient calculations
            for instances, labels in ValidationLoader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)  # Forward pass
                _, predicted = torch.max(output.data, 1)  # Prediction
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the validation set: {accuracy:.2f}%')

        if accuracy > BestACC:
            BestACC = accuracy

            torch.save(model.state_dict(), 'C:/Users/saaba/Desktop/COMP472-Project/main_model.pt')

        print(f'Best accuracy on the test set: {BestACC:.2f}%')

        model.train()
