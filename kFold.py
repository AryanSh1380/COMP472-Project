from skorch.helper import SliceDataset
from torch import optim, nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import PIL.Image as Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
from skorch.callbacks import Checkpoint
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np

# Define the dataset class
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


# Define the neural network model
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

    # Create an instance of the dataset for the whole training data
    train_dataset = Pclass('train')
    # Wrap the PyTorch training dataset to make it compatible with scikit-learn
    train_sliceable = SliceDataset(train_dataset)

    # Retrieve targets for the training set
    y_train = []
    for _, target in DataLoader(train_dataset, batch_size=len(train_dataset)):
        y_train = target.numpy()
        break

    # Set device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the checkpoint directory where the best model will be saved
    checkpoint_dir = 'checkpoint/'

    # Ensure the directory exists, create if not
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the Checkpoint callback
    checkpoint = Checkpoint(
        dirname=checkpoint_dir,
        f_params='best_model.pt',
        monitor='valid_acc_best',  # determine the best model
        f_optimizer=None,
        f_criterion=None,
        event_name='validation_end',
    )

    # Initialize the Skorch wrapper for the neural network
    net = NeuralNetClassifier(
        module=MultiLayerFCNet,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        lr=0.001,
        iterator_train__shuffle=True,
        batch_size=32,
        max_epochs=10,
        device=device,
        callbacks=[checkpoint]
    )

    # Define the K-Fold cross-validator This line creates a Stratified K-Fold cross-validator, a technique for
    # splitting the dataset into k folds while maintaining the class distribution in each fold. - shuffle=True:
    # Indicates whether to shuffle the data before splitting it into folds. Shuffling helps ensure that each fold
    # contains a random sample of the dataset, reducing bias.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Define the scoring dictionary
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'precision_micro': make_scorer(precision_score, average='micro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'recall_micro': make_scorer(recall_score, average='micro'),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_micro': make_scorer(f1_score, average='micro')
    }

    # Perform k-fold cross-validation and return scores
    print("Starting k-fold cross-validation...")
    cv_results = cross_validate(
        estimator=net,
        X=train_sliceable,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    print("Cross-validation completed.")

    # Document the results with macro and micro scores
    print("Cross-validation scores:")
    for i in range(cv.get_n_splits()):
        print(f"Fold {i + 1}:")
        print(f"Accuracy: {cv_results['test_accuracy'][i]}")
        print(f"Precision Macro: {cv_results['test_precision_macro'][i]}")
        print(f"Precision Micro: {cv_results['test_precision_micro'][i]}")
        print(f"Recall Macro: {cv_results['test_recall_macro'][i]}")
        print(f"Recall Micro: {cv_results['test_recall_micro'][i]}")
        print(f"F1-score Macro: {cv_results['test_f1_macro'][i]}")
        print(f"F1-score Micro: {cv_results['test_f1_micro'][i]}\n")

    # Calculate and print the average of the metrics with macro and micro scores
    print("Average Scores:")
    print("Average Accuracy:", np.mean(cv_results['test_accuracy']))
    print("Average Precision Macro:", np.mean(cv_results['test_precision_macro']))
    print("Average Precision Micro:", np.mean(cv_results['test_precision_micro']))
    print("Average Recall Macro:", np.mean(cv_results['test_recall_macro']))
    print("Average Recall Micro:", np.mean(cv_results['test_recall_micro']))
    print("Average F1-score Macro:", np.mean(cv_results['test_f1_macro']))
    print("Average F1-score Micro:", np.mean(cv_results['test_f1_micro']))



    # # Perform k-fold cross-validation
    # cv_scores = cross_val_score(
    #     estimator=net,
    #     X=train_sliceable,
    #     y=y_train,
    #     cv=5,  # 5-fold cross-validation
    #     scoring='accuracy',
    #     n_jobs=-1  # Use all available CPU cores
    # )
    #
    # print('Cross-validation scores:', cv_scores)
