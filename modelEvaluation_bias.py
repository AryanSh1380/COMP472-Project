from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from Ass2_mainModel import MultiLayerFCNet, Pclass  # Ensure correct import paths
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import PIL.Image as Image
import os
from torchvision import transforms


class PclassBias(Dataset):

    def __init__(self, mode, age_or_gender, choice):
        # Load the CSV file into a DataFrame
        csv_file = 'Labeled Data.csv'
        self.mode = mode
        data_frame = pd.read_csv(csv_file)

        if age_or_gender == 'age':
            if choice == 'young':
                self.data_frame = data_frame[(data_frame['age'] == 'Young')]
            elif choice == 'adult':
                self.data_frame = data_frame[(data_frame['age'] == 'Adult')]
            elif choice == "senior":
                self.data_frame = data_frame[(data_frame['age'] == 'Senior')]
            else:
                raise ValueError("For age, must be either 'young', 'adult', 'senior'.")

        elif age_or_gender == 'gender':
            if choice == 'male':
                self.data_frame = data_frame[(data_frame['gender'] == 'Male')]
            elif choice == 'female':
                self.data_frame = data_frame[(data_frame['gender'] == 'Female')]
            else:
                raise ValueError("For gender, must be either 'male', 'female'.")

        else:
            raise ValueError("Incorrect values entered.")

        train, evaluate = train_test_split(self.data_frame, test_size=0.15, random_state=42)
        if mode == 'train':
            self.data_frame = train
        elif mode == 'eval':
            self.data_frame = evaluate
        else:
            raise ValueError("Must be either 'train' or 'eval'.")

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


def run_eval():
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset for testing
    testset = PclassBias('eval', 'age', 'young')
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    model = MultiLayerFCNet().to(device)
    model = nn.DataParallel(model)
    # Load the model with given parameters
    model.load_state_dict(torch.load('main_model_bias.pt', map_location=device))
    # Set model to evaluation mode
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Compute metrics(macro and micro)
    acc = accuracy_score(true_labels, predicted_labels)

    print(f'Accuracy: {acc}')
    precision_macro = precision_score(true_labels, predicted_labels, average='macro')
    recall_macro = recall_score(true_labels, predicted_labels, average='macro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    print(f"Precision Macro: {precision_macro}")
    print(f"Recall Macro: {recall_macro}")
    print(f"F1 Score Macro: {f1_macro}")

    # precision_micro = precision_score(true_labels, predicted_labels, average='micro')
    # recall_micro = recall_score(true_labels, predicted_labels, average='micro')
    # f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    # print(f"Precision Micro: {precision_micro}")
    # print(f"Recall Micro: {recall_micro}")
    # print(f"F1 Score Micro: {f1_micro}")

    # Compute and plot confusion matrix
    # cm = confusion_matrix(true_labels, predicted_labels)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
    #             xticklabels=['Neutral', 'Surprised', 'Happy', 'Focused'],
    #             yticklabels=['Neutral', 'Surprised', 'Happy', 'Focused'],
    #             annot_kws={"size": 12, "weight": "bold", "color": "black"})
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()


if __name__ == '__main__':
    run_eval()
