from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from Ass2_modelVar2 import MultiLayerFCNet, Pclass  # Ensure correct import paths
import torch.nn as nn


def run_eval():

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset for testing
    testset = Pclass('test')
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    model = MultiLayerFCNet().to(device)
    model = nn.DataParallel(model)
    # Load the model with given parameters
    model.load_state_dict(torch.load('modelVar2.pt', map_location=device))
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

    precision_micro = precision_score(true_labels, predicted_labels, average='micro')
    recall_micro = recall_score(true_labels, predicted_labels, average='micro')
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    print(f"Precision Micro: {precision_micro}")
    print(f"Recall Micro: {recall_micro}")
    print(f"F1 Score Micro: {f1_micro}")

    # Compute and plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Neutral', 'Surprised', 'Happy', 'Focused'],
                yticklabels=['Neutral', 'Surprised', 'Happy', 'Focused'],
                annot_kws={"size":12, "weight":"bold", "color":"black"})
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    run_eval()
