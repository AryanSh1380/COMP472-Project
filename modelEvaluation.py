import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from Assignment2 import MultiLayerFCNet, Pclass

def run_eval():

    input_size = 3 * 96 * 96  # 3 channels, 96x96 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data = Pclass('test')
    load = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=8, drop_last=True)

    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('model_epoc50_k3_11Layer.pt', map_location=device), strict=False)
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode


    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for input, labels in load:
            input = input.to(device) if torch.cuda.is_available() else input
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.cpu().tolist() if torch.cuda.is_available() else predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(np.array(true_labels), predicted_labels)
    print("Accuracy on test set: ", accuracy)

    precision = precision_score(np.array(true_labels), predicted_labels, average='macro', zero_division=1)
    print("Precision on test set: ", precision)

    recall = recall_score(np.array(true_labels), predicted_labels, average='macro', zero_division=1)
    print("Recall on test set: ", recall)

    f1_measure = f1_score(np.array(true_labels), predicted_labels, average='macro', zero_division=1)
    print("F1 on test set: ", f1_measure)

    class_names = {0: 'Neutral', 1: 'Surprised', 2: 'Happy', 3: 'Focused'}

    conf_matrix = confusion_matrix(np.array(true_labels), predicted_labels)
    for i in range(len(conf_matrix)):  # Assuming conf_matrix is square
        tn, fp, fn, tp = calculate_tn_fp_fn_tp(conf_matrix, i)
        class_name = class_names[i]
        print(f"'{class_name}' - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Plotting the confusion matrix
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(1, 5), yticklabels=range(1, 5))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

def calculate_tn_fp_fn_tp(conf_matrix, class_index):
    # True Positive (TP)
    tp = conf_matrix[class_index, class_index]
    # False Positive (FP): Sum of column for class_index excluding TP
    fp = conf_matrix[:, class_index].sum() - tp
    # False Negative (FN): Sum of row for class_index excluding TP
    fn = conf_matrix[class_index, :].sum() - tp
    # True Negative (TN): Sum of all elements excluding the row and column for class_index
    tn = conf_matrix.sum() - (fp + fn + tp)
    return tn, fp, fn, tp

if __name__ == '__main__':
    run_eval()
