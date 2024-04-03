import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Assignment2 import MultiLayerFCNet, Pclass
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

input_size = 3 * 96 * 96  # 3 channels, 96x96 image size
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_single_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # The resize should match the preprocessing in training
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities).item()
    class_probability = probabilities[0][predicted_class].item()
    return predicted_class, class_probability



if __name__ == '__main__':
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('test.pt', map_location=device), strict=False)
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode

    # Call the function with the model and the device as arguments
    predicted_class, class_probability = predict_single_image('Happy/ffhq_0.png', model, device)
    print(f'Predicted class: {predicted_class} with probability {class_probability}')
    #'Neutral': 0, 'Surprised': 1, 'Happy': 2, 'Focused': 3


