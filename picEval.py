import torch
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

#DEPENDING ON THE MODEL YOU USE COMMENT OUT THE 3 OTHER IMPORTS
from kFold import MultiLayerFCNet
# from Ass2_mainModel import MultiLayerFCNet
# from Ass2_modelVar1 import MultiLayerFCNet
# from Ass2_modelVar2 import MultiLayerFCNet

def load_model(model_path):
    # Initialize the model
    model = MultiLayerFCNet()

    # Load the state dictionary from the file
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if the model was trained using nn.DataParallel, which prepends 'module.' to all parameters keys.
    # If so, we need to remove 'module.' from the keys since we are not using nn.DataParallel for inference.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove `module.` prefix.
        new_state_dict[name] = v

    # Load the adjusted state dictionary
    model.load_state_dict(new_state_dict)

    # Set the model to evaluation mode
    model.eval()

    return model


def predict_image(model, image_path):
    # Define the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    # Add batch dimension (B x C x H x W)
    image = image.unsqueeze(0)

    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(image)
        # Get the predicted class with the highest score
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


if __name__ == "__main__":

    # Path to the trained model (CHOOSE THE MODEL YOU WANT TO TEST & COMMENT OUT THE 3 OTHERS)
    model_path = 'checkpoint/best_model.pt'
    # model_path = 'main_model.pt'
    # model_path = 'modelVar1.pt'
    # model_path = 'modelVar2.pt'

    # Path to the image you want to predict
    image_path = 'Happy/ffhq_108.png'

    # Load the trained model
    model = load_model(model_path)

    # Predict the image
    predicted_label = predict_image(model, image_path)

    # Dictionary to map numeric labels to class names
    label_to_class = {
        0: 'Neutral',
        1: 'Surprised',
        2: 'Happy',
        3: 'Focused'
    }

    # Get the predicted class name
    predicted_class_name = label_to_class[predicted_label]

    # Print the predicted class name
    print(f'Predicted class: {predicted_class_name}')
