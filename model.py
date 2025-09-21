
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F


def load_classes(path="classes.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path="resnet_18.pth", num_classes=35):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(model, image, class_names,threshold=0.6):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

        if confidence.item() < threshold:
            return None, confidence.item()  # Not confident

        return class_names[predicted.item()], confidence.item()