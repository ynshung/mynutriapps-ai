from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


class FoodProductCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FoodProductCNN, self).__init__()

        # Use a pre-trained ResNet18 model with explicit weights
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze early layers to prevent overfitting
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False

        # Modify the final fully connected layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load("model/resnet18.pth", map_location=device)
    model = FoodProductCNN(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint["class_names"]


model, class_names = load_model()


def predict_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]

    class_idx = predicted.item()
    class_name = class_names[class_idx]
    confidence = prob[class_idx].item()

    return {"class_name": class_name, "confidence": confidence}
