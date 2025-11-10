import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/processed"
MODEL_PATH = "models/fabric_classifier.pt"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=len(dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

y_true, y_pred = [], []

for x, y in dataset:
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x)).item()
    y_true.append(y)
    y_pred.append(pred)

print(classification_report(y_true, y_pred, target_names=dataset.classes))
print(confusion_matrix(y_true, y_pred))
