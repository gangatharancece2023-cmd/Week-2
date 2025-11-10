import torch
from torch import nn, optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

DATA_DIR = "data/processed"
MODEL_SAVE = "models/fabric_classifier.pt"

transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_SAVE)
print("✅ Model saved to:", MODEL_SAVE)
