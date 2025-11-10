import os
from PIL import Image
import torchvision.transforms as T

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
])

os.makedirs(PROC_DIR, exist_ok=True)

for cls in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, cls)
    save_path = os.path.join(PROC_DIR, cls)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        try:
            img = Image.open(os.path.join(class_path, img_name)).convert("RGB")
            img = transform(img)
            img.save(os.path.join(save_path, img_name))
        except:
            print("Skipped:", img_name)
          
