import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from effdet import create_model, DetBenchTrain
from pycocotools.coco import COCO

# Dataset class
class CustomCOCODataset(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.annotation_path = os.path.join(root, annotation)
        self.coco = COCO(self.annotation_path)
        self.ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0  # Filter out images with no annotations
        ]
        self.cat_id_to_class_idx = {cat_id: idx for idx, cat_id in enumerate(self.coco.getCatIds())}

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Prepare bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [xmin, ymin, w, h]
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_id_to_class_idx[ann['category_id']] + 1)  # 1-based indexing

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

    def __len__(self):
        return len(self.ids)

# Dataset and DataLoader
train_dataset = CustomCOCODataset(root='trashy_dataset.coco/train', annotation='_annotations.coco.json')
val_dataset = CustomCOCODataset(root='trashy_dataset.coco/valid', annotation='_annotations.coco.json')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model setup
num_classes = 4  # Adjusted to match the number of unique classes
model = create_model('tf_efficientdet_d1', pretrained=True, num_classes=num_classes)
model = DetBenchTrain(model)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training function
def train_one_epoch(loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)  # Pass raw targets
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # Pass raw targets
            total_loss += loss_dict['loss'].item()

    return total_loss / len(loader)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 20
for epoch in range(epochs):
    train_loss = train_one_epoch(train_loader, model, optimizer, device)
    val_loss = validate(val_loader, model, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'efficientdet_coco.pth')
