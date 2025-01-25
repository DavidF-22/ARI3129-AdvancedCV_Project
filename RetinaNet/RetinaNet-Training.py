# imports
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional
import xml.etree.ElementTree as ET


# * ##############################################################################################################


# Class Mapping
CLASS_MAPPING = {
    "Background": 0,
    "Mixed Waste -Black Bag-": 1,
    "Organic Waste -White Bag-": 2,
    "Other": 3,
    "Recycled Waste -Grey or Green Bag-": 4
}


# * ##############################################################################################################


# Function to get RetinaNet model
def get_retinanet_model(num_classes):
    # Load pre-trained RetinaNet model
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

    # Update the number of classes in the classification head
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Update classification head
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
    )
    
    # Update number of classes
    model.head.classification_head.num_classes = num_classes
    
    # Return model
    return model

# Class to transform VOC XML annotations to PyTorch tensors
class VOCTransform:
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = functional.to_tensor(image)

        # Parse bounding boxes and labels from XML
        boxes = []
        labels = []
        
        # Extract bounding boxes and labels from XML (target)
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            # Validate bounding boxes
            if xmax <= xmin or ymax <= ymin:
                print(f"Invalid bounding box detected: {bbox}")
                continue
            
            # Append bounding boxes
            boxes.append([xmin, ymin, xmax, ymax])

            # Map class name to numerical ID
            class_name = obj['name'] # get class name
            
            # If class name is in the mapping
            if class_name in CLASS_MAPPING:
                # Append the numerical ID
                labels.append(CLASS_MAPPING[class_name])
            else:
                # Error
                raise ValueError(f"Unknown class name: {class_name}")

        # Validate non-empty targets
        if len(boxes) == 0:
            print("No valid boxes found in the annotation.")
            
        if len(labels) == 0:
            print("No valid labels found in the annotation.")

        # Convert target to PyTorch tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        
        # Return transformed image and target
        return image, target

# Class to load VOC dataset
class VOCDataset(VisionDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms=transforms)   # Initialise parent class
        self.images = sorted([f for f in os.listdir(root) if f.endswith(".jpg")])   # Get sorted list of all images in dataset
        self.annotations = sorted([f for f in os.listdir(root) if f.endswith(".xml")]) # Get sorted list of all annotations in dataset

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])    # Get image path
        annotation_path = os.path.join(self.root, self.annotations[index]) # Get annotation path

        # Load image
        image = functional.to_pil_image(torchvision.io.read_image(image_path)) # Read image and convert to PIL image

        # Load annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot() 

        # Parse annotation into dictionary
        target = {"annotation": {}}             # Empty dictionary to store annotation key
        target['annotation']['object'] = []     # Empty list to store objects
        
        # Iterate over all 'object' elements in the XML root
        for obj in root.findall('object'):
            
            # Create dictionary for each object with name and bounding box coordinates
            obj_struct = {
                "name": obj.find("name").text,
                "bndbox": {
                    "xmin": obj.find("bndbox/xmin").text,
                    "ymin": obj.find("bndbox/ymin").text,
                    "xmax": obj.find("bndbox/xmax").text,
                    "ymax": obj.find("bndbox/ymax").text,
                },
            }
            
            # Append object dictionary to the target dictionary
            target['annotation']['object'].append(obj_struct)

        # Apply transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Return image and target
        return image, target

    def __len__(self):
        # Return number of samples in dataset
        return len(self.images)


# * ##############################################################################################################


# Function to train one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # Set model to training mode
    model.train()
    # Initialize total loss
    total_loss = 0

    # Iterate over all batches and extract images and targets
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Output progress
        print(f"Epoch: {epoch + 1} | Batch: {batch_idx + 1}/{len(data_loader)}", end='\r')

        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Debugging loss incase of NaN values
        if torch.isnan(losses).any():
            print(f"NaN Detected in Loss at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            return
        
        # Update total loss
        total_loss += losses.item()

        # Perform backpropagation and optimizer step
        optimizer.zero_grad()   # Reset gradients for all model parameters to zero
        losses.backward()       # Compute gradients by performing backpropagation using the loss

        # Clip gradients to a maximum norm of 10.0 prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()        # Update model parameters using the computed gradients

    # Calculate average loss over all batches and display
    average_loss = total_loss / len(data_loader)
    print(f"Epoch: {epoch + 1} | Batch: {len(data_loader)}/{len(data_loader)} | Average Loss (over all batches in epoch): {average_loss:.3f}")


# * ##############################################################################################################


# Main Pipeline
if __name__ == "__main__":
    # Load Dataset and Create DataLoader
    print("----- <Loading Training Dataset> -----")
    train_dir = "./trashy-dataset-roboflow.voc/train"
    
    train_dataset = VOCDataset(train_dir, transforms=VOCTransform())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    print(f"----- <Training Dataset Loaded Successfully - {len(train_dataset)} Training Samples> -----\n")

    # Number of classes (including background)
    num_classes = 5  # Background, Mixed Waste, Organic Waste, Other, Recycled Waste
    
    # Get RetinaNet model
    print("----- <Loading RetinaNet Model Architecture> -----")
    model = get_retinanet_model(num_classes)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # output message that GPU is available and display the device name
        print(f"{torch.cuda.get_device_name(0)} Found - Moving to GPU")
    else:
        device = torch.device("cpu")
        # output message that GPU is not available
        print("GPU Not Found - Moving to CPU")
        
    # Move model to device
    model.to(device)
    print("----- <Model Loaded and Moved Successfully> -----\n")

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # Reduce learning rate by a factor of 0.1 every 3 epochs

    # Training loop
    print("----- <Training Model> -----")
    
    num_epochs = 100
    model_save_path = "./RetinaNet-Weights"
    
    # Iterate over all epochs
    for epoch in range(num_epochs):
        # Train epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update learning rate
        scheduler.step()

        # If last epoch
        if epoch + 1 == num_epochs:
            # Create directory if it doesn't exist
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
                
            # Save model
            torch.save(model.state_dict(), os.path.join(model_save_path, f"retinanet_epoch_{epoch + 1}.pth"))
            print(f"\nModel saved at epoch {epoch + 1} in {model_save_path}")
            
    # Completion message
    print("----- <Model Trained Successfully> -----\n") 