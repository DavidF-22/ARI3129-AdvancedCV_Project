# imports
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional

# * ######################################################################################################################

# convert image/s to tensor
class CocoToTensor:
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = functional.to_tensor(image)
        
        # return image and target - target mean the class name and bounding box
        return image, target

# Load the COCO dataset
def get_dataset(img_dir, ann_file):
    # Load the COCO dataset
    return CocoDetection(
        root=img_dir, 
        annFile=ann_file, 
        transforms=CocoToTensor()
    )

print(f"\n----- <Loading Training And Validation Datasets> -----")
# Load training and validation data
training_data = get_dataset(
    img_dir='./trashy-dataset-roboflow.coco/train', 
    ann_file='./trashy-dataset-roboflow.coco/train/_annotations.coco.json'
)
print()
validation_data = get_dataset(
    img_dir='./trashy-dataset-roboflow.coco/valid', 
    ann_file='./trashy-dataset-roboflow.coco/valid/_annotations.coco.json'
)
print(f"----- <Datasets Loaded Successfully> -----")

# Create two respective dataloaders
print(f"\n----- <Creating Training And Validation DataLoaders> -----")
training_dataloader = DataLoader(training_data, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
validation_dataloader = DataLoader(validation_data, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print(f"----- <DataLoaders Created Successfully> -----")

# * ######################################################################################################################

# function to load Faster R-CNN with ResNet50 backend
def getModel(numOfClasses):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classifier
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, numOfClasses)
    
    # Return model
    return model

# Initialise the Model
numOfClasses = 5 # Background, Mixed Waste -Black Bag-, Organic Waste -White Bag-, Other, Recycled Waste -Grey or Green Bag-
model = getModel(numOfClasses)

# Move model to device if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\nGPU: {torch.cuda.get_device_name(0)} is available - moving model to GPU")
else:
    device = torch.device('cpu')
    print("No GPU available. Moving training to CPU.")

# move model to device
model.to(device)

# Define the optimizer and hyperparameters
parameters = [p for p in model.parameters() if p.requires_grad]
optimiser = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
learningRate_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3, gamma=0.1)

# Training
def trainEpoch(model, optimizer, data_loader, device, epoch):
    # Set model to training mode
    model.train()
    
    # Iterate over the data
    for batch_idx, (images, targets) in enumerate(data_loader):
        print(f"Processing Batch {batch_idx + 1}/{len(data_loader)}")

        # Move images to device
        images = [img.to(device) for img in images]
        
        # Validate and process targets
        processed_targets = []
        valid_images = []
        
        # Iterate over targets
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            
            for obj in target:
                # Extract bounding box coordinates
                bbox = obj['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Validate width and height are positive
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
                    labels.append(obj['category_id'])
                    
            # If valid boxes
            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),  # Corrected key
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device)
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])
        
        # Skip iteration if no valid targets
        if not processed_targets:
            print(f"Batch {batch_idx + 1}: No valid targets, skipping.")
            continue
        
        # Ensure alignment of images and targets
        images = valid_images
        
        # Forward pass
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    print(f"\nEpoch [{epoch + 1}] Loss: {losses.item():.3f}")

# Training loop
num_epochs = 15

print(f"\n----- <Training Model> -----")
for epoch in range(num_epochs):
    print(f"\n----- <Starting Epoch {epoch + 1}/{num_epochs}> -----")
    trainEpoch(model, optimiser, training_dataloader, device, epoch)
    learningRate_scheduler.step()
    
    # Create output directory if it doesn't exist
    outputFolder = './modelOut'
    
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    
    # Save model's state dictionary after each epoch
    model_path = f"{outputFolder}/model_epoch{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nEpoch {epoch + 1}: Model saved at {model_path}")
    
print(f"\n----- <Model Training Completed> -----")
