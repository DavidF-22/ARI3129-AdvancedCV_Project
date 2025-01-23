import os
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional


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

# function to preprocess image
def preprocess_image(img_path, device):
    # Open image
    img = Image.open(img_path).convert('RGB')
    
    # Convert image to tensor and add batch dimension
    img_tensor = functional.to_tensor(img).unsqueeze(0)
    
    # move image to device and return it
    return img_tensor.to(device)

# Get class name
def get_class_name(class_id, COCO_CLASSES):
    return COCO_CLASSES.get(class_id, 'Unknown') # return 'Unknown' if class_id not found


# * ######################################################################################################################


# Draw bounding box with correct class name and increase image size
def draw_bboxes(output_dir, image, prediction, fig_size, COCO_CLASSES, saved_images_counter, total_images):
    boxes = prediction[0]['boxes'].cpu().numpy() # get predicted bounding boxes
    labels = prediction[0]['labels'].cpu().numpy() # get predicted labels
    scores = prediction[0]['scores'].cpu().numpy() # get predicted scores
    
    # Set a threshold for showing bounding boxes
    threshold = 0.5
    
    # Create a figure and axes using subplots
    fig, ax = plt.subplots(figsize=fig_size)

    # Display the image
    ax.imshow(image)
    
    # Draw bboxes
    for box, label, score in zip(boxes, labels, scores):
        # check is score is above threshold
        if score > threshold:
            # Draw bbox
            x_min, y_min, x_max, y_max = box
            # Get class name
            class_name = get_class_name(label, COCO_CLASSES)
            
            # Draw bbox
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min, 
                    fill=False, 
                    edgecolor='red', 
                    linewidth=2
                )
            )
            # Add class name
            ax.text(
                x_min, y_min, 
                f'{class_name} ({score:.3f})', 
                color='blue', 
                fontsize=10,
            )
            
    # Turn off axis
    ax.axis('off')
    # Saving plt using Image
    fig.savefig(f'{output_dir}/{img}', bbox_inches='tight', pad_inches=0)
    
    # ouput saved image number
    if (saved_images_counter + 1) == (total_images - 1):
        print(f"Saved image {saved_images_counter + 1}/{total_images - 1}")
    else:
        print(f"Saved image {saved_images_counter + 1}/{total_images - 1}", end='\r')


# * ###################################################################################################################### 


# Main Pipeline
if __name__ == '__main__':
    # Figure size
    fig_size = (8, 8)
    # Num of classes
    numOfClasses = 5 # Trash, Mixed Waste -Black Bag-, Organic Waste -White Bag-, Other, Recycled Waste -Grey or Green Bag-
    # COCO classes - 5 classes
    COCO_CLASSES = {0:'Trash', 1:'Mixed Waste -Black Bag-', 2:'Organic Waste -White Bag-', 3:'Other', 4:'Recycled Waste -Grey or Green Bag-'}
    # Get testing directory
    testing_dir = './trashy-dataset-roboflow.coco/test'
    # Output directory
    output_dir = './imagesOut'
    # saved images counter
    saved_images_counter = 0
    
    # * ######################################################################################################################

    # Move model to device if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nGPU: {torch.cuda.get_device_name(0)} is available - moving model to GPU\n")
    else:
        device = torch.device('cpu')
        print("\nNo GPU available. Moving training to CPU\n")

    # Initialise the Model

    # Load trained model with weights_only=True
    model = getModel(numOfClasses)
    # Load the state dictionary (safe way)
    state_dict = torch.load('./modelOut/model_epoch15.pth', weights_only=True)
    model.load_state_dict(state_dict)
    # Move model to the appropriate device
    model.to(device)
    # Set model to evaluation mode
    model.eval()

    # * ######################################################################################################################

    # Create output images directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all test images in the directory
    for img in os.listdir(testing_dir):
        # If file is not _annotations.coco.json
        if not img == '_annotations.coco.json':
            # Get full image path
            img_path = os.path.join(testing_dir, img)
            # Convert image to tensor
            image_tensor = preprocess_image(img_path, device)

            # Disable gradient computation
            with torch.no_grad():
                # Get prediction
                prediction = model(image_tensor)   
        
            # Display image with bounding boxes
            draw_bboxes(output_dir, Image.open(img_path), prediction, fig_size, COCO_CLASSES, saved_images_counter, total_images=len(os.listdir(testing_dir)))
            # Increment saved images counter
            saved_images_counter += 1
            
            # Close all figures
            plt.close('all')
            
    print("\n----- <Testing Completed> ----- ")