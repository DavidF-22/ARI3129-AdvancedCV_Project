import os

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional
from PIL import Image

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix


# * ######################################################################################################################


# Function to load Faster R-CNN with ResNet50 backend
def getModel(numOfClasses):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classifier
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, numOfClasses)
    
    # Return model
    return model

# Function to preprocess image
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


# Function to load true labels from COCO annotations
def load_coco_annotations(annotation_file, image_files):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create a mapping of image ID to its annotations
    image_id_to_annotations = {image['id']: [] for image in annotations['images']}
    for annotation in annotations['annotations']:
        image_id_to_annotations[annotation['image_id']].append(annotation['category_id'])

    # Map file names to true labels
    file_name_to_labels = {}
    for image in annotations['images']:
        file_name = image['file_name']
        image_id = image['id']
        true_labels = image_id_to_annotations[image_id]
        file_name_to_labels[file_name] = true_labels

    # Filter only the labels for images in the test set
    true_labels = {img: file_name_to_labels[img] for img in image_files if img in file_name_to_labels}
    return true_labels

# Function to plot precision-recall curve and calculate AUC
def plot_precision_recall(y_true, y_scores, num_classes, output_dir, COCO_CLASSES):
    # Set up the figure size for the plot
    plt.figure(figsize=(10, 8))

    # Loop through each class and compute its precision-recall curve
    for class_id in range(num_classes):
        # Check if there are any positive examples for the class
        if np.sum(y_true[:, class_id]) == 0:
            print(f"Warning: No positive samples found for class '{COCO_CLASSES[class_id]}'. Skipping.")
            continue

        # Compute precision, recall, and thresholds for the class
        precision, recall, _ = precision_recall_curve(y_true[:, class_id], y_scores[:, class_id])

        # Calculate the area under the precision-recall curve (AUC)
        pr_auc = auc(recall, precision)

        # Plot the precision-recall curve for the class, including the AUC in the label
        plt.plot(recall, precision, label=f"{COCO_CLASSES[class_id]} (AUC = {pr_auc:.2f})")

    # Label the axes
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Faster R-CNN - Precision-Recall Curve")
    # Add a legend to identify the curves by class
    plt.legend(loc="best")
    # Add a grid for better readability
    plt.grid()
    # Save the plot as an image file
    plt.savefig(f"{output_dir}/precision_recall_curve_with_auc.png")
    # Close the plot to free up memory
    plt.close('all')
    
# Function to plot a single confusion matrix with better formatting
def plot_multiclass_confusion_matrix(y_true, y_pred, num_classes, output_dir, COCO_CLASSES):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=range(num_classes))

    plt.figure(figsize=(14, 10))
    
    # Create a heatmap for the confusion matrix
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=COCO_CLASSES.values(), 
                          yticklabels=COCO_CLASSES.values(),
                          annot_kws={"size": 14})  # Annotation font size

    # Add titles and axis labels
    plt.title("Faster R-CNN - Confusion Matrix", fontsize=18)
    plt.xlabel("True Labels", fontsize=14)
    plt.ylabel("Predicted Labels", fontsize=14)
    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    # Add a color bar label
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("Count", fontsize=12)
    # Automatically adjust the layout to avoid truncation
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "confusion_matrix_multiclass.png"))
    plt.close('all')


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
    # Output directories
    output_dir_images = './images'
    output_dir_plots = './plots'
    # saved images counter
    saved_images_counter = 0
    
    # Initialize lists for ground truth and predictions
    y_true_list = []
    y_scores_list = []
    
    # * ######################################################################################################################
    
    # Path to the directory containing the saved model
    saved_model_dir = './Faster_R-CNN - Saved_Model'
    # Get the path to the only file in the directory
    model_path = os.path.join(saved_model_dir, os.listdir(saved_model_dir)[0])


    # Move model to device if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nGPU: {torch.cuda.get_device_name(0)} is available - moving model to GPU\n")
    else:
        device = torch.device('cpu')
        print("\nNo GPU available. Moving training to CPU\n")

    # Initialise the Model
    print("----- <Loading Model> -----")
    # Load trained model with weights_only=True
    model = getModel(numOfClasses)
    # Load the state dictionary (safe way)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    # Move model to the appropriate device
    model.to(device)
    # Set model to evaluation mode
    model.eval()
    print(f"----- <Model [{os.listdir(saved_model_dir)[0]}] Loaded Successfully> -----\n")

    # * ######################################################################################################################

    # Create output directories
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    
    # Get all test images
    test_images = [img for img in os.listdir(testing_dir) if img != '_annotations.coco.json']

    # Load true labels from annotations
    true_labels = load_coco_annotations("./trashy-dataset-roboflow.coco/test/_annotations.coco.json", test_images)
        
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
                
            true_label_indices = true_labels.get(img, [])
            true_one_hot = np.zeros((1, numOfClasses))
            
            for idx in true_label_indices:
                true_one_hot[0, idx] = 1

            pred_scores = np.zeros(numOfClasses)
            
            for label, score in zip(prediction[0]['labels'].cpu().numpy(), prediction[0]['scores'].cpu().numpy()):
                pred_scores[label] = max(pred_scores[label], score)

            y_true_list.append(true_one_hot)
            y_scores_list.append(pred_scores)
        
            # Display image with bounding boxes
            draw_bboxes(output_dir_images, Image.open(img_path), prediction, fig_size, COCO_CLASSES, saved_images_counter, total_images=len(os.listdir(testing_dir)))
            # Increment saved images counter
            saved_images_counter += 1
            
            # Closing all figures to free up memory
            plt.close('all')
    
    # * ######################################################################################################################
    
    # Create output directory for plots
    if not os.path.exists(output_dir_plots):
        os.makedirs(output_dir_plots)
    
    # Stack the true labels and predictions
    y_true = np.vstack(y_true_list)
    y_scores = np.vstack(y_scores_list)
    
    # Remove the "Trash" class from the variables
    trash_index = 0  # Index of the "Trash" class

    # Remove the Trash column from y_true and y_scores
    y_true_filtered = np.delete(y_true, trash_index, axis=1)
    y_scores_filtered = np.delete(y_scores, trash_index, axis=1)

    # Remove the "Trash" class from COCO_CLASSES
    COCO_CLASSES_FILTERED = {k - 1: v for k, v in COCO_CLASSES.items() if k != trash_index}
    
    # Adjust the number of classes
    num_classes_filtered = numOfClasses - 1

    # Plot Precision-Recall curve
    plot_precision_recall(y_true_filtered, y_scores_filtered, num_classes_filtered, output_dir_plots, COCO_CLASSES_FILTERED)
    # Plot Confusion Matrix
    plot_multiclass_confusion_matrix(y_true_filtered, y_scores_filtered, num_classes_filtered, output_dir_plots, COCO_CLASSES_FILTERED)
    
    # Completion Message
    print("\n----- <Testing Completed Successfully> ----- ")