# imports
import os
import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix


# * ######################################################################################################################


# Class Mapping
CLASS_MAPPING = {
    "Background": 0,
    "Mixed Waste -Black Bag-": 1,
    "Organic Waste -White Bag-": 2,
    "Other": 3,
    "Recycled Waste -Grey or Green Bag-": 4
}


# * ######################################################################################################################


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

# Function to preprocess image
def preprocess_image(img_path, device):
    # Open image
    img = Image.open(img_path).convert('RGB')
    
    # Convert image to tensor and add batch dimension
    img_tensor = functional.to_tensor(img).unsqueeze(0)
    
    # move image to device and return it
    return img_tensor.to(device)

# Function to load ground truth from XML files in dataset directory
def load_ground_truth(dataset_dir, CLASS_MAPPING):
    ground_truth = {}
    
    # Iterate through all file in dataset directory
    for file in os.listdir(dataset_dir):
        # Check if file is a XML (annotation) file
        if file.endswith('.xml'):
            # Pase filr and extract structure
            tree = ET.parse(os.path.join(dataset_dir, file))
            root = tree.getroot()  # Get the root element of the XML tree

            # Empty list for labels
            labels = []
            
            # Iterate through all objects in the XML file
            for obj in root.findall('object'):
                # Extract class name
                class_name = obj.find('name').text
                
                # Map class name to class ID and append to labels
                if class_name in CLASS_MAPPING:
                    labels.append(CLASS_MAPPING[class_name])
                else:
                    # Error
                    print(f"Warning: Unknown class '{class_name}' in {file}")
                    
            # Extract image name
            image_name = root.find('filename').text
            # Add labels to ground truth dictionary
            ground_truth[image_name] = labels
    
    # return ground truth dictionary
    return ground_truth


# * ######################################################################################################################


# Function to plot precision-recall curve and calculate AUC
def plot_precision_recall(y_true, y_scores, num_classes, output_dir, CLASS_MAPPING):
    # Output progress
    print("Plotting Precision-Recall Curve and Calculating AUC", end='\r')
    
    # Convert dictionary keys to a list for consistent indexing
    class_names = list(CLASS_MAPPING.keys())

    # Set up the figure size for the plot
    plt.figure(figsize=(10, 8))

    # Initialize variables for micro-average
    y_true_flat = y_true.ravel()
    y_scores_flat = y_scores.ravel()

    # Compute the micro-average Precision-Recall curve
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_flat, y_scores_flat)

    # Calculate the micro-average AUC
    micro_auc = auc(recall_micro, precision_micro)

    # Plot the micro-average curve
    plt.plot(recall_micro, precision_micro, linestyle='--', color='mediumblue', linewidth=2.5,
             label=f"Overall Micro-average (AUC = {micro_auc:.3f})")

    # Loop through each class and compute its precision-recall curve
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        # Check if there are any positive examples for the class
        if np.sum(y_true[:, class_id]) == 0:
            print(f"Warning: No positive samples found for class '{class_name}'. Skipping.")
            continue

        # Compute precision, recall, and thresholds for the class
        precision, recall, _ = precision_recall_curve(y_true[:, class_id], y_scores[:, class_id])

        # Calculate the area under the precision-recall curve (AUC)
        pr_auc = auc(recall, precision)

        # Plot the precision-recall curve for the class, including the AUC in the label
        plt.plot(recall, precision, label=f"{class_name} (AUC = {pr_auc:.3f})")

    # Label the axes
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RetinaNet - Precision-Recall Curve")
    # Add a legend to identify the curves by class
    plt.legend(loc="best")
    # Add a grid for better readability
    plt.grid()
    # Save the plot as an image file
    plt.savefig(f"{output_dir}/precision_recall_curve_with_auc.png")
    # Close the plot to free up memory
    plt.close('all')
    
    print("Plotting Precision-Recall Curve and Calculating AUC | Done")

# Function to plot confusion matrix
def plot_multiclass_confusion_matrix(y_true, y_pred, num_classes, output_dir, CLASS_MAPPING, normalize=False):
    # Output progress
    print("Plotting Confusion Matrix", end='\r')
    
    # Reverse CLASS_MAPPING to get class labels
    COCO_CLASSES = {v: k for k, v in CLASS_MAPPING.items()}

    # Compute the confusion matrix
    cm = confusion_matrix(
        [COCO_CLASSES[label] for label in y_true.argmax(axis=1)],
        [COCO_CLASSES[label] for label in y_pred.argmax(axis=1)],
        labels=list(CLASS_MAPPING.keys())
    )

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with 0 if division by 0 occurs

    plt.figure(figsize=(14, 10))

    # Create a heatmap for the confusion matrix
    heatmap = sns.heatmap(cm, 
                          annot=True, 
                          fmt='.2f' if normalize else 'd', 
                          cmap='Blues', 
                          xticklabels=list(CLASS_MAPPING.keys()), 
                          yticklabels=list(CLASS_MAPPING.keys()),
                          annot_kws={"size": 14})  # Annotation font size

    # Add titles and axis labels
    plt.title("RetinaNet - Confusion Matrix", fontsize=18)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    # Add a color bar label
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("Count" if not normalize else "Proportion", fontsize=12)
    # Automatically adjust the layout to avoid truncation
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "confusion_matrix_multiclass.png"))
    plt.close('all')
    
    print("Plotting Confusion Matrix | Done")


# * ######################################################################################################################


# Function to draw bounding boxes on images
def draw_bboxes(output_dir, image, image_name, prediction, fig_size, CLASS_MAPPING, saved_images_counter, total_images):
    boxes = prediction[0]['boxes'].cpu().numpy() # get predicted bounding boxes
    labels = prediction[0]['labels'].cpu().numpy() # get predicted labels
    scores = prediction[0]['scores'].cpu().numpy() # get predicted scores

    # Set a threshold for showing bounding boxes
    threshold = 0.35

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image)

    # Draw bboxes
    for box, label, score in zip(boxes, labels, scores):
        # If score is below threshold, ignore
        if score > threshold:
            # Get box coordinates
            x_min, y_min, x_max, y_max = box
            # Get class name from mapping - Switching from IDs to class names
            class_name = CLASS_MAPPING.get(label)

            # Draw bbox
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    fill=False, edgecolor='red', linewidth=2
                )
            )
            
            # Add class name and confidence score
            ax.text(
                x_min, y_min,
                f'{class_name} ({score:.3f})',
                color='blue',
                fontsize=10,
            )
    
    # Remove axis
    ax.axis('off')
    # Save image
    fig.savefig(f'{output_dir}/{image_name}.png', bbox_inches='tight', pad_inches=0)
    
    # Display progress
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
    testing_dir = './trashy-dataset-roboflow.voc/test'  # Get testing directory
    output_dir_images = './images'                      # Output directories for images
    output_dir_plots = './plots'                        # Output directories for plots
    saved_model_dir = './RetinaNet-Weights'             # Saved model directory
    # saved images counter
    saved_images_counter = 0
    
    # Initialize lists for true labels and predicted scores
    y_true_list = []
    y_scores_list = []
    
    # * ##################################################################################################################
    
    # Load Model and Move to Device
    print("\n----- <Loading Model> -----")
    model_path = os.path.join(saved_model_dir, os.listdir(saved_model_dir)[0])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)} is available - moving model to GPU")
    else:
        device = torch.device('cpu')
        print("No GPU available. Moving testing to CPU")

    model = get_retinanet_model(numOfClasses)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"----- <Model [{os.listdir(saved_model_dir)[0]}] Loaded and Moved Successfully> -----\n")

    # * ##################################################################################################################
    
    print("----- <Testing Model> -----\n")
    
    # Create output image directory
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Get all images in testing directory
    test_images = [img for img in os.listdir(testing_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Load ground truth
    ground_truth = load_ground_truth(testing_dir, CLASS_MAPPING)

    # Iterate through all images
    for image in test_images:
        # Get ground truth labels for the image
        true_labels = ground_truth.get(image, [0])  # Default to background if no labels
        
        # Get image path
        img_path = os.path.join(testing_dir, image)
        # Preprocess image
        image_tensor = preprocess_image(img_path, device)
        
        # Convert to one-hot encoding for precision-recall computation
        true_one_hot = np.zeros(numOfClasses)
        
        for label in true_labels:
            true_one_hot[label] = 1

        y_true_list.append(true_one_hot)

        # Disable gradient computation for faster inference
        with torch.no_grad():
            # Get model prediction
            prediction = model(image_tensor)
            
            # Collect predicted scores
            pred_scores = np.zeros(numOfClasses)
            
            for label, score in zip(prediction[0]['labels'].cpu().numpy(), prediction[0]['scores'].cpu().numpy()):
                pred_scores[label] = max(pred_scores[label], score)

            # Append the predicted scores to the y_scores_list for later metric computation
            y_scores_list.append(pred_scores)

        # Draw bounding boxes on image and save result
        draw_bboxes(
            output_dir_images, 
            Image.open(img_path),
            os.path.splitext(image)[0], 
            prediction, 
            fig_size, 
            CLASS_MAPPING, 
            saved_images_counter, 
            total_images=len(test_images)
        )
        
        # Increment saved images counter
        saved_images_counter += 1

        # Closing all figures to free up memory
        plt.close('all')
        
    print(f"Results Saved in {output_dir_images} Folder\n")
        
    # * ##################################################################################################################
    
    # Add Output Directory for Plots
    if not os.path.exists(output_dir_plots):
        os.makedirs(output_dir_plots)
    
    # Convert scores and predictions to NumPy arrays
    y_true_np = np.array(y_true_list)
    y_scores_np = np.array(y_scores_list)
    
    # Plot Precision-Recall Curve and AUC for each class
    plot_precision_recall(y_true_np, y_scores_np, numOfClasses, output_dir_plots, CLASS_MAPPING)
    # Plot Cconfusion Matrix
    plot_multiclass_confusion_matrix(y_true_np, y_scores_np, numOfClasses, output_dir_plots, CLASS_MAPPING, normalize=True)
    
    # Display completion message
    print("\n----- <Testing Completed Successfully> -----\n")