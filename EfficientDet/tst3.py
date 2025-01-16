import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_hub as hub
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

CLASS_MAPPING = {
    "Mixed Waste -Black Bag-": 0,
    "Organic Waste -White Bag-": 1,
    "Other": 2,
    "Recycled Waste -Grey or Green Bag-": 3,
}

# Constants
NUM_CLASSES = 4
INPUT_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 10
MAX_BOXES = 20

# Dataset paths
TRAIN_DIR = "trashy-dataset.voc/train/"
VALID_DIR = "trashy-dataset.voc/valid/"

# Parse Pascal VOC annotations
def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes = []
    classes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = CLASS_MAPPING[class_name]  # Use class mapping
        bndbox = obj.find("bndbox")
        boxes.append([
            float(bndbox.find("xmin").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymax").text),
        ])
        classes.append(class_id)
    return filename, boxes, classes

# Preprocess dataset
def preprocess_dataset(data_dir):
    dataset = []
    for file in os.listdir(data_dir):
        if file.endswith(".xml"):
            annotation_path = os.path.join(data_dir, file)
            filename, boxes, classes = parse_voc_annotation(annotation_path)
            image_path = os.path.join(data_dir, filename)
            if os.path.exists(image_path):
                dataset.append((image_path, boxes, classes))
    return dataset

def preprocess_image_and_labels(image_path, boxes, classes):
    # Read and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SIZE)
    image = tf.cast(image, dtype=tf.uint8)

    # Pad boxes and classes
    num_boxes = len(boxes)
    padded_boxes = tf.pad(boxes, [[0, MAX_BOXES - num_boxes], [0, 0]], constant_values=-1)
    padded_classes = tf.pad(classes, [[0, MAX_BOXES - num_boxes]], constant_values=-1)

    return image, {"boxes": padded_boxes, "classes": padded_classes}

def data_generator(dataset):
    for image_path, boxes, classes in dataset:
        yield preprocess_image_and_labels(image_path, boxes, classes)

# Convert to tf.data.Dataset
def create_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        lambda: data_generator(dataset),
        output_signature=(
            tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=tf.uint8),
            {
                "boxes": tf.TensorSpec(shape=(MAX_BOXES, 4), dtype=tf.float32),
                "classes": tf.TensorSpec(shape=(MAX_BOXES,), dtype=tf.int32),
            },
        ),
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load datasets
print("Loading datasets...")
train_dataset = preprocess_dataset(TRAIN_DIR)
valid_dataset = preprocess_dataset(VALID_DIR)

print("Creating tf datasets...")
train_data = create_tf_dataset(train_dataset)
valid_data = create_tf_dataset(valid_dataset)

# Load pre-trained EfficientDet model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"

def build_model(model_url, num_classes):
    # Define input tensor
    input_tensor = layers.Input(shape=(None, None, 3), dtype=tf.uint8)

    # Load pre-trained EfficientDet from TensorFlow Hub
    efficientdet_layer = hub.KerasLayer(model_url, trainable=False)(input_tensor)

    # Add custom prediction layers
    detection_classes = layers.Dense(num_classes, activation="softmax", name="detection_classes")(efficientdet_layer["detection_classes"])
    detection_boxes = efficientdet_layer["detection_boxes"]
    detection_scores = efficientdet_layer["detection_scores"]

    # Create the final model
    model = Model(inputs=input_tensor, outputs={
        "detection_boxes": detection_boxes,
        "detection_classes": detection_classes,
        "detection_scores": detection_scores,
    })

    # Freeze all EfficientDet layers
    for layer in model.layers[:-3]:
        layer.trainable = False

    return model

model = build_model(MODEL_URL, NUM_CLASSES)

# Check trainable variables
trainable_vars = model.trainable_variables
print(f"Number of trainable variables: {len(trainable_vars)}")
for var in trainable_vars:
    print(var.name)

def custom_loss(y_true, y_pred):
    # You can use a combination of classification loss and regression loss here
    classification_loss = SparseCategoricalCrossentropy()(y_true["classes"], y_pred["detection_classes"])
    # Box regression loss (this is a simplified example, you may use more complex loss functions)
    regression_loss = tf.reduce_mean(tf.abs(y_true["boxes"] - y_pred["detection_boxes"]))
    return classification_loss + regression_loss

model.compile(optimizer=Adam(learning_rate=1e-4), loss=custom_loss)

