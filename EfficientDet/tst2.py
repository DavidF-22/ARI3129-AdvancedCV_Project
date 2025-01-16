import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
import os
import xml.etree.ElementTree as ET

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

print(f"Loaded {len(train_dataset)} training images.")
print(f"Loaded {len(valid_dataset)} validation images.")

train_data = create_tf_dataset(train_dataset)
valid_data = create_tf_dataset(valid_dataset)

# Load EfficientDet from TF Hub
print("Loading EfficientDet from TF Hub...")
base_model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Fine-tuning head
print("Defining fine-tuning head...")
class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.cls_head = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        return self.cls_head(inputs)

# Add a detection head
detection_head = DetectionHead(NUM_CLASSES)

def custom_loss(y_true, y_pred):
    # Bounding box regression loss
    box_loss = tf.keras.losses.Huber(reduction="none")(
        y_true["boxes"], y_pred["boxes"]
    )
    box_loss = tf.reduce_mean(box_loss)  # Average over all boxes and batch

    # Classification loss
    class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        y_true["classes"], y_pred["classes"]
    )
    class_loss = tf.reduce_mean(class_loss)  # Average over batch

    # Combine losses
    total_loss = box_loss + class_loss
    return total_loss

# Training model
class EfficientDetFineTuned(tf.keras.Model):
    def __init__(self, base_model, detection_head, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.detection_head = detection_head

    def call(self, inputs):
        def process_image(image):
            # Add batch dimension
            image = tf.expand_dims(image, axis=0)
            base_outputs = self.base_model(image)

            # Remove batch dimension and slice to MAX_BOXES
            detection_boxes = tf.squeeze(base_outputs["detection_boxes"], axis=0)[:MAX_BOXES]
            detection_scores = tf.squeeze(base_outputs["detection_scores"], axis=0)[:MAX_BOXES]

            # Compute detection_classes from limited detection_scores
            detection_classes = tf.cast(tf.argmax(detection_scores, axis=-1), tf.int32)

            tf.print("Detection boxes:", tf.shape(detection_boxes))
            tf.print("Detection classes:", tf.shape(detection_classes))
            
            # Ensure detection_boxes and detection_classes are not empty
            detection_boxes = tf.cond(
                tf.shape(detection_boxes)[0] > 0,
                lambda: tf.pad(detection_boxes, [[0, MAX_BOXES - tf.shape(detection_boxes)[0]], [0, 0]], constant_values=-1),
                lambda: tf.fill([MAX_BOXES, 4], -1.0)  # Default value for empty boxes
            )

            detection_classes = tf.cond(
                tf.shape(detection_classes)[0] > 0,
                lambda: tf.pad(detection_classes, [[0, MAX_BOXES - tf.shape(detection_classes)[0]]], constant_values=-1),
                lambda: tf.fill([MAX_BOXES], -1)  # Default value for empty classes
            )

            return detection_boxes, detection_classes


        # Map process_image over the batch
        detection_boxes, detection_classes = tf.map_fn(
            lambda image: process_image(image),
            inputs,
            fn_output_signature=(
                tf.TensorSpec(shape=(MAX_BOXES, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(MAX_BOXES,), dtype=tf.int32),
            ),
        )

        return {"boxes": detection_boxes, "classes": detection_classes}


model = EfficientDetFineTuned(base_model, detection_head)

# Compile the model
print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=custom_loss,
)

# Train the model
print("Training the model...")
model.fit(train_data, validation_data=valid_data, epochs=EPOCHS)

# Save the model
print("Saving the model...")
model.save("fine_tuned_efficientdet")
