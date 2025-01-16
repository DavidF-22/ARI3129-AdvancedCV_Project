import tensorflow as tf
import tensorflow_hub as hub
import os
import xml.etree.ElementTree as ET

# Constants
NUM_CLASSES = 4
INPUT_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 10

CLASS_MAPPING = {
    "Mixed Waste -Black Bag-": 0,
    "Organic Waste -White Bag-": 1,
    "Other": 2,
    "Recycled Waste -Grey or Green Bag-": 3,
}

# Dataset paths
TRAIN_DIR = "trashy-dataset.voc/train/"
VALID_DIR = "trashy-dataset.voc/valid/"

# Dataset parsing
def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    classes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = CLASS_MAPPING[class_name]
        bndbox = obj.find("bndbox")
        boxes.append([
            float(bndbox.find("xmin").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymax").text),
        ])
        classes.append(class_id)
    return boxes, classes

def preprocess_data(image_path, boxes, classes):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SIZE)
    image = tf.cast(image, tf.uint8)  # Convert to uint8
    return image, {"boxes": tf.convert_to_tensor(boxes, dtype=tf.float32),
                   "classes": tf.convert_to_tensor(classes, dtype=tf.int32)}

def create_dataset(data_dir):
    dataset = []
    for file in os.listdir(data_dir):
        if file.endswith(".xml"):
            annotation_path = os.path.join(data_dir, file)
            image_file = os.path.join(data_dir, ET.parse(annotation_path).getroot().find("filename").text)
            boxes, classes = parse_voc_annotation(annotation_path)
            dataset.append((image_file, boxes, classes))
    return tf.data.Dataset.from_generator(
        lambda: ((img_path, boxes, classes) for img_path, boxes, classes in dataset),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    ).map(preprocess_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load datasets
train_data = create_dataset(TRAIN_DIR)
valid_data = create_dataset(VALID_DIR)

# Load pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/efficientdet/d1/1"
model = hub.load(model_url)

# Fine-tuning setup
def build_model():
    # Input layer
    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=tf.float32, name="image")

    # Preprocessing layer
    def preprocess_image(image):
        resized_inputs = tf.image.resize(image, (512, 512))
        uint8_inputs = tf.cast(resized_inputs, tf.uint8)
        batched_inputs = tf.expand_dims(uint8_inputs, axis=0)  # Add batch dimension
        return batched_inputs

    # Apply preprocessing
    processed_inputs = tf.keras.layers.Lambda(preprocess_image)(inputs)

    # EfficientDet model outputs
    model_outputs = model.signatures['serving_default'](input_tensor=processed_inputs)

    detection_boxes = model_outputs["detection_boxes"]
    detection_scores = model_outputs["detection_scores"]
    detection_classes = model_outputs["detection_classes"]

    # Create the model
    detection_model = tf.keras.Model(
        inputs=inputs,
        outputs=[detection_boxes, detection_scores, detection_classes]
    )
    return detection_model


# Compile the model
detection_model = build_model()
detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss="categorical_crossentropy")

# Train the model
history = detection_model.fit(train_data,
                              validation_data=valid_data,
                              epochs=EPOCHS)

