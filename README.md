
# ARI3129 - Advanced Computer Vision Project 2024/25

## Introduction
This project explores the use of computer vision for waste collection analysis on Maltese streets. The initiative involves automating the detection and classification of domestic waste bags using object detection models. By streamlining the waste detection process, this project aims to contribute towards improving waste management systems.

The work is divided into three core tasks:
1. **Image Capturing**: Collecting images of domestic waste bags on Maltese streets while adhering to GDPR regulations.
2. **Dataset Construction**: Creating and annotating a robust dataset for training object detection models.
3. **Object Detection**: Training and evaluating three different object detection models (Faster R-CNN, YOLOv11, and RetinaNet).

---

## Distribution of Work
The implementation of this project was a collaborative effort, with each team member responsible for training and evaluating a specific object detection model:

- **Faster R-CNN** - Jason Spiteri  
- **YOLOv11** - David Lee Parnis  
- **RetinaNet** - David Farrugia  

Each team member contributed to the preparation of the dataset, documentation, and analysis of model performance.

---

## How to Use

### Cloning the Repository
To begin, clone this repository to your local machine using the following command:
```bash
git clone https://github.com/DavidF-22/ARI3129-AdvancedCV_Project.git
```

### Setting Up the Environment
It is recommended that a virtual environment be used to manage dependencies and avoid conflicts with existing Python packages. Follow these steps:

1. Create a virtual environment:

On Windows:
```bash
python -m venv .venv
```
On Linux or Mac:
```bash
python3 -m venv .venv
```

3. Activate the Virtual Environment

On Windows:
```bash
.venv\Scripts\activate
```
On Linux or Mac:
```bash
source .venv/bin/activate
```

5. Install the Required Libraries
If a **requirements.txt** file is available in the repository, use the following command to install dependencies:
```bash
pip install -r __path_to_requirements.txt__
```

If **requirements.txt** is unavailable or causing issues then identify the required packages from the project files and using **pip** or **pip3** install them manually.

---

## Datasets

The dataset for this project is hosted on Roboflow. It includes annotated images of domestic waste bags in four categories:

-   **Mixed Waste** (Black Bags)
-   **Organic Waste** (White Bags)
-   **Recyclable Material** (Grey Bags)
-   **Other Waste**

### Accessing the Dataset

To obtain the dataset:

1.  Visit the [Roboflow Dataset](https://app.roboflow.com/ari3129-advancedcv-project/ari3129-advancedcv_project/1) and navigate to the version tab. Here you'll find the complied dataset used for this project.
2.  Download the dataset in the format required for training the available object detection models.
	- **Faster R-CNN** - COCO Foramt 
	- **YOLOv11**  - YOLOv11 Format
	- **RetinaNet** -  Pascal VOC
