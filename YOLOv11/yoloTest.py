from ultralytics import YOLO
import os

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data="data.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        batch=0.5, 
        workers = 0
    )

    # Evaluate model performance on the validation set
    metrics = model.val(save_json = True)

    folder_path = "test\\images"
    # Perform object detection on a folder of images
    results = model.predict(folder_path, save=True, save_txt = True, save_conf = True)#(file_path)
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model