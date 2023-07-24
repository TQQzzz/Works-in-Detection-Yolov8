import cv2
import os
import json
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluation_detection(class_names_to_keep):
    """
    Evaluate the YOLO model's performance on the validation dataset. 
    It calculates and returns the number of True Positives (TP), False Positives (FP), 
    False Negatives (FN), and True Negatives (TN), along with accuracy, precision, and recall.

    Args:
        class_names_to_keep (List[str]): A list of class names that the model needs to detect.

    Raises:
        IOError: An error occurred while trying to read an image file.

    Returns:
        Tuple[int, int, int, int, float, float, float]: A tuple containing the number of 
        TP, FP, FN, and TN, along with accuracy, precision, and recall.
    """

    # Initialize model, folders, and counters
    # Note: Replace the '/scratch/tz2518/runs/segment/{class_names_to_keep}/weights/best.pt',
    # image_folder, and val_labels_folder's paths
    # to your actual model path.   
    model = YOLO(f'/scratch/tz2518/runs/detect/{class_names_to_keep}/weights/best.pt')
    image_folder = f'/scratch/tz2518/TargetDetection_YOLO/{class_names_to_keep}_data_YOLO_Single_targetDetection/images/train'
    val_labels_folder = f'/scratch/tz2518/TargetDetection_YOLO/{class_names_to_keep}_data_YOLO_Single_targetDetection/labels/train'


    TP = 0
    FP = 0
    TN = 0
    FN = 0

    predictions = []
    true_labels = []

    # iterate over every image file in the image_folder
    for image_file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file_name)

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise IOError
        except IOError:
            print(f"Failed to load image: {image_path}")
            continue

        results = model([img])  # perform detection using yolov8
        result = results[0]  # get the first result

        # check if the model predicted a mask
        prediction = len(result.boxes) > 0
        predictions.append(prediction)

        # get the true label
        label_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        label_file_path = os.path.join(val_labels_folder, label_file_name)
        true_label = os.path.exists(label_file_path) and os.stat(label_file_path).st_size > 0
        true_labels.append(true_label)

        # Update the counts in the confusion matrix
        if prediction and true_label:  # TP: Both the true and the predicted labels are positive
            TP += 1
        elif prediction and not true_label:  # FP: The true label is negative but the predicted label is positive
            FP += 1
        elif not prediction and true_label:  # FN: The true label is positive but the predicted label is negative
            FN += 1
        elif not prediction and not true_label:  # TN: Both the true and the predicted labels are negative
            TN += 1

    # calculate the scores
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    print('For the class:', class_names_to_keep)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print('TP',TP,'FP',FP,'FN',FN,'TN',TN)
    

    return TP, FP, FN, TN, accuracy, precision, recall


#build the dataframe
results = pd.DataFrame(columns=['Feature', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'Recall'])



#for each feature
features = ["WINDOW", "PC", "BRICK", "LIGHT-ROOF", "RC2-SLAB", "RC2-COLUMN",
            "RC-SLAB", "RC-JOIST", "RC-COLUMN", "TIMBER-COLUMN", "TIMBER-JOIST"]


for feature in features:
    
    start_time = time.time()
    TP, FP, FN, TN, accuracy, precision, recall = evaluation_detection(feature)
    elapsed_time = time.time() - start_time
    print('time-consuming:',elapsed_time)
    print('---------------------------------------------')
    results.loc[len(results)] = [feature, TP, FP, FN, TN, accuracy, precision, recall]



# save to Excel
results.to_excel("evaluation_yolo_detection.xlsx", index=False)