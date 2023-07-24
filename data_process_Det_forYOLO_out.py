import json
import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import yaml
from sklearn.model_selection import train_test_split



def divide_files_into_folders(main_folder,output_folder_images,output_folder_jsons):
    """
    Move image and JSON files from a main folder to separate image and JSON folders.

    Args:
        main_folder (str): The directory containing the unorganized files.
        row_folder (str): The directory to which the organized files will be moved.
    """
    # Iterate through the subfolders in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the path is a folder and not an output folder
        if os.path.isdir(subfolder_path) and subfolder != output_folder_images and subfolder != output_folder_jsons:
            
            # Iterate through the files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.startswith("._"):
                    # Delete the file
                    file_path = os.path.join(subfolder_path, file)
                    os.remove(file_path)
                    continue

                file_path = os.path.join(subfolder_path, file)

                # Check if the file is an image (assuming .jpg or .png format)
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Check if the filename contains the word "Building"
                    #if 'diaphragm' not in file.lower() and 'structural' not in file.lower():
                    if "building" in file.lower():
                    #if 1:
                        # Copy the image to the images folder
                        shutil.copy(file_path, output_folder_images)

                # Check if the file is a .json file
                elif file.lower().endswith(".json"):
                    # Check if the filename contains the word "Building"
                    if "building" in file.lower():
                    #if 1:
                        # Copy the .json file to the json_files folder
                        shutil.copy(file_path, output_folder_jsons)

    print("Files have been successfully divided into images and json_files folders within the data folder.")


def clean_label(label):
    '''
    Cleans a label by removing trailing 's' and converting it to uppercase.

    Args:
        label (str): The label to clean.

    Returns:
        str: The cleaned label.
    '''
    # Remove trailing 's'
    if label.endswith('s'):
        label = label[:-1]

    # Convert to uppercase
    label = label.upper()

    return label

def change_label_name(label):
    # this part is to transform the old names to new ones.
    label_dict = {
        "Window": "Window",
        "Precast-RC-slabs": "RC-Slab",
        #"RC-solid-slab": "RC-Slab"
        "RC-solid-slab": "RC2-Slab",# special for the outside some columns was wrong
        "RC-Joist": "RC-Joist",
        "PC1": "PC",
        "PC2": "PC",
        #"RC-Column": "RC-Column",
        "RC-Column": "RC2-Column",# special for the outside some columns was wrong
        "Slab": "RC2-Slab",
        "UCM/URM7": "Brick",
        "Timber-Frame": "Timber",
        "Timber-Column": "Timber-Column",
        "Timber-Joist": "Timber-Joist",
        "Light-roof": "Light-roof",
        "UCM/URM4": "Brick",
        "RM1": "Masonry",
        "Adobe": "Brick",
        'RC2-Column':'RC2-Column'#outside
    }
    
    return label_dict.get(label, label)


def clean_labels(data_folder):    
    '''
    Cleans and transfer the labels in JSON files within the specified data folder. 

    Args:
        data_folder (str): The path to the data folder.
    '''
    for file in os.listdir(data_folder):
        if file.lower().endswith(".json"):
            # Load the .json file
            with open(os.path.join(data_folder, file), 'r') as json_file:
                data = json.load(json_file)
                new_shapes = []

                for obj in data['shapes']:
                    print(obj['label'])

                    new_label = change_label_name(obj['label'])  

                    print(new_label)

                    # clean the label
                    new_label = clean_label(new_label)
                    print(new_label)

                    if new_label in class_map:
                        obj['label'] = new_label
                        print(new_label)
                        new_shapes.append(obj)
                    
                    print("----------------------------")

                # Replace the old shapes with the new ones
                data['shapes'] = new_shapes

            # Save the cleaned json file
            with open(os.path.join(data_folder, file), 'w') as json_file:
                json.dump(data, json_file)
    print('clean successfully')



def transform_annotations(data_folder, output_folder):
    """
    Transform the annotations from JSON files into a format suitable for object detection tasks.

    Args:
        data_folder (str): The path to the directory that contains the original JSON files.
        output_folder (str): The path to the directory where the transformed files will be saved.
    """
    output_folder_images = os.path.join(output_folder, 'images')
    output_folder_labels = os.path.join(output_folder, 'labels')
    
    # Create the necessary folders if they do not exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    # Create train and val folders for images and labels
    os.makedirs(os.path.join(output_folder_images, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_images, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_labels, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_labels, 'val'), exist_ok=True)

    # Split into train and val
    files = os.listdir(data_folder)
    np.random.shuffle(files)
    num_train = int(0.8 * len(files))
    train_files = files[:num_train]
    val_files = files[num_train:]

    # Function to handle each file
    def handle_file(file, output_folder_images, output_folder_labels):
        """
        Processes a single file: reads the data, checks if the label exists in the class map, 
        calculates bounding boxes, and writes the output to the appropriate .txt file. 
        Also, it copies the image to a new folder.

        Args:
            file (str): The path to the input file. Expected to be a JSON file containing bounding box annotations.
            output_folder_images (str): The path to the folder where the images should be saved.
            output_folder_labels (str): The path to the folder where the label files should be saved.

        Raises:
            IOError: An error occurred while trying to read the file or write the output.
        """
        if file.lower().endswith(".json"):
            # Load the .json file
            with open(os.path.join(data_folder, file), 'r') as json_file:
                data = json.load(json_file)

                img_width = int(data['imageWidth'])
                img_height = int(data['imageHeight'])
                # Get the image file path
                image_file_name = Path(file).stem + ".jpg"
                image_file_path = os.path.join(data_folder, image_file_name)

                # Create a new .txt file for each .json file
                with open(os.path.join(output_folder_labels, f"{Path(file).stem}.txt"), 'w') as txt_file:
                    # Iterate through the objects in the .json file
                    for obj in data['shapes']:


                        # Check if label exists in the class_map
                        if obj['label'] not in class_map:
                            print(f"Skipping label not found in class_map: {obj['label']}")
                            continue
                        
                        points = np.array(obj['points'])
                        
                        # Calculate the min and max x and y coordinates
                        xmin, ymin = np.min(points, axis=0)
                        xmax, ymax = np.max(points, axis=0)
                        
                        xmin = max(0, min(xmin, img_width))
                        xmax = max(0, min(xmax, img_width))
                        ymin = max(0, min(ymin, img_height))
                        ymax = max(0, min(ymax, img_height))

                        # Check if the width and height are valid
                        if xmax - xmin <= 0 or ymax - ymin <= 0:
                            print(f"Invalid box in {file}: ({xmin}, {ymin}, {xmax}, {ymax})")
                            continue
                        
                        # Transform to the format that YOLOv5 expects
                        x_center = ((xmin + xmax) / 2) / img_width
                        y_center = ((ymin + ymax) / 2) / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        if width>1:
                            width = 1
                        if height>1:
                            height = 1

                        class_id = class_map[obj['label']]
                        txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                # Copy the image to the new folder
                shutil.copyfile(image_file_path, os.path.join(output_folder_images, f"{Path(file).stem}.jpg"))

    # Process all the files
    for file in train_files:
        handle_file(file, os.path.join(output_folder_images, 'train'), os.path.join(output_folder_labels, 'train'))

    for file in val_files:
        handle_file(file, os.path.join(output_folder_images, 'val'), os.path.join(output_folder_labels, 'val'))

    # Write YAML file
    data_yaml = {
        'train': './images/train',
        'val': './images/val',
        'nc': len(class_map),
        'names': list(class_map.keys()),
    }
    #This is to write the yolo running comment 
    comment = f"#yolo detect train data=/scratch/tz2518/TargetDetection_YOLO/{class_name}_data_YOLO_Single_targetDetection/data.yaml model=yolov8x.yaml pretrained=/scratch/tz2518/ultralytics/yolov8x.pt epochs=1000 imgsz=1024 cache=True name={class_name}"

    with open(os.path.join(output_folder, 'data.yaml'), 'w') as yaml_file:
        yaml_file.write(comment + "\n")
        yaml.dump(data_yaml, yaml_file, sort_keys=False)

# define the classes to save
all_classes = {
    "WINDOW": 0,
    "PC": 1,
    "BRICK": 2,
    "LIGHT-ROOF": 3,
    "TIMBER": 4,
    "RC2-SLAB": 5,
    "RC2-COLUMN": 6,
}

# input and output path
main_folder = '/scratch/tz2518/data_7.3'
output_base_folder = '/scratch/tz2518/TargetDetection_YOLO'
class_map = {}

# for every class
for class_name in all_classes.keys():
    # clean class_map and add one new class
    class_map.clear()
    class_map[class_name] = 0
    # create ouput folder
    output_folder = os.path.join(output_base_folder, f'{class_name}_data_YOLO_Single_targetDetection')
    os.makedirs(output_folder, exist_ok=True)
    # create folders in the ouput folder
    row_folder = os.path.join(output_folder, 'row')
    os.makedirs(row_folder, exist_ok=True)

    # divide the images to one folder
    divide_files_into_folders(main_folder, row_folder, row_folder)
    #clean the json files
    clean_labels(row_folder)
    # transfer labels to yolo's
    transform_annotations(row_folder, output_folder)





