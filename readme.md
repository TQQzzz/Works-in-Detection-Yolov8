- `Detection_YOLO/`
    - `data_process_Det_forYOLO_in.py`
    - `data_process_Det_forYOLO_out.py`
    - `evaluation_yolo_Detection.py`

--- 
--- 

### **Detection-YOLO**

- ###### **Data Process**

 ```
  python data_process_Det_forYOLO_in.py
  python data_process_Det_forYOLO_out.py
  ```

  Entering the commands mentioned above in the terminal will create folders similar to 'BRICK_data_YOLO_Single_targetDetection', which can be used as training data.

- ###### **Training Models**

  In the folder BRICK_data_YOLO_Single_targetDetection, there is a data.yaml file contained a command 

  ```
  yolo detect train data=/scratch/tz2518/TargetDetection_YOLO/BRICK-COLUMN_data_YOLO_Single_targetDetection/data.yaml  model=yolov8x.yaml pretrained=/scratch/tz2518/ultralytics/yolov8x.pt epochs=1000 imgsz=1024 cache=True name=BRICK
  ```

  copy this line and paste it into the terminal to run the training of BRICK. A similarly prepared command can be used in the data for other features.yaml files.

- ###### **Get the Output Folder**

  After returning the command at the last step, folders named by feature names are created in the folder-runs/detect. It may take a few hours to train the models. When it is complete, the folder like-runs/detect/BRICK will contain a few images as the output and predictions. There will also be a folder -runs/detect/BRICK/weights containing the model's weight.

- ###### **Evaluate the model**

  Run the evaluation code using this command

  ```
  python evaluation_yolo_Detection.py
  ```

  An Excel will be got in the same folder of evaluation_yolo.py. It calculates each feature's TP,  TN,  FP, FN, Accuracy, Precision, and Recall.
