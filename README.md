# Human-pose-estimation
Human pose estimation

## Installation

- Clone the Repository:                                                
  ```bash 
  git clone https://github.com/Jd8111997/Human-pose-estimation
  ```                   
- Navigate to the Simple-HRNet Subdirectory:                              
  ```bash
  cd simple-HRNet
  ```                     
- Install Required Packages:  
  ```bash
  pip install -r requirements.txt
  ```                   
- Install the Ultralytics package                                           
  ```bash 
  pip install ultralytics
  ```                      
- Obtain YOLOv5:                                              
    - Clone [YOLOv5](git clone https://github.com/ultralytics/yolov5)
into the ``./models_/detectors`` folder and change the folder name from ``yolov5`` to ``yolo``.                                
- Download the official pre-trained weights for the model:                     
    - For COCO w48 384x288 (Default in inference.py):
      [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)                                     
    - Create a new weights directory within the main repository and copy the downloaded pre-trained weights there.                              
- Optional Uninstallation (For CPU-Only Machine):
  - Uninstall the `nvidia_cublas_cu11` package:                   
  ```bash 
  pip uninstall nvidia_cublas_cu11
  ```   

## Usage

- The `inference.py` script analyzes human poses in images, offering the ability to detect bounding boxes and keypoints. Below are the available options:


```bash
python inference.py [-h] [--visualize] [--output_folder_name OUTPUT_FOLDER_NAME] image_path
```

- `image_path`: Path to a single image or a directory containing multiple images.
- `--visualize`: Flag to enable visualization of bounding box detection and keypoints in the input image.
- `--output_folder_name OUTPUT_FOLDER_NAME`: Path to the output directory to save the final output images with visualizations.

## Output

Upon executing the `inference.py` script, it runs a human pose estimation model and prints a dictionary containing model output at the end. This dictionary comprises:

- **Key**: Represents the name of the input image.
- **Value**: A nested dictionary containing:
  - **bounding_box**: A numpy array indicating the detected bounding box coordinates of the human.
  - **key_points**: A numpy array containing 17 keypoints for the detected human within the bounding box.

For Example: 

```python
{'baseball1.jpg': {'bounding_box': array([[141,  45, 618, 681]], dtype=int32), 'key_points': array([[[     144.38,      372.88,      1.0009],
        [     131.12,      372.88,     0.87439],
        [     137.75,      359.62,     0.98067],
        [     117.88,      319.88,     0.48314],
        ...
        [     561.75,      200.62,     0.88002]]], dtype=float32)}}
```
## Visualization of model output


