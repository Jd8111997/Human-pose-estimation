# Human-pose-estimation
Human pose estimation

## Installation Guide:

- Clone the repository                                     
  ``git clone https://github.com/Jd8111997/Human-pose-estimation``                    
- Switch to the Simple-HRNet sub directory                            
  ``cd simple-HRNet``                     
- Install all required packages                                    
  ``pip install -r requirements.txt``                   
- Install ultralytics package                                           
  ``pip install ultralytics``                      
- Get YOLOv5:                                              
    - Clone [YOLOv5](git clone https://github.com/ultralytics/yolov5)
in the folder ``./models_/detectors`` and change the folder name from ``yolov5`` to ``yolo``                      
- Download the official pre-trained weights for the model:                     
    - COCO w48 384x288 - Used as default in `inference.py`                  
      [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)                                     
    - Create a new weights directory inside main repo and copy the pre-trained weights over there                               
- Uninstall the following package if it is getting deployed on cpu only machine                     
  ``python inference.py [-h] [--visualize]
                    [--output_folder_name OUTPUT_FOLDER_NAME]
                    image_path``   

## Usage Guide:

- The `inference.py` script will take an Image path (This could be either a path of a single image or a directory of images) and user can also provide a flag to visualize the output bounding box detection and keypoints on the input, third argument is the path of the output directory where the final output image with visualization will be saved.
``pip uninstall nvidia_cublas_cu11`` 



   
