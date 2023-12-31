# Import statements

import argparse
import cv2
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
import torch
from PIL import Image
import os
import glob
import sys
sys.path.append("simple-HRNet")
from misc.visualization import joints_dict
from collections import defaultdict
import tqdm
from SimpleHRNet import SimpleHRNet

# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type = str, help = 'path of the single image or directory of images')
parser.add_argument('--visualize', action = 'store_true', help = 'Flag to enable visualization')
parser.add_argument('--output_folder_name', type=str, help='Folder name to save visualizations', default = './output')
args = parser.parse_args()

def plot_joints(ax, output, im):
    """
    Arguments:
        ax: AxesSubplot object to plot the subplot for visualization
        output: model prediction object
        im: CV2 Image object
    Returns:
        None
    """
    # Retrieve the pre-trained coco skeleton to map predicted key points
    bones = joints_dict()["coco"]["skeleton"]
    print(im.shape)
    dh, dw, _ = im.shape

    # Plotting all detected bounding boxes in the input image
    for idx in range(len(output[0])):
        bbox = output[0][idx]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Applying thresholding to bound out of image predictions.
        x1 = max(min(x1, dw), 0)
        y1 = max(min(y1, dh), 0)
        x2 = max(min(x2, dw), 0)
        y2 = max(min(y2, dh), 0)
        
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plotting predicted key points for each detected bounding box.
    # Reference : https://github.com/stefanopini/simple-HRNet/blob/master/SimpleHRNet_notebook.ipynb
    for bone in bones:
        xS = [output[1][:,bone[0],1], output[1][:,bone[1],1]]
        yS = [output[1][:,bone[0],0], output[1][:,bone[1],0]]
        ax.plot(xS, yS, linewidth=3, c=(0,0.3,0.7))
    ax.scatter(output[1][:,:,1], output[1][:,:,0], s=20, c='r')

def visualize_img(im, img_name, joints, output_folder_name):

    """
    Arguments:
        im : CV2 Image object
        img_name: name of input image
        joints: model prediction object
        output_folder_name: path of output folder name to save visualization
    Returns:
        None
    """

    
    fig = plt.figure(figsize=(60/2.54, 30/2.54))
    ax = fig.add_subplot(121)
    ax.imshow(im)
    ax = fig.add_subplot(122)
    ax.imshow(im)
    
    try:
        plot_joints(ax, joints, im)
        save_img_name = img_name.split(".")[0] + '_viz.jpg'
        path = os.path.join(output_folder_name, save_img_name)
        plt.savefig(path)
    except:
        print(f'No detections found for image : {img_name}')
        pass

def inference_single_image(model, file_path, visualize, output_folder_name):
    """
    Arguments:
        model: An object of simple-HRNet model
        visualize: visualize input flag
        output_folder_name: path of the output folder name to save visualization
    Returns:
        An output dictionary with image name as a key and output dictionary as a value with
        corrosponding bounding box and key points as a keys.
    """

    # Reading image
    img_name = file_path.split('/')[-1]
    try:
        im = cv2.imread(file_path)
    except:
        print('Invalid Image type')
        return 
    
    # Model prediction
    joints = model.predict(im)

    if visualize:
        visualize_img(im, img_name, joints, output_folder_name)
    
    # Output
    return {img_name : {'bounding_box' : joints[0], 'key_points' : joints[1]}}
        

def inference_multiple_images(model, dir_path, visualize, output_folder_name):
    """
    Arguments:
        model: An object of simple-HRNet model
        dir_path: Path of a directory containing all input images
        visualize: visualize input flag
        output_folder_name: path of the output folder name to save visualization
    Returns:
        An output dictionary with image name as a key and output dictionary as a value with
        corrosponding bounding box and key points as a keys.
    """

    outputs = defaultdict() # output dictionary

    # Process input images
    imgs = glob.glob(dir_path + "/*.jpg") + glob.glob(dir_path + "/*.png")
    for img in tqdm.tqdm(imgs):
        img_name = img.split('/')[-1]

        try:
            im = cv2.imread(img)
            # Model prediction
            joints = model.predict(im)
            if visualize:
                visualize_img(im, img_name, joints, output_folder_name)
            # Appending the model output for the input image in a dictionary
            outputs[img_name] = {'bounding_box' : joints[0], 'key_points' : joints[1]}

        except:
            print('Invalid Image type')
            pass

    return outputs

def main():
    
    if not os.path.exists(args.image_path):
        print("The specified path doesn't exists")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Initializing SimpleHRNet object below.
    First argument is number of channels (when using HRNet model) or resnet size (when using PoseResNet model) - 48 hardcoded for pretrained.
    Second argument is number of keypoints that the model predicts.
    Third argument presents path of pretrained weights.

    """
    model = SimpleHRNet(48, 17, './weights/pose_hrnet_w48_384x288.pth', return_bounding_boxes = True, yolo_version = 'v5', yolo_model_def = 'yolov5n', device = device)

    if not os.path.exists(args.output_folder_name):
        os.makedirs(args.output_folder_name)

    if os.path.isfile(args.image_path):
        output = inference_single_image(model, args.image_path, args.visualize, args.output_folder_name)
    elif os.path.isdir(args.image_path):
        output = inference_multiple_images(model, args.image_path, args.visualize, args.output_folder_name)
    else:
        print("Invalid path provided.")

    print("Predictions : ")
    print(output)


if __name__ == "__main__":
    main()