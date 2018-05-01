import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
#import utils
import model as modellib
import visualize

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1 if torch.cuda.is_available() else 0
    IMAGES_PER_GPU = 1

    def __init__(self, num_classes):
        super(InferenceConfig, self).__init__(num_classes)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

type = 'parts'
if type == 'binary':
    num_classes = 1

    # Path to trained weights file
    # Download this file and place in the root of your
    # project (See README file for details)
    COCO_MODEL_PATH = "/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/pytorch_mask_rcnn_2/logs/coco20180430T1954/mask_rcnn_coco_0002.pth"  # os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

    # Directory of images to run detection on
    # IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    IMAGE_DIR = "/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/data/annotations/binary_folds/fold_0/train2014"
    class_names = ['BG', 'Instrument']
elif type == 'parts':
    num_classes = 4
    COCO_MODEL_PATH = "/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/pytorch_mask_rcnn_2/logs/coco20180501T1421/mask_rcnn_coco_0001.pth"  # os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
    IMAGE_DIR = "/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/data/annotations/parts_folds/fold_0/train2014"
    class_names = ['BG', 'Shaft', 'Wrist', 'Claspers']
else:
    raise Exception("Invalid type arg")

config = InferenceConfig(num_classes)
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
if config.GPU_COUNT:
    model.load_state_dict(torch.load(COCO_MODEL_PATH))
else:
    model.load_state_dict(torch.load(COCO_MODEL_PATH, map_location='cpu'))

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
n_images = 50
for i in range(n_images):
    random_img_file = random.choice(file_names)
    random_img_path = os.path.join(IMAGE_DIR, random_img_file)
    #random_img_path = "/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/data/cropped_train/instrument_dataset_1/images/frame016.jpg"
    print("img: %s"%random_img_path)
    image = skimage.io.imread(random_img_path)

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    visualize.display_instances_tempVR(random_img_file, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    plt.show()