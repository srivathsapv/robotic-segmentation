import cv2
from cocoapi_master.PythonAPI.pycocotools import mask
import numpy as np

a = cv2.imread("/Users/vishalrao/PycharmProjects/DeepLearning/robot-surgery-segmentation/data/cropped_train/instrument_dataset_1/binary_masks/frame000.png", 0)
b = mask.encode(np.asfortranarray(a))
print(b)