# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:29:44 2019

@author: ozanyildiz
About the R-CNN Code
Proposed a method where we use selective search to extract just 2000 regions from the image and he called them region proposals.
Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region. proposals are generated using the selective search algorithm which is written below.
Selective search applied by these steps:
Generate initial sub-segmentation, we generate many candidate regions
Use greedy algorithm to recursively combine similar regions into larger ones
Use the generated regions to produce the final candidate region proposals
These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network that produces a 4096-dimensional feature vector as output.

The extracted features are fed into an SVM to classify the presence of the object within that candidate region proposal.
"""
# importing required libraries
# import the necessary packages
from mrcnn.config import Config
#Mask R-CNN.
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os

#it is a simple config.It inherits from Matterport's Mask R-CNN
class Config(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"
 
	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)
    
def FR-CNN_Detection(frame,hsv,colors):
# initialize the inference configuration
config = Config()
 
# initialize the Mask R-CNN model for inference and then load the
# weights
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

#performing instance segmentation
# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)
 
# perform a forward pass of the network to obtain the results
print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

#process the results so that we can visualize the objects’ bounding boxes and masks using OpenCV:
# loop over of the detected object's bounding boxes and masks
for i in range(0, r["rois"].shape[0]):
	# extract the class ID and mask for the current detection, then
	# grab the color to visualize the mask (in BGR format)
	classID = r["class_ids"][i]
	mask = r["masks"][:, :, i]
	color = COLORS[classID][::-1]
 
	# visualize the pixel-wise mask of the object
	image = visualize.apply_mask(image, mask, color, alpha=0.5)
    
    #we’ll draw bounding boxes and class label + score texts for each object in the
    # convert the image back to BGR so we can use OpenCV's drawing
# functions
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
# loop over the predicted scores and class labels
for i in range(0, len(r["scores"])):
	# extract the bounding box information, class ID, label, predicted
	# probability, and visualization color
	(startY, startX, endY, endX) = r["rois"][i]
	classID = r["class_ids"][i]
	label = CLASS_NAMES[classID]
	score = r["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]
 
	# draw the bounding box, class label, and score of the object
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}: {:.3f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, color, 2)
 
# show the output image
cv2.imshow("Output", image)
cv2.waitKey()