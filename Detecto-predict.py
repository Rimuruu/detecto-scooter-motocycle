# Simple object detection using Detecto library from AlanBi
#
# https://www.analyticsvidhya.com/blog/2021/06/simplest-way-to-do-object-detection-on-custom-datasets/

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms


model = core.Model.load('c:/Users/Del/Documents/Python Scripts/model_weights.pth', ['astrild', 'others'])

# test on an image

image = utils.read_image('c:/Users/Del/Documents/Python Scripts/test/astrild2.jpg') 
predictions = model.predict(image)
labels, boxes, scores = predictions
#show_labeled_image(image, boxes, labels)

# threshold for a valid detection
thresh=0.18
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(image, filtered_boxes, filtered_labels)


