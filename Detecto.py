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

#import matplotlib.pyplot as plt

# data augmentation

custom_transforms = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize(900),
transforms.RandomHorizontalFlip(0.5),
transforms.ColorJitter(saturation=0.2),
transforms.ToTensor(),
utils.normalize_transform(),
])

# build model and train

Train_dataset=core.Dataset('./train',transform=custom_transforms)
Test_dataset = core.Dataset('./test')
loader=core.DataLoader(Train_dataset, batch_size=2, shuffle=True)
model = core.Model(['Bicycle', 'others'])
losses = model.fit(loader, Test_dataset, epochs=25, lr_step_size=5, learning_rate=0.001, verbose=True)

# draw the loss evolution chart

#plt.plot(losses)
#plt.show()

# save the model
model.save('model_weights.pth')
model = core.Model.load('model_weights.pth', ['astrild', 'others'])

# test on an image

image = utils.read_image('./test/astrild.jpg') 
predictions = model.predict(image)
labels, boxes, scores = predictions
show_labeled_image(image, boxes, labels)

# threshold for a valid detection
thresh=0.8
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(image, filtered_boxes, filtered_labels)


