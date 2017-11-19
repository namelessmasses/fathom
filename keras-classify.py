#!/usr/bin/python
# -*- mode: python -*-

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

f = 'coffee.png'
img_path = datadir + f
img = cv2.imread(img_path)
print(img.dtype, img.shape)
x = cv2.resize(img, (224, 224))
print(x.dtype, x.shape)
x = np.expand_dims(x, axis=0)
print(x.dtype, x.shape)
#    x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
dp = decode_predictions(preds, top=3)[0]
print('Predicted:', dp)

