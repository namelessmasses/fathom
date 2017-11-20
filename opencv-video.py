
#!/usr/bin/python2

import cv2
import time
import math
import threading

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

class ClassificationThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.frame = None
        self.running = False
        self.frameReadyEvent = threading.Event()
        self.resultsReadyEvent = threading.Event()
        self.results = None
        
    def setFrame(self, frame):
        if self.frameReadyEvent.is_set():
            raise 'Trying to set frame while classification still in progress'
        
        self.frame = frame
        self.frameReadyEvent.set()

    def getResults(self):
        self.resultsReadyEvent.wait()
        self.resultsReadyEvent.clear()
        return results

    def stop(self):
        self.running = True
        self.join()
    
    def run(self):
        self.running = True
        print('Classification thread running')
        model = ResNet50(weights='imagenet')

        avg_time = 0
        count = 0
        while self.running:
            self.frameReadyEvent.wait()
            count += 1
            # \todo what is this (224, 224) magic with this
            # classification model?
            img_to_classify = cv2.resize(self.frame, (224, 224))
            self.frameReadyEvent.clear()

            img_to_classify = np.expand_dims(img_to_classify, axis=0)

            # \todo MAJOR BOTTLENECK IS HERE
            preds = model.predict(img_to_classify)
            self.results = decode_predictions(preds, top=3)[0]
            self.resultsReadyEvent.set()

# Start the classification thread running and waiting for work
classification = ClassificationThread()
classification.start()
            
def centroid(moments):
    return (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

def d4sigma(moments, centroid = None):
    if centroid == None:
        centroid = centroid(moments)
        
    sigma2x = moments['m20']/moments['m00'] - centroid[0] * centroid[0]
    sigma2y = moments['m02']/moments['m00'] - centroid[1] * centroid[1]
        
    d4sx = 4 * math.sqrt(sigma2x)
    d4sy = 4 * math.sqrt(sigma2y)

    return (d4sx, d4sy)

def displayResults(img, results):
    textRow = 10
    textRowSize = 20

    # \todo determine where in the image text will appear and select a
    # color to contrast the image background.
    #
    keys = results.keys()
    keys.sort()
    for key in keys:
        print('{}: {}'.format(key, results[key]))
        # cv2.putText(img,
        #             '{}: {}'.format(key, results[key]),
        #             (10,textRow),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.4,
        #             (255, 255, 255))
        # textRow += textRowSize
        
cam = cv2.VideoCapture(0)
last_ts = time.time()
sum_acquisition_intervals = 0
count = 0
results = dict()

total = 0
while cam.isOpened():
    ret, frame = cam.read()

    # Hand off the frame to the classification thread to start working
    # asap
    classification.setFrame(frame)
    
    this_ts = time.time()
    delta_ts = this_ts - last_ts
    last_ts = this_ts
    sum_acquisition_intervals += delta_ts
    count += 1
    results['1. Image dimensions'] = frame.shape[:2]
    
    # Split out RGB channels. OpenCV stores pixels in BGR order.
    redChannel = frame[:,:,2]
    greenChannel = frame[:,:,1]
    blueChannel = frame[:,:,0]

    # Compute mean and std dev of the pixels in the green channel
    greenMean = greenChannel.mean()
    greenStd = greenChannel.std()
    results['2. mean(green)'] = '{:.3f}'.format(greenMean)
    results['3. std(green)'] = '{:.3f}'.format(greenStd)

    # Centroid for each channel - while we could use ndarray syntax
    # unrolling here and operating on the specific channels is
    # clearer

    redMoments = cv2.moments(redChannel)
    redCentroid = centroid(redMoments)
    redD4s = d4sigma(redMoments, redCentroid)
    results['4. Red Centroid'] = redCentroid
    results['4. Red d4sx'] = redD4s[0]
    results['4. Red d4sy'] = redD4s[1]
    
    greenMoments = cv2.moments(greenChannel)
    greenCentroid = centroid(greenMoments)
    greenD4s = d4sigma(greenMoments, greenCentroid)
    results['5. Green Centroid'] = greenCentroid
    results['5. Green d4sx'] = greenD4s[0]
    results['5. Green d4sy'] = greenD4s[1]

    blueMoments = cv2.moments(blueChannel)
    blueCentroid = centroid(blueMoments)
    blueD4s = d4sigma(blueMoments, blueCentroid)
    results['6. Blue Centroid'] = blueCentroid
    results['6. Blue d4sx'] = blueD4s[0]
    results['6. Blue d4sy'] = blueD4s[1]
    
    results['7. rate (frames/sec)'] = '{:.3f}'.format(count / sum_acquisition_intervals)

    # Count the saturated pixels - any pixel in any channel that is saturated
    #saturated_pixel_count = len([1 for row in frame for col in range(len(row)) if (row[col] == 255).any()])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturated_pixel_count = (hsv[:,:,1].flatten() == 255).sum()
    results['8. Saturated Pixel Count'] = saturated_pixel_count

    decoded_preds = classification.getResults()
    results['9. Predicted'] = decoded_preds
    total = (total * (count - 1) + (time.time() - this_ts)) / count
    results['99. total'] = total
    
    displayResults(frame, results)

    if count > 1000:
        break
    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break
    
cam.release()
cv2.destroyAllWindows()
print('closing')
