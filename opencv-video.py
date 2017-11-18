#!/usr/bin/python2

import cv2
import time
import math

def displayResults(img, results):
    textRow = 10
    textRowSize = 20

    # \todo determine where in the image text will appear and select a
    # color to contrast the image background.
    #
    # \todo sort the keys
    keys = results.keys()
    keys.sort()
    for key in keys:
        cv2.putText(img,
                    '{}: {}'.format(key, results[key]),
                    (10,textRow),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255))
        textRow += textRowSize
        

# cv2.namedWindow('frame')
# cv2.startWindowThread()

cam = cv2.VideoCapture(0)
last_ts = time.time()
sum_acquisition_intervals = 0
count = 0
results = dict()

while cam.isOpened():
    ret, frame = cam.read()

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
    results['2. mean(green)'] = '{:.3f}'.format(blueChannel.mean())
    results['3. std(green)'] = '{:.3f}'.format(blueChannel.std())

    # Centroid for each channel - while we could use ndarray syntax
    # unrolling here and operating on the specific channels is
    # clearer

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

    results['7. rate'] = '{:.3f}'.format(count / sum_acquisition_intervals)

    # Count the saturated pixels - any pixel in any channel that is saturated
    #saturated_pixel_count = len([1 for row in frame for col in range(len(row)) if (row[col] == 255).any()])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturated_pixel_count = (hsv[:,:,1].flatten() == 255).sum()
    results['8. Saturated Pixel Count'] = saturated_pixel_count
    
    displayResults(frame, results)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()
print('closing')
