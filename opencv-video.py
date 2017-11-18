#!/usr/bin/python2

import cv2
import time

def displayResults(img, results):
    textRow = 10
    textRowSize = 20
    
    cv2.putText(img,
                'rate:{rate:.3f}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize
    
    cv2.putText(img,
                'mean(G):{green_mean:.3f}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize

    cv2.putText(img,
                'std(G):{green_stddev:.3f}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize

    cv2.putText(img,
                'Centroid(red):{red_centroid}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize

    cv2.putText(img,
                'Centroid(green):{green_centroid}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize

    cv2.putText(img,
                'Centroid(blue):{blue_centroid}'.format(**results),
                (10,textRow),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255))
    textRow += textRowSize

    cv2.putText(img,
                'Saturated pixel count:{saturated_pixel_count}'.format(**results),
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
    results['rate'] = count / sum_acquisition_intervals

    # Split out RGB channels. OpenCV stores pixels in BGR order.
    redChannel = frame[:,:,2]
    greenChannel = frame[:,:,1]
    blueChannel = frame[:,:,0]

    # Compute mean and std dev of the pixels in the green channel
    results['green_mean'] = blueChannel.mean()
    results['green_stddev'] = blueChannel.std()

    # Centroid for each channel - while we could use ndarray syntax
    # unrolling here and operating on the specific channels is
    # clearer

    def centroid(moments):
        return (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
    redMoments = cv2.moments(redChannel)
    results['red_centroid'] = centroid(redMoments)
    greenMoments = cv2.moments(greenChannel)
    results['green_centroid'] = centroid(greenMoments)
    blueMoments = cv2.moments(blueChannel)
    results['blue_centroid'] = centroid(blueMoments)
    
    # Count the saturated pixels - any pixel in any channel that is saturated
    saturated_pixel_count = 0 #len([1 for row in frame for col in range(len(row)) if (row[col] == 255).any()])
    results['saturated_pixel_count'] = saturated_pixel_count
    
    displayResults(frame, results)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    
cam.release()
cv2.destroyAllWindows()
print('closing')
