#!/usr/bin/python2

import cv2
import time

# cv2.namedWindow('frame')
# cv2.startWindowThread()

cam = cv2.VideoCapture(0)
last_ts = time.time()
sum_acquisition_intervals = 0
count = 0
while cam.isOpened():
    ret, frame = cam.read()

    this_ts = time.time()
    delta_ts = this_ts - last_ts
    last_ts = this_ts
    sum_acquisition_intervals += delta_ts
    count += 1
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    print('delta_ts: {} avg: {} rate: {}'.format(delta_ts,
                                                 sum_acquisition_intervals / count,
                                                 count / sum_acquisition_intervals))

cam.release()
cv2.destroyAllWindows()
print('closing')
