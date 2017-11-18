import cv2

cv2.startWindowThread()
cv2.namedWindow('image')
datadir = '/usr/lib/python2.7/dist-packages/skimage/data/'

img = cv2.imread(datadir + 'astronaut.png')

# Adding text
cv2.putText(img,
            'Some text',
            (10,10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255))

cv2.putText(img,
            'Some more text',
            (10,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255))

cv2.imshow('image', img)

