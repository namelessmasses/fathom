import cv2

cv2.startWindowThread()
cv2.namedWindow('image')
datadir = '/usr/lib/python2.7/dist-packages/skimage/data/'
img_path = datadir + 'horse.png'
img = cv2.imread(img_path)
print(img.shape[:2])

cv2.imshow('image', img)

