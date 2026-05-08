import cv2
image = cv2.imread("images.jpg")
image_hsv = cv2.cvtColor(image,
cv2.COLOR_BGR2HSV)
image_color_low = (1,40,60)
image_color_high = (18 ,255,200)
only_image_hsv = cv2.inRange(image_hsv,
image_color_low, image_color_high)
cv2.imwrite("image_color_hsv.jpg",
only_image_hsv)
cv2.waitKey(0)
# Визначимо особливі точки на зображенні (б)
# (рис. 4).
img = cv2.imread("image_color_hsv.jpg",0)
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None,
color=(255,0,0))
br = cv2.BRISK_create()
kp, des = br.compute(img, kp)
cv2.imwrite("im_end.png",img2)
cv2.waitKey(0)