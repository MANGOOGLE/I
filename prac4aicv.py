import cv2

image = cv2.imread("image1.jpg")
cv2.imshow("first", image)

gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("second", gray)

resize = cv2.resize(image, (220, 200))
cv2.imshow("third", resize)

imagerotation = (h,w) = image.shape[:2]
center = (h//2, w//2)
N = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, N, (w,h))
cv2.imshow("fourth", rotated)

crop = image[50:200, 100:300]
cv2.imshow("fifth", crop)

blur = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow("sixth", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
