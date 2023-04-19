import numpy as npcd
import cv2
img = cv2.imread('shape.png')
h, w, _ = img.shape
area=h*w
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", img)
for i in range(len(contours)):
 approx = cv2.approxPolyDP(contours[i], 0.01* cv2.arcLength(contours[i], True), True)
 if i!=0:
  if len(approx) == 3:
        print("Triangle")
        x1, y1, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),2)
  elif len(approx) == 4:
          x1 ,y1, w, h = cv2.boundingRect(approx)
          print("Square")
          cv2.rectangle(img,(x1, y1),(x1+w,y1+h),(0,255,0),2)
  else:
        x1, y1, w, h = cv2.boundingRect(approx)
        print("Circle")
        cv2.rectangle(img,(x1, y1),(x1+w,y1+h),(0,255,0),2)


cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()