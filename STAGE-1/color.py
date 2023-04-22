import cv2
import numpy as np

def nothing(x):
    pass

def color(hsvFrame):
 red_lower = np.array([161, 155, 84], np.uint8)
 red_upper = np.array([179, 255, 255], np.uint8)
 red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

 # Set range for green color and
 # define mask
 green_lower = np.array([25, 52, 72], np.uint8)
 green_upper = np.array([102, 255, 255], np.uint8)
 green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

# Set range for blue color and
# define mask
 blue_lower = np.array([94, 80, 2], np.uint8)
 blue_upper = np.array([126, 255, 255], np.uint8)
 blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

 # Morphological Transform, Dilation
# for each color and bitwise_and operator
# between imageFrame and mask determines
# to detect only that particular color
 kernel = np.ones((5, 5), "uint8")

# For red color
 red_mask = cv2.dilate(red_mask, kernel)
# For green color
 green_mask = cv2.dilate(green_mask, kernel)
# For blue color
 blue_mask = cv2.dilate(blue_mask, kernel)
# Creating contour to track blue color
 contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
 for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            print("BLUE")
# Creating contour to track red color
 contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

 for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            print("RED")

# Creating contour to track green color
 contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

 for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
           print("GREEN")




cap = cv2.VideoCapture(r"C:\Users\admin\OpenCV-IR\test1.mp4");

cv2.namedWindow("Tracking")
cv2.resizeWindow("Tracking",640,240)
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
while True:
    #frame = cv2.imread('smarties.png')
    _,img=cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(img,img, mask=mask)

    cv2.imshow("frame",img)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    color(hsv)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
cap=cv2.VideoCapture(r'C:\Users\admin\OpenCV-IR\test2.mp4')
framewidth=840
frameheight=480
cap.set(3,framewidth)
cap.set(4,frameheight)

def empty(a):
    pass
cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",640,240)
cv2.createTrackbar("threshold1","parameters",34,255,empty)
cv2.createTrackbar("threshold2","parameters",24,255,empty)
cv2.createTrackbar("Area","parameters",7500,20000,empty)
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areamin=cv2.getTrackbarPos("Area", "parameters")
        if area > areamin:
            '''cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)'''
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3:
                print("Triangle")
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif objCor == 4:
                    print("Square")
                    cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Circle")
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
while True:
  img = cv2.absdiff(frame1, frame2)
  imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
  imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
  threshold1=cv2.getTrackbarPos("threshold1","parameters")
  threshold2=cv2.getTrackbarPos("threshold2","parameters")
  imgCanny = cv2.Canny(imgBlur,threshold1,threshold1)
  kernel = np.ones((5,5))
  imgDil=cv2.dilate(imgCanny,kernel,iterations=1)
  getContours(imgDil,frame1)
  imgStack = stackImages(0.33, ([img, imgGray, imgCanny],
                             [imgDil,frame1,frame1]))

  cv2.imshow("Stack",imgStack)
  frame1 = frame2
  ret, frame2 = cap.read()
  if cv2.waitKey(1)==ord('q'):
      cv2.destroyAllWindows()
      break
