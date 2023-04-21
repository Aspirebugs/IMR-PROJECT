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