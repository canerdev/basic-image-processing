from cmath import rect
from turtle import width
import cv2
import numpy as np

# img = cv2.imread('img/rov.jpg')
# imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow('normal img', img)
# # cv2.imshow('gray img', imgGray)

# size_y = img.shape[0]
# size_x = img.shape[1]
# kanal = img.shape[2]

# print(size_x, size_y, kanal) # 400 400 3
# print(img[(150,210)]) # [111 118 115] -- to get the RGB code of a specific pixel

camera = cv2.VideoCapture(0)
kernel = np.ones((12,12), np.uint8)
while True: 
    ret, frame =  camera.read()
    square = frame[0:250, 0:250]
    square_HSV = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    square_GRAY = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    
    min_value = np.array([0, 20, 40])
    max_value = np.array([40, 255, 255])
    # print(min.ndim) => it gives the dimension of the array (1D, 2D, etc.) 
    
    filtered_result = cv2.inRange(square_HSV, min_value, max_value)
    filtered_result = cv2.morphologyEx(filtered_result, cv2.MORPH_CLOSE, kernel)
    
    result = square.copy()
    
    
    cnts,_ = cv2.findContours(filtered_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    max_width = 0 
    max_height = 0
    max_index = -1
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>max_width and h>max_height):
            max_height = h
            max_width = w
            max_index = t
            
    
    if(len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[max_index])
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2) 
    
    
    cv2.imshow("camera",frame)  
    cv2.imshow('square', square)      
    cv2.imshow('filtered result', filtered_result)
    cv2.imshow('result', result)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # q'ya basinca  
        break
    
camera.release() # turns the camera off 
cv2.destroyAllWindows()
