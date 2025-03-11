import cv2 as cv
import pytesseract
import win32api, win32con
import time
from PIL import ImageGrab
import fnmatch
from enum import Enum
import os
from datetime import datetime
import numpy as np



# dictionary of the main UI elements (left,top,right,bottom)
focus = {
    "mainMenu": [4,38,638,70],
    "mainMenuItems": [4,66,931,619],
    # "mainMenuItems": [4,66,931,619],
    "centrRegion": [600,400, 1700, 800],
    "objectInspector": [256,256,512,512],
    #"fullScreen": [0,0,1920*1.25,966*1.25],#additional 4k dimension scale
    "fullScreen": [0,0,1920,966]    #TODO: consider to replace it with: if all zero-> ImageGrab.grab() (func takesnapshot )
}

def textBlocks():
    img = cv.imread('C:\\ocr\\OCR_Tester\\CVtestBench\\menu.jpg')
    
        # Preprocessing the image starts
    
        # Convert the image to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


        # Performing OTSU threshold
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    # ret, thresh2 = cv.threshold(gray, 120, 255,cv.THRESH_BINARY_INV)

    # cv.imshow('step1',thresh1)    
    # cv.imshow('step2',thresh2)    
    # cv.waitKey(0) 

    # Specify structure shape and kernel size. 
        # Kernel size increases or decreases the area 
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect 
        # each word instead of a sentence.
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
        
    #     # Applying dilation on the threshold image (text will become white blobs )
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)
    cv.imshow('step2',dilation)    
    cv.waitKey(0) 

    #     # Finding contours
    #     #contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)     
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # Drawing a rectangle on copied image
        rect = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    cv.imshow('step1',img)    
    cv.waitKey(0) 


def screenshotFocusDiff(uiElement):
    #return ImageGrab.grab(bbox=(uiElement[0], uiElement[1], uiElement[2], uiElement[3]))
    oldo = cv.imread('C:\\ocr\\OCR_Tester\\CVtestBench\\0.jpg')
    newo = cv.imread('C:\\ocr\\OCR_Tester\\CVtestBench\\1.jpg')
    #y:y+h, x:x+w
    cropped_image_old = oldo[uiElement[1]:uiElement[3], uiElement[0]:uiElement[2]]
    cropped_image_new = newo[uiElement[1]:uiElement[3], uiElement[0]:uiElement[2]]     
    old = cv.cvtColor(cropped_image_old, cv.COLOR_BGR2GRAY)
    new = cv.cvtColor(cropped_image_new, cv.COLOR_BGR2GRAY)   

    diff=cv.subtract(old,new)
    ret, thresh1 = cv.threshold(diff, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
        
   
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 3)    
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)        
        # Drawing a rectangle on copied image
        if w>200:
            #top, left, bottom, right
            rect = cv.rectangle(newo, (uiElement[0]+x, uiElement[1]+y), (uiElement[0]+x + w, uiElement[1]+y + h), (0, 255, 0), 1)            
       
    print(len(contours))
    cv.imwrite('C:\\ocr\\OCR_Tester\\CVtestBench\\outDiff.jpg',newo)
    cv.imshow('step1',cropped_image_new)    
    cv.waitKey(0) 


# textBlocks()
screenshotFocusDiff(focus["mainMenuItems"])