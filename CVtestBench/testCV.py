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
import pyautogui as pag

 #pip install --upgrade scikit-image
 #pip install --upgrade imutils

from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils



# dictionary of the main UI elements (left,top,right,bottom)
focus = {
    "mainMenu": [4,38,638,70],
    "mainMenuItems": [4,66,931,619],   
    "centrRegion": [600,400, 1700, 800],
    "objectInspector": [256,256,512,512],
    "palette": [1407,558,1828,963],
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

    (score, diff) = compare_ssim(old, new, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    # threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
	    # compute the bounding box of the contour and then draw the
	    # bounding box on both input images to represent where the two
	    # images differ
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(oldo, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(newo, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv.imshow("Original", oldo)
    cv.imshow("Modified", newo)
    cv.imshow("Diff", diff)
    cv.imshow("Thresh", thresh)
    cv.waitKey(0)

    # diff=cv.subtract(old,new)  
    # cv.imwrite("diff.jpg",diff)    
    
    # # ret, thresh1 = cv.threshold(diff, 12, 255, cv.THRESH_BINARY)    
    # rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 12))      
    # # cv.imwrite("thresh1.jpg",thresh1) 
    # dilation = cv.dilate(diff, rect_kernel, iterations = 1)  
    # cv.imwrite("dilation.jpg",dilation)
    # edge = cv.Canny(dilation, 50, 100)  
    # cv.imwrite("edge.jpg",edge)    
    # contours, hierarchy = cv.findContours(edge, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     x, y, w, h = cv.boundingRect(cnt)        
    #     # Drawing a rectangle on copied image
    #     if w>200:
    #         #top, left, bottom, right
    #         rect = cv.rectangle(newo, (uiElement[0]+x, uiElement[1]+y), (uiElement[0]+x + w, uiElement[1]+y + h), (0, 255, 0), 1)            
       
    # print(len(contours))
    # cv.imwrite('C:\\ocr\\OCR_Tester\\CVtestBench\\outDiff.jpg',newo)
    # cv.imshow('step1',cropped_image_new)    
    # cv.waitKey(0) 

def screenshotFocusDiff2(uiElement):
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
    # cv.imshow('step1',cropped_image_new)    
    # cv.waitKey(0) 

def findFeature(uiElement, target, xOffset, yOffset):
    srcImg=cv.imread('C:\\ocr\\OCR_Tester\\CVtestBench\\findTarget.jpgt')
    trgImg = cv.imread('C:\\ocr\\OCR_Tester\\target\\hd\\'+target)
    srcImgCropped = srcImg[uiElement[1]:uiElement[3], uiElement[0]:uiElement[2]]
    cv.imshow('src',srcImgCropped)  
    
    cv.waitKey(0) 

    result = cv.matchTemplate(srcImgCropped, trgImg, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)  
    #[width,height]
    trgCenter = [int(max_loc[0]+trgImg.shape[1]/2),int(max_loc[1]+trgImg.shape[0]/2)]    
    #cv.circle(srcImgCropped, (trgCenter[0]+xOffset, trgCenter[1]+yOffset) , radius=3, color=(0, 0, 255), thickness=-1)
    return trgCenter[0]+xOffset, trgCenter[1]+yOffset   
    # cv.imshow('result',result)  

def dragDrop(src,dst):
    # sx,sy=(win32api.GetCursorPos()) # get current mouse pos
    # print(f"Current pos: {sx} {sy}")
    # # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,src[0]-sx,src[1]-sy,0,0) # use the MOUSEEVENTF_MOVE but send the deffrice betwwen current position and  first position
    # win32api.SetCursorPos((src[0],src[1]))
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,src[0],src[1],0,0)# press left clik (send the first postion)
    # sx,sy=(win32api.GetCursorPos())  # get current mouse pos
    # print(f"Current pos after move: {sx} {sy}")
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,dst[0]-sx,dst[1]-sy,0,0)  # use the MOUSEEVENTF_MOVE but send the deffrice betwwen current position and end position
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,dst[0],dst[1],0,0) # relase the left click (send the second postion)
    srcx=int(1*src[0])
    srcy=int(1*src[1])
    dstx=int(1*dst[0])
    dsty=int(1*dst[1])
    print(f'dragDrop from X:{src[0]}, Y:{src[1]} to X:{dst[0]}, Y:{dst[1]}')  
    win32api.SetCursorPos((srcx,srcy))
    sx,sy=(win32api.GetCursorPos()) 
    print(f'before move X:{sx}, Y:{sy}')    
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    sx,sy=(win32api.GetCursorPos()) 
    print(f'left_down X:{sx}, Y:{sy}')         
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,dstx-srcx,dsty-srcy,0,0)    
    print(f'move X:{dstx-srcx}, Y:{dsty-srcy}')  
    time.sleep(0.25)  
    sx,sy=(win32api.GetCursorPos()) 
    print(f'pos X:{sx}, Y:{sy}')   
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,srcx,srcy,0,0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,dstx,dsty,0,0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,dstx,dsty,0,0) 

def dragDropPy(src,dst): 
    pag.moveTo(src[0],src[1]) 
    pag.dragTo(dst[0], dst[1],2, button='left')  
# textBlocks()
# screenshotFocusDiff(focus["mainMenuItems"])

# time.sleep(4) 
# sampleTest(context)
# chekBitmapStyleDesigner(context)
# createNewVCLProject(context)
# ATfindComponent(context)
# dragDropPy((1517,677),(962, 598))
# dragDrop((1517,677),(962, 598))
# dragDrop((int(0.4*1459),int(0.4*773)),(int(0.4*1559),int(0.4*773)))

screenshotFocusDiff(focus["fullScreen"])
# findFeature(focus["palette"],'magnifingGlass.pngt',0,0)