# How to run this script:
# install python3
# pip install opencv-python
# download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' to your script.
# pip install pytesseract
# pip install pywin32
# python3 test.py
# Import required packages
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

class Actions(Enum):
    Click = 1
    FailIfExist = 2
    PassedIfExist = 3 
    


# dictionary of the main UI elements (left,top,right,bottom)
focus = {
    "mainMenu": [4,38,638,70],
    "mainMenuItems": [4,66,931,619],
    "centrRegion": [600,400, 1700, 800],
    "objectInspector": [256,256,512,512],
    #"fullScreen": [0,0,1920*1.25,966*1.25],#additional 4k dimension scale
    "fullScreen": [0,0,1920,966]    #TODO: consider to replace it with: if all zero-> ImageGrab.grab() (func takesnapshot )
}


def rescale(scaleFactor):
    for key in focus:
        if isinstance(focus[key], list):  # Check if the value is a list
            focus[key] = [item * scaleFactor for item in focus[key]]

class Context: 
    
    def __init__(self,scaleFactor,clickScale):
        self.dirPath=os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.dirPath)
        self.actionCounter=-1
        self.artifactsFolder=""
        self.contextFolder=""
        #Scaling coefficient (e.g. 125%) scrren_x_dim/(img_res*1.25)=1/1.25
        #Regular display Display resolution 1920*966 scale 125 = 0.8
        #4k(3849*2160) Display resolution 3840*1948 scale 200 = 0.5
        self.clickScale=clickScale
        # Originaly ui elements was created for HD monitor with scale=125%
        # With 4K monitor with scale set to 200% we need muliply each dimension by 200/125=1.6 
        rescale(scaleFactor)
        self.prevPicture = None
        self.prevDimensions=None
    
    def setTest(self,testName): 
        self.actionCounter=-1        
        self.artifactsFolder=os.path.join(self.dirPath, testName)
        self.contextFolder=os.path.join(self.artifactsFolder, 'context')
        os.makedirs(self.artifactsFolder)
        os.makedirs(self.contextFolder)
        print(testName)
    
    def getSnapshotPath(self):
        self.actionCounter+=1
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"snapshot.jpg")
    
    #Writes last full-screen snapshot into the circular buffer with 2 elements
    def update(self):
        snapshot = takeSnapshot(focus["fullScreen"])        
        save_path = os.path.join(self.contextFolder, str(self.actionCounter%2)+".jpg")
        snapshot.save(save_path)   
    
    def getSnapshotForDiffPath(self):
        return os.path.join(self.contextFolder, str((self.actionCounter-1)%2)+".jpg"), os.path.join(self.contextFolder, str(self.actionCounter%2)+".jpg")
    
    def getRecognizedTextPath(self):        
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"recognized.txt")
        
    def getOutPath(self):
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"withCountours.jpg")

def clickRight(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
 
def clickLeft(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0) 


def takeSnapshot(uiElement):      
    return ImageGrab.grab(bbox=(uiElement[0], uiElement[1], uiElement[2], uiElement[3])) 
    # take whole screen 
    # snapshot = ImageGrab.grab()
    
 
def clikElement(uiElement, x, y, w, h,clickScale):     
    clickLeft(int(clickScale*(x+uiElement[0]+w/2)),int(clickScale*(y+uiElement[1]+h/2)))  

#Detects and returns region of change between the last two screenshots 
#TODO:add size constraints parameter
def diffFocusRegion(context:Context, uiElement=focus["fullScreen"]):
    old,new = context.getSnapshotForDiffPath(); 
    oldM = cv.imread(old)
    newM = cv.imread(new)    
    #y:y+h, x:x+w
    cropped_image_old = oldM[uiElement[1]:uiElement[3], uiElement[0]:uiElement[2]]
    cropped_image_new = newM[uiElement[1]:uiElement[3], uiElement[0]:uiElement[2]]     
    old = cv.cvtColor(cropped_image_old, cv.COLOR_BGR2GRAY)
    new = cv.cvtColor(cropped_image_new, cv.COLOR_BGR2GRAY)   

    diff=cv.subtract(old,new)
    ret, thresh1 = cv.threshold(diff, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))      
   
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 3)    
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)       
        if w>200:           
            #rect = cv.rectangle(newo, (uiElement[0]+x, uiElement[1]+y), (uiElement[0]+x + w, uiElement[1]+y + h), (0, 255, 0), 1)            
            return (uiElement[0]+x, uiElement[1]+y, uiElement[0]+ x + w, uiElement[1]+y + h)
    

def prepareSnapshot(uiElement, context:Context):     
    snapshot = takeSnapshot(uiElement)
    save_path = context.getSnapshotPath()
    snapshot.save(save_path)
       
    
    # Read image from which text needs to be extracted
    img = cv.imread(save_path)
 
    # Preprocessing the image starts
 
    # Convert the image to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   
    # Performing OTSU threshold
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    
    # Specify structure shape and kernel size. 
    # Kernel size increases or decreases the area 
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect 
    # each word instead of a sentence.
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
    
    # Applying dilation on the threshold image
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    #contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)     
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)     
    return contours, img 

def performAction(action, contours, name, image, uiElement,psm,context):    
    file = open(context.getRecognizedTextPath(), "w+")     
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    debugImage=image.copy()
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # Drawing a rectangle on copied image
        rect = cv.rectangle(debugImage, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]
        # Check what we feed into pytesseract <<DEBUG_FUNCTION>>
        #cv.imwrite('cropped %d.jpg' %x ,cropped)  
        # Apply OCR on the cropped image
        # page segmentation modes (psm) is set
        text = pytesseract.image_to_string(cropped,lang='eng',config=psm)  
        if name in text : 
            match action: #TODO: escape on match (leave as it's for debug purpouse)
                case Actions.Click:
                    clikElement(uiElement, x, y, w, h, context.clickScale)                 
                    cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)
                case Actions.FailIfExist:                          
                    print("Failed!")
                case Actions.PassedIfExist:
                    print("Passed!")
        if text:
            file.write(text)            
    # Close the file
    file.close
    cv.imwrite(context.getOutPath(),debugImage)

#Must call this method to update context
#Also, between action we need to wait a little to let IDE update it's GUI    
def actionDone(waitTime):
    time.sleep(waitTime)
    context.update()  

#Define all the tests here

    
# Demonstarete main functionality (contour drawing, element clicking) 
def sampleTest(context): 
    context.setTest("sampleTest") 
    contours, im2 = prepareSnapshot(focus["mainMenu"],context)
    performAction(Actions.Click, contours, "File", im2, focus["mainMenu"],"--psm 8",context)
    #restore for a next test
    performAction(Actions.Click, contours, "File", im2, focus["mainMenu"],"--psm 8",context)
    
#https://embt.atlassian.net/browse/RS-124356
def chekBitmapStyleDesigner(context):    
    context.setTest("chekBitmapStyleDesigner")
    contours, im2 = prepareSnapshot(focus["mainMenu"],context)
    performAction(Actions.Click, contours, "Tools", im2, focus["mainMenu"],"--psm 8",context)
    actionDone()
    contours, im2 = prepareSnapshot(focus["mainMenuItems"],context)
    performAction(Actions.Click, contours, "Bitmap", im2, focus["mainMenuItems"], "--psm 7",context)
    
    contours, im2 = prepareSnapshot(focus["centrRegion"],context)
    performAction(Actions.FailIfExist, contours, "Access", im2, focus["centrRegion"], "--psm 11",context)

def createNewVCLProject(context):    
    context.setTest("createNewVCLProject")
    contours, im2 = prepareSnapshot(focus["mainMenu"],context)
    performAction(Actions.Click, contours, "File", im2, focus["mainMenu"],"--psm 8",context)
    actionDone(0.5)
    contours, im2 = prepareSnapshot(focus["mainMenuItems"],context)
    performAction(Actions.Click, contours, "New", im2, focus["mainMenuItems"], "--psm 7",context)
    actionDone(0.5) 
    # diffFocus = diffFocusRegion(context,focus["mainMenuItems"])
    # contours, im2 = prepareSnapshot(focus["mainMenuItems"],context)
    # performAction(Actions.Click, contours, "Windows VCL Application - Delphi", im2, focus["mainMenuItems"], "--psm 7",context)
    diffFocus = diffFocusRegion(context,focus["mainMenuItems"])
    contours, im2 = prepareSnapshot(diffFocus,context)
    performAction(Actions.Click, contours, "Windows VCL Application - Delphi", im2, diffFocus, "--psm 7",context)
    actionDone(0.5) 
   
    
def testSubstract(context): 
    context.setTest("testSubstract")
    contours, im2 = prepareSnapshot(focus["mainMenu"],context)
    performAction(Actions.Click, contours, "File", im2, focus["mainMenu"],"--psm 8",context)
    time.sleep(0.5) 
    contours, im2 = prepareSnapshot(focus["mainMenuItems"],context)
    performAction(Actions.Click, contours, "New", im2, focus["mainMenuItems"], "--psm 7",context)
    time.sleep(0.5) 
    contours, im2 = prepareSnapshot(focus["diff"],context)
    performAction(Actions.Click, contours, "Windows VCL Application - Delphi", im2, focus["mainMenuItems"], "--psm 7",context)
       
#    
    
    
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  

# Screen space scale
# Originaly ui elements was created for HD monitor with scale=125%
# With 4K monitor with scale set to 200% we need muliply each dimension by 200/125=1.6 

#Click space scale
#Click space scale coefficient (e.g. 125%) scrren_x_dim/(img_res*1.25)=1/1.25
#Regular display Display resolution 1920*966 scale 125 = 0.8
#4k(3849*2160) Display resolution 3840*1948 scale 200 = 0.5
#context = Context(1.6,0.5)#4K   
context = Context(1,0.8)#HD   
#Test to run:
time.sleep(4) 
# sampleTest(context)
# chekBitmapStyleDesigner(context)
createNewVCLProject(context)
#testSubstract(context)
