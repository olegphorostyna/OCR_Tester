# How to run this script:
# install python3
# pip install opencv-python
# download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' to your script.
# pip install pytesseract
# pip install pywin32
# python3 test.py
# Import required packages
import cv2
import pytesseract
import win32api, win32con
import time
from PIL import ImageGrab
import fnmatch
from enum import Enum
import os
from datetime import datetime

class Actions(Enum):
    Click = 1
    FailIfExist = 2
    PassedIfExist = 3 
    


# dictionary of the main UI elements (left,top,right,bottom)
ideUi = {
    "mainMenu": [4,38,638,70],
    "mainMenuItems": [4,66,931,619],
    "centrRegion": [600,400, 1700, 800],
    "objectInspector": [256,256,512,512]   
}


def rescale(scaleFactor):
    for key in ideUi:
        if isinstance(ideUi[key], list):  # Check if the value is a list
            ideUi[key] = [item * scaleFactor for item in ideUi[key]]

class Context: 
    
    def __init__(self,scaleFactor,clickScale):
        self.dirPath=os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.dirPath)
        self.actionCounter=-1
        self.artifactsFolder=""
        #Scaling coefficient (e.g. 125%) scrren_x_dim/(img_res*1.25)=1/1.25
        #Regular display Display resolution 1920*966 scale 125 = 0.8
        #4k(3849*2160) Display resolution 3840*1948 scale 200 = 0.5
        self.clickScale=clickScale;
        # Originaly ui elements was created for HD monitor with scale=125%
        # With 4K monitor with scale set to 200% we need muliply each dimension by 200/125=1.6 
        rescale(scaleFactor)
    
    def setTest(self,testName): 
        self.actionCounter=-1        
        self.artifactsFolder=os.path.join(self.dirPath, testName)
        os.makedirs(self.artifactsFolder)
        print(testName)
    
    def getArtifactName(self):
        self.actionCounter+=1
        return os.path.join(self.artifactsFolder, str(self.actionCounter))
        
    def recognized(self):        
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"recognized.txt")
        
    def out(self):
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
    clickLeft(int(clickScale*(x+uiElement[0]+w/2)),int(clickScale*(y+uiElement[1]+h/2))) #4k click scale 

def prepareSnapshot(uiElement, context):
    snapshot = takeSnapshot(uiElement)
    save_path = context.getArtifactName()+"snapshot.jpg"
    snapshot.save(save_path)
    # Read image from which text needs to be extracted
    img = cv2.imread(save_path)
 
    # Preprocessing the image starts
 
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
 
    # Specify structure shape and kernel size. 
    # Kernel size increases or decreases the area 
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect 
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    #contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)     
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     
    return contours, img 

def performAction(action, contours, name, image, uiElement,psm,context):    
    file = open(context.recognized(), "w+")     
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    debugImage=image.copy();
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(debugImage, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]
        # Check what we feed into pytesseract <<DEBUG_FUNCTION>>
        #cv2.imwrite('cropped %d.jpg' %x ,cropped)  
        # Apply OCR on the cropped image
        # page segmentation modes (psm) is set
        text = pytesseract.image_to_string(cropped,lang='eng',config=psm)  
        if name in text : 
            match action:
                case Actions.Click:
                    clikElement(uiElement, x, y, w, h, context.clickScale)                 
                    cv2.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)
                case Actions.FailIfExist:                          
                    print("Failed!")
                case Actions.PassedIfExist:
                    print("Passed!")
        if text:
            file.write(text)            
    # Close the file
    file.close
    cv2.imwrite(context.out(),debugImage)
    

#Define all the tests here

    
# Demonstarete main functionality (contour drawing, element clicking) 
def sampleTest(context): 
    context.setTest("sampleTest") 
    contours, im2 = prepareSnapshot(ideUi["mainMenu"],context)
    performAction(Actions.Click, contours, "File", im2, ideUi["mainMenu"],"--psm 8",context)
    #restore for a next test
    performAction(Actions.Click, contours, "File", im2, ideUi["mainMenu"],"--psm 8",context)
    
#https://embt.atlassian.net/browse/RS-124356
def chekBitmapStyleDesigner(context):    
    context.setTest("chekBitmapStyleDesigner")
    contours, im2 = prepareSnapshot(ideUi["mainMenu"],context)
    performAction(Actions.Click, contours, "Tools", im2, ideUi["mainMenu"],"--psm 8",context)
    
    contours, im2 = prepareSnapshot(ideUi["mainMenuItems"],context)
    performAction(Actions.Click, contours, "Bitmap", im2, ideUi["mainMenuItems"], "--psm 7",context)
    
    contours, im2 = prepareSnapshot(ideUi["centrRegion"],context)
    performAction(Actions.FailIfExist, contours, "Access", im2, ideUi["centrRegion"], "--psm 11",context)

def createNewVCLProject(context):    
    context.setTest("createNewVCLProject")
    contours, im2 = prepareSnapshot(ideUi["mainMenu"],context)
    performAction(Actions.Click, contours, "File", im2, ideUi["mainMenu"],"--psm 8",context)
    time.sleep(0.5) 
    contours, im2 = prepareSnapshot(ideUi["mainMenuItems"],context)
    performAction(Actions.Click, contours, "New", im2, ideUi["mainMenuItems"], "--psm 7",context)
    time.sleep(0.5) 
    contours, im2 = prepareSnapshot(ideUi["mainMenuItems"],context)
    performAction(Actions.Click, contours, "Windows VCL Application - Delphi", im2, ideUi["mainMenuItems"], "--psm 7",context)
    
    
    
    
    
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  

#Set display scale and click scale before running test sequence
context = Context(1.6,0.5)   
#Test to run:
time.sleep(4) 
#sampleTest(context)
#chekBitmapStyleDesigner(context)
createNewVCLProject(context)
