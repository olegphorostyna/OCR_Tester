# Test pywinauto (Spy)
# How to run this script:
# install python3
# pip install opencv-python
# download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' to your script.
# pip install pytesseract
# pip install pywin32
# pip install pyautogui
# pip install PyDirectInput
# pip install easyocr
# python3 test.py
# Import required packages
import cv2 as cv
import random
import pytesseract
import win32api, win32con
import time
from PIL import ImageGrab
import fnmatch
from enum import Enum
import os
from datetime import datetime
import numpy as np
import pydirectinput as pdi
import easyocr as eocr
import sys
import pyautogui as pag
 #pip install --upgrade scikit-image
 #pip install --upgrade imutils

from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils

debugLog = True

class Actions(Enum):
    Click = 1 #Text and Simpe action
    DoubleClick=2 #Text and Simpe action
    FailIfExist = 3 #Text action
    PassedIfExist = 4 #Text action
    ClickAndType = 5 #Simpe action
    FindClickLocation = 6 #Text action
    Type=7 #Simpe action
    

#TODO: program or routine for manual/automatic dict items detection
# dictionary of the main UI elements (left,top,right,bottom)
focus = {
    "mainMenu": [0,38,638,70],
    "mainMenuItems": [0,70,1050,820],    
    "editor": [385,115,1422,956],
    "projects": [1422,115,1828,560],
    "objectInspector": [0,456,385,966],
    "palette": [1422,560,1828,963],
    "structure":[0,115,385,456], 
    "fullScreen": [0,0,1920,966],
    "header": [0,36,1828,115]  
    # "fullScreen": [0,0,1920,966]    #TODO: consider to replace it with: if all zero-> ImageGrab.grab() (func takesnapshot )
}


def draw_rectangles(image_path, rectangles, output_path='output.png'):
    """
    Draws rectangles on an image and saves the result.
    
    :param image_path: Path to the input image.
    :param rectangles: Dictionary with rectangle names as keys and coordinates as values (left, top, right, bottom).
    :param output_path: Path to save the output image with rectangles drawn.
    """
    # Load the image
    image = cv.imread(image_path)
    
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")
    
    # Iterate over the rectangles and draw them
    for name, (left, top, right, bottom) in rectangles.items():
        color = (0, 0, random.randint(0, 255))  # Random color
        cv.rectangle(image, (left, top), (right, bottom), color, 2)  # Random colored rectangle with thickness 2
        cv.putText(image, name, (left+int((right-left)/2), top+int((bottom-top)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    
    # Save the output image
    cv.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")



def rescale(scaleFactor):
    for key in focus:
        if isinstance(focus[key], list):  # Check if the value is a list
            focus[key] = [int(item * scaleFactor) for item in focus[key]]

#TODO: Feed old context
class Context: 
    
    def __init__(self,monitorType):
        self.dirPath=os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.dirPath)
        self.actionCounter=-1        
        self.artifactsFolder=""
        self.contextFolder=""
        self.monitorType=monitorType
        # Screen space scale
        # 4k(3849*2160) Display resolution is almost twice of HD monitor(1920*966) 
        # Originaly ui elements was created for HD monitor with scale=125%
        # With 4K monitor we need to set scale to 250% and then use 2 as a scale 
        # parameter for Context

        #Click space scale
        #Click space scale coefficient (e.g. 125%)=1/1.25
        #Regular display Display resolution 1920*966 scale 125 = 0.8
        #4k(3849*2160) Display resolution 3840*1948 scale 250 = 0.4
        if monitorType =="HD":
             self.clickScale=1        
            #  self.clickScale=0.8        
             rescale(0.8)
        if monitorType =="4K":
            self.clickScale=1        
            rescale(2) 
        self.prevPicture = None
        self.prevDimensions=None
        self.lastScreenshot=None        
        self.eocrReader = eocr.Reader(['en'])

    def getTargetPath(self):
        trgFolder = os.path.join(os.getcwd(), 'target')
        if self.monitorType =="HD":
             trgFolder = os.path.join(trgFolder, 'hd')
        if self.monitorType =="4K":
            trgFolder = os.path.join(trgFolder, '4k') 
        return trgFolder
       
    def setTest(self,testName): 
        self.actionCounter=-1        
        self.artifactsFolder=os.path.join(self.dirPath, testName)
        self.contextFolder=os.path.join(self.artifactsFolder, 'context')
        os.makedirs(self.artifactsFolder)
        os.makedirs(self.contextFolder)
        print(testName)

    #TODO: scale indent, or size filter 
    def scaleParam(self,param):
        return param
    
    def getDebugSnapshotPath(self):       
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"asnapshot.jpg")
    
    #Writes last full-screen snapshot into the circular buffer with 2 elements
    def update(self):
        self.actionCounter+=1
        snapshot = takeSnapshot(focus["fullScreen"])             
        save_path = os.path.join(self.contextFolder, str(self.actionCounter%2)+".jpg")
        snapshot.save(save_path)
        self.lastScreenshot = cv.imread(save_path)  

    def getLastSnapshotPath(self):
        return os.path.join(self.contextFolder, str(self.actionCounter%2)+".jpg") 
    
    #returns previous and recent screenshots for comparison
    def getSnapshotForDiffPath(self):
        return os.path.join(self.contextFolder, str((self.actionCounter-1)%2)+".jpg"), os.path.join(self.contextFolder, str(self.actionCounter%2)+".jpg")
    
    def getRecognizedTextPath(self):        
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"recognized.txt")
        
    def getDebugAfterActionOutPath(self):
        return os.path.join(self.artifactsFolder, str(self.actionCounter)+"yafterAction.jpg")
    
    def getFindFeaturePath(self):
         return os.path.join(self.contextFolder, "featureFounded.jpg") 
    
    def getDiffFocusPath(self):
         return os.path.join(self.contextFolder, "diffRegion.jpg") 
    
    def writeDebugScreenshot(self,img):
        cv.imwrite(self.getDebugSnapshotPath(),img) 

    def writeDebugScreenshotAfterAction(self,img):
        cv.imwrite(self.getDebugAfterActionOutPath(),img)        

def clickRight(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
 
def clickLeft(x,y):
    pdi.moveTo(x, y)
    pdi.click()
    # win32api.SetCursorPos((x,y))
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0) 
def typeText(text):
    pag.write(text, interval=0.01)      
    # pdi.write(text, interval=0.25)

def lastCursorPosition():
    return win32api.GetCursorPos()

def dragDrop(src,dst,context):    
    pag.moveTo(src[0],src[1]) 
    pag.dragTo(dst[0],dst[1],1, button='left')  

def takeSnapshot(focus):      
    return ImageGrab.grab(bbox=(focus[0], focus[1], focus[2], focus[3])) 
    # take whole screen 
    # snapshot = ImageGrab.grab()
    
#TODO create click scaling routine 
def clikElement(focus, x, y, w, h,clickScale):     
    clickLeft(int(clickScale*(x+focus[0]+w/2)),int(clickScale*(y+focus[1]+h/2)))  

#Detects and returns region of change between the last two screenshots 
#TODO: debug image
def diffFocusContour(focus, minWidth, minHeight, context:Context, ):
    old,new = context.getSnapshotForDiffPath(); 
    oldM = cv.imread(old)
    newM = cv.imread(new)    
    #y:y+h, x:x+w
    cropped_image_old = oldM[focus[1]:focus[3], focus[0]:focus[2]]
    cropped_image_new = newM[focus[1]:focus[3], focus[0]:focus[2]]  
      
    old = cv.cvtColor(cropped_image_old, cv.COLOR_BGR2GRAY)
    new = cv.cvtColor(cropped_image_new, cv.COLOR_BGR2GRAY)   

    diff=cv.subtract(old,new)  
    cv.imwrite("diff.jpg",diff)  
    ret, thresh1 = cv.threshold(diff, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)    
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))      
   
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 7)  
    cv.imwrite("dilation.jpg",dilation)   
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)       
        if (w>minWidth and h>minHeight) :                     
            cv.rectangle(newM, (focus[0]+x, focus[1]+y), (focus[0]+x + w, focus[1]+y + h), (0, 255, 0), 1)  
            cv.imwrite(context.getDiffFocusPath(),newM)                       
            return (focus[0]+x, focus[1]+y, focus[0]+ x + w, focus[1]+y + h)
    return None

def diffFocusContourWindow(focus, minWidth, minHeight, context:Context, ):
    old,new = context.getSnapshotForDiffPath(); 
    oldM = cv.imread(old)
    newM = cv.imread(new)    
    #y:y+h, x:x+w
    cropped_image_old = oldM[focus[1]:focus[3], focus[0]:focus[2]]
    cropped_image_new = newM[focus[1]:focus[3], focus[0]:focus[2]]  
      
    old = cv.cvtColor(cropped_image_old, cv.COLOR_BGR2GRAY)
    new = cv.cvtColor(cropped_image_new, cv.COLOR_BGR2GRAY)   

    (score, diff) = compare_ssim(old, new, full=True)
    diff = (diff * 255).astype("uint8")
    
    # threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(cnts)
    
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)       
        if w>context.scaleParam(minWidth) and h>context.scaleParam(minHeight):                     
            cv.rectangle(newM, (focus[0]+x, focus[1]+y), (focus[0]+x + w, focus[1]+y + h), (0, 255, 0), 1)  
            cv.imwrite(context.getDiffFocusPath(),newM)                       
            return (focus[0]+x, focus[1]+y, focus[0]+ x + w, focus[1]+y + h)
    return None

#detect target location 
def findFeature(focus, target, xOffset, yOffset, context: Context):
    """Find target location in a focus region

    Args:
      focus(focus dictionary element): UI region to search      
      target (string): Valid file name from ".//target//{screenType}//" location 
      xOffset (int): x offset that will be added to returned location 
      yOffset (int): y offset that will be added to returned location 
      context(Context): Current test-run context object  
    Returns:
      Coordinates of a founded target center
    """
    srcImg=cv.imread(context.getLastSnapshotPath())
    trgPath=os.path.join(context.getTargetPath(), target)
    trgImg = cv.imread(trgPath)
    srcImgCropped = srcImg[focus[1]:focus[3], focus[0]:focus[2]]
    result = cv.matchTemplate(srcImgCropped, trgImg, cv.TM_CCOEFF_NORMED)    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)  
    #[width,height]
    trgCenter = [int(max_loc[0]+trgImg.shape[1]/2),int(max_loc[1]+trgImg.shape[0]/2)]    
    cv.circle(srcImg, (focus[0]+trgCenter[0]+context.scaleParam(xOffset), focus[1]+trgCenter[1]+context.scaleParam(yOffset)) , radius=3, color=(0, 0, 255), thickness=-1)
    cv.imwrite(context.getFindFeaturePath(),srcImg)
    return trgCenter[0]+context.scaleParam(xOffset), trgCenter[1]+context.scaleParam(yOffset)   

def prepareSnapshot(img):  
    
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
    return contours

def performSimpleAction(action, focus,coordinates,message=None,context=None):
    lastScreenshot = context.lastScreenshot
    img = lastScreenshot[focus[1]:focus[3], focus[0]:focus[2]]
    debugImage = img.copy()
    context.writeDebugScreenshot(img)
    if coordinates: 
        x = int(context.clickScale*(coordinates[0]+focus[0]))
        y = int(context.clickScale*(coordinates[1]+focus[1]))
        drawX = coordinates[0]
        drawY = coordinates[1]
    match action: 
        case Actions.Click:
                   clickLeft(x,y)    
                   cv.circle(debugImage, (drawX,drawY), radius=7, color=(0, 0, 255), thickness=-1)
                   if debugLog: print(f'performSimpleAction.Click Click X:{x}, Y:{y}')
        case Actions.ClickAndType: 
                   clickLeft(x,y)
                   cv.circle(debugImage, (drawX,drawY), radius=7, color=(0, 0, 255), thickness=-1)
                   if debugLog: print(f'performSimpleAction.ClickAndType Click X:{x}, Y:{y}')          
                   typeText(message) 
        case Actions.Type:
                   typeText(message) 
    context.writeDebugScreenshotAfterAction(debugImage)  

def performTextActionEocr(action, name, focus,context,width_ths=1.0,skip=0):
    actionDone = False 
    where = None   
    file = open(context.getRecognizedTextPath(), "w+")     
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    lastScreenshot = context.lastScreenshot
    #debugImage=lastScreenshot.copy()
    img = lastScreenshot[focus[1]:focus[3], focus[0]:focus[2]]
    debugImage = img.copy()
    context.writeDebugScreenshot(img)
    #TODO make parameter width_ths  
    result = context.eocrReader.readtext(debugImage,width_ths=width_ths)
    # contours = prepareSnapshot(debugImage) 
    for res in result:
        coord=res[0]
        text=res[1]
        conf=res[2]
        # Convert bounding box format
        x = int(min([point[0] for point in coord]))
        y = int(min([point[1] for point in coord]))
        w = int(max([point[0] for point in coord]))-x
        h = int(max([point[1] for point in coord]))-y    
        cv.rectangle(debugImage, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if not (actionDone):  
            if name.lower() in text.lower() and text.lower().startswith(name.lower()) : 
                if bool(skip):
                    skip-=1
                    continue
                match action: #TODO: escape on match (leave as it's for debug purpouse)
                    case Actions.Click:
                        clikElement(focus, x, y, w, h, context.clickScale)                 
                        cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)
                        actionDone=True
                    case Actions.DoubleClick:
                        clikElement(focus, x, y, w, h, context.clickScale)  
                        clikElement(focus, x, y, w, h, context.clickScale)               
                        cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)  
                        actionDone=True                  
                    case Actions.FailIfExist:                          
                        print("Failed!")
                        actionDone=True
                    case Actions.PassedIfExist:
                        print("Passed!")
                        actionDone=True
                    case Actions.FindClickLocation:                        
                        where = (int(x+focus[0]+w/2),int(y+focus[1]+h/2))
                        if debugLog: print(f"performTextActionEocr.FindClickLocation Click X:{where[0]}, Y:{where[1]}")
                        cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1) 
                        actionDone=True
        if text:
            file.write(text+'\n') 
    file.close
    context.writeDebugScreenshotAfterAction(debugImage) 
    if not (actionDone):
        print("Text not found. Please check your test. Interrupting")
        sys.exit("Error message")
    if action == Actions.FindClickLocation: return where  
    


def performTextAction(action, name, focus,psm,context):
    actionDone = False    
    file = open(context.getRecognizedTextPath(), "w+")     
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    lastScreenshot = context.lastScreenshot
    #debugImage=lastScreenshot.copy()
    img = lastScreenshot[focus[1]:focus[3], focus[0]:focus[2]]
    debugImage = img.copy()
    context.writeDebugScreenshot(img)
    contours = prepareSnapshot(debugImage)        
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)        
        # Cropping the text block for giving input to OCR
        cropped = img[y:y + h, x:x + w]
        # Drawing a rectangle on copied image
        rect = cv.rectangle(debugImage, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Check what we feed into pytesseract <<DEBUG_FUNCTION>>
        #cv.imwrite('cropped %d.jpg' %x ,cropped)  
        # Apply OCR on the cropped image
        # page segmentation modes (psm) is set
        text = pytesseract.image_to_string(cropped,lang='eng',config=psm)
        if not (actionDone):  
            if name in text : 
                match action: #TODO: escape on match (leave as it's for debug purpouse)
                    case Actions.Click:
                        clikElement(focus, x, y, w, h, context.clickScale)                 
                        cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)
                        actionDone=True
                    case Actions.DoubleClick:
                        clikElement(focus, x, y, w, h, context.clickScale)  
                        clikElement(focus, x, y, w, h, context.clickScale)               
                        cv.circle(debugImage, (x+int(w/2),y+int(h/2)), radius=7, color=(0, 0, 255), thickness=-1)  
                        actionDone=True                  
                    case Actions.FailIfExist:                          
                        print("Failed!")
                        actionDone=True
                    case Actions.PassedIfExist:
                        print("Passed!")
                        actionDone=True
        if text:
            file.write(text)            
    # Close the file    
    file.close
    context.writeDebugScreenshotAfterAction(debugImage)
    if not (actionDone):
        print("Text not found. Please check your test. Interrupting")
        # sys.exit("Error message")    

#Must call this method to update context
#Also, between action we need to wait a little to let IDE update it's GUI    
def updateContext(waitTime):
    time.sleep(waitTime)
    context.update()  

#Define all the tests here

    
# Demonstarete main functionality (contour drawing, element clicking) 
def sampleTest(context): 
    context.setTest("sampleTest") 
    updateContext(0)    
    performTextAction(Actions.Click, "File", focus["mainMenu"],"--psm 8",context)
    #restore for a next test
    performTextAction(Actions.Click, "File", focus["mainMenu"],"--psm 8",context)
    
def ATcreateNewProject(context, projName):
    updateContext(0)  
    performTextActionEocr(Actions.Click, "File", focus["mainMenu"],context,0.5)
    updateContext(0.5)    
    performTextActionEocr(Actions.Click, "New", focus["mainMenuItems"],context)
    updateContext(2)    
    diffFocus = diffFocusContour(focus["mainMenuItems"], 200, 0, context)    
    performTextActionEocr(Actions.Click, projName, diffFocus, context, 1.1)

def ATgoToMenuPath(context,path):
    updateContext(0)
    performTextActionEocr(Actions.Click, path[0], focus["mainMenu"],context,0.5)
    for item in path[1:]:
         updateContext(1)
         diffFocus = diffFocusContour(focus["mainMenuItems"], 150, 0, context)
         performTextActionEocr(Actions.Click, item, diffFocus, context, 1.1) 


#TODO: add methot for correct focus gaining
def ATgainFocus(context):
    updateContext(1)
    performSimpleAction(Actions.Click,focus["fullScreen"],(600, 600),None,context)
   
def ATaddComponent(context,component):
    """Adds component to the form by double clicking on a corresponding entry from the palette

    Prereqisits: project is opened palette is visible

    Args:
      context(Context): Current test-run context object      
      component (string): Name of component to drop into the form  

    Returns:
      None
    """   
    updateContext(0)
    searchField = findFeature(focus["palette"],'magnifingGlass.pngt',-40,0,context)
    performSimpleAction(Actions.ClickAndType, focus["palette"],searchField, component,context=context)
    updateContext(0.5)
    # performTextAction(Actions.DoubleClick, "TButton", focus["palette"], "--psm 7",context)
    performTextActionEocr(Actions.DoubleClick, component, focus["palette"],context,skip=1)
# Prereqisits: project is opened palette is visible   
def ATdropComponentNear(context,component,near):    
    updateContext(0)
    searchField = findFeature(focus["palette"],'magnifingGlass.pngt',-40,0,context)
    performSimpleAction(Actions.ClickAndType, focus["palette"],searchField, component,context=context)
    updateContext(0.5)
    src=performTextActionEocr(Actions.FindClickLocation, component, focus["palette"],context,skip=1)    
    dst=performTextActionEocr(Actions.FindClickLocation, near, focus["editor"],context)
    dragDrop(src,(dst[0]+100, dst[1]),context)  

def ATCloseAll(context): 
    updateContext(0.5)
    performTextActionEocr(Actions.Click, "File", focus["mainMenu"],context)
    updateContext(0.5) 
    fileMenu = diffFocusContour(focus["fullScreen"], 200, 0, context)      
    performTextActionEocr(Actions.Click, "Close AII", fileMenu,context) 
    updateContext(0)
    updateContext(1)
    fileMenu = diffFocusContour(focus["fullScreen"], 200, 0, context) 
    dialogRegion = diffFocusContourWindow(focus["fullScreen"], 200, 50, context)
    if(dialogRegion):        
        confirmDialog = performTextActionEocr(Actions.FindClickLocation, "Confirm", dialogRegion,context)
        if(confirmDialog):
            # closeButton = findFeature(dialogRegion,'closeApp.pngt',0,0,context)
            # performSimpleAction(Actions.Click, dialogRegion,closeButton, None,context)
            performTextActionEocr(Actions.Click, "No", dialogRegion,context)
        

def ATplaceBreakpoint(context, text):
    updateContext(0)
    where = performTextActionEocr(Actions.FindClickLocation,text,focus["editor"],context)
    performSimpleAction(Actions.Click,focus["fullScreen"],(focus["editor"][0],where[1]),None,context)




def ATinputOneButtonDelphiText(context):
    updateContext(0)
    performSimpleAction(Actions.Type, focus["editor"],None, "var",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "text: UnicodeString;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "num:integer;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "val:double;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "begin",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "num:=0;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "val:=3.14;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "text:= Edit1.Text;",context)
    pag.press('enter')
    performSimpleAction(Actions.Type, focus["editor"],None, "showmessage(text+' '+num.tostring+' '+val.tostring+'   '+'GreatSuccess');",context)

def DelphiVCLOneButtonApp(context):
    context.setTest("DelphiVCLOneButtonApp")
    ATcreateNewProject(context,"Windows VCL Application - Delphi")
    updateContext(2)
    ATaddComponent(context,"TButton")
    ATdropComponentNear(context,"TEdit","Button1")
    updateContext(0)
    performTextActionEocr(Actions.DoubleClick, "Button1", focus["editor"],context)
    for i in range(6):
        pag.press('backspace')
    ATinputOneButtonDelphiText(context)
    updateContext(1)
    runButton = findFeature(focus["header"],'runWithoutDebugging.pngt',0,0,context)
    performSimpleAction(Actions.Click, focus["header"],runButton, None,context)
    updateContext(2)
    appRegion = diffFocusContourWindow(focus["fullScreen"], 400, 200, context) 
    performTextActionEocr(Actions.Click, "Button1", appRegion,context)
    updateContext(1)
    dialogRegion = diffFocusContourWindow(focus["fullScreen"], 200, 50, context)
    performTextActionEocr(Actions.PassedIfExist, "GreatSuccess", dialogRegion,context)    
    performTextActionEocr(Actions.Click, "OK", dialogRegion,context)
    updateContext(1)
    closeApp = findFeature(appRegion,'closeApp.pngt',0,0,context)
    performSimpleAction(Actions.Click, appRegion,closeApp, None,context)    
    ATgainFocus(context)
    ATCloseAll(context)

def test(context):
    context.setTest("PlaceBreakpoint")
    # ATinputOneButtonDelphiText(context)    
    ATplaceBreakpoint(context,"num:=0;")
    # ATgoToMenuPath(context,("Project","Options")) 
    # updateContext(0)
    # updateContext(3)
    # dialogRegion = diffFocusContourWindow(focus["fullScreen"], 700, 400, context)
    # performTextActionEocr(Actions.Click, "Delphi compiler", dialogRegion,context)  
    
    

# def dragDropTest(context): 
#     context.setTest("dragDropTest") 
#     updateContext(0)
#     src=performTextActionEocr(Actions.FindClickLocation, "Button1", focus["editor"],context)
#     dragDrop(src,(src[0]+50, src[1]),context)
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  


context = Context("HD")
# context = Context("4K")   
#Test to run:
#time.sleep(4) 
# ATcreateNewProject(context,"Windows VCL Application - Delphi")
# DelphiVCLOneButtonApp(context)
test(context)
# Diag:
# draw_rectangles("C:\\ocr\\OCR_Tester\\ref.png", focus, output_path='output.png')
# sampleTest(context)
# chekBitmapStyleDesigner(context)
# createNewVCLProject(context)
# ATfindComponent(context)
# dragDrop((1459,773),(1559,773))
# searchField = findFeature(focus["palette"],'magnifingGlass.pngt',-40,0,context)

