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

class Actions(Enum):
    Click = 1
    FailIfExist = 2
    


# dictionary of the main UI elements (left,top,right,bottom)
ideUi = {
    "mainMenu": [4,35,638,70],
    "mainMenuItems": [4,66,931,619],
    "centrRegion": [600,400, 1700, 800],
    "objectInspector": [256,256,512,512]   
}


def clickRight(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
 
def clickLeft(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0) 


def takeSnapshot(uiElement):
    snapshot = ImageGrab.grab(bbox=(uiElement[0], uiElement[1], uiElement[2], uiElement[3]))
    # take whole screen 
    # snapshot = ImageGrab.grab()
    save_path = "sample.jpg"
    snapshot.save(save_path)
 
def clikElement(uiElement, x, y, w, h):
    #print ("Match found: "+str(int(0.8*(x+uiElement[0]+w/2)))+" "+str(int(0.8*(y+uiElement[1]+h/2))))
    #Scaling coefficient (e.g. 125%) scrren_x_dim/(img_res*1.25) 
    clickLeft(int(0.8*(x+uiElement[0]+w/2)),int(0.8*(y+uiElement[1]+h/2)))  

def prepareSnapshot(uiElement):
    takeSnapshot(uiElement)
    # Read image from which text needs to be extracted
    img = cv2.imread("sample.jpg")
 
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
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
    
    
    # Creating a copy of image
    im2 = img.copy()
    
    # A text file is created and flushed
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()  
    
    
    return contours, im2, file 

def performAction(action, contours, name, image, uiElement,psm):
    # Open the file in append mode
    file = open("recognized.txt", "a")
    im2 = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]
        # Apply OCR on the cropped image
        # page segmentation modes (psm) is set
        text = pytesseract.image_to_string(cropped,lang='eng',config=psm)  
        if name in text : 
            match action:
                case Actions.Click:
                    clikElement(uiElement, x, y, w, h)
                case Actions.FailIfExist:                          
                    print("Failed!")                   
        if text:
            file.write(text)            
    # Close the file
    file.close
    cv2.imwrite('out.jpg',im2)
    
    
# Demonstarete main functionality (contour drawing, element clicking) 
def sampleTest(): 
    contours, im2, file = prepareSnapshot(ideUi["mainMenu"])
    file = open("recognized.txt", "w+")
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)       
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w] 
        # Check what we feed into pytesseract
        # cv2.imwrite('cropped %d.jpg' %x ,cropped)    
        
        # Apply OCR on the cropped image
        # page segmentation modes (psm) is set
        text = pytesseract.image_to_string(cropped,lang='eng',config='--psm 8')  
        if "File" in text :       
            clikElement(ideUi["mainMenu"], x, y, w, h)              
        # Appending the text into file
        if text:
            file.write(text)            
    # Close the file
    file.close
    cv2.imwrite('out.jpg',im2)
    

def chekBitmapStyleDesigner():
    print("chekBitmapStyleDesigner test:")
    contours, im2, file = prepareSnapshot(ideUi["mainMenu"])
    performAction(Actions.Click, contours, "Tools", im2, ideUi["mainMenu"],"--psm 8")
    
    contours, im2, file = prepareSnapshot(ideUi["mainMenuItems"])
    performAction(Actions.Click, contours, "Bitmap", im2, ideUi["mainMenuItems"], "--psm 7")
    
    contours, im2, file = prepareSnapshot(ideUi["centrRegion"])
    performAction(Actions.FailIfExist, contours, "Access", im2, ideUi["centrRegion"], "--psm 11")
    
    
    
    
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'    

time.sleep(4) 
#sampleTest()
chekBitmapStyleDesigner()
