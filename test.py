# Import required packages
import cv2
import pytesseract
import win32api, win32con
import time
from PIL import ImageGrab
import fnmatch


def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
    


def takeSnapshot():
    snapshot = ImageGrab.grab()
    save_path = "sample.jpg"
    snapshot.save(save_path)
    


time.sleep(4)
takeSnapshot() 
# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
 
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
 
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if(w>100):
        continue
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
     
    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
     
    # Open the file in append mode
    file = open("recognized.txt", "a")
     
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)  
    if "Project" in text :
        print ("Match found: "+str(x+w/2)+" "+str(y+h/2))
        #Scaling coefficient (e.g. 125%) scrren_x_dim/(img_res*1.25) 
        click(int(0.8*(x+w/2)),int(0.8*(y+h/2)))
        click(int(0.8*(x+w/2)),int(0.8*(y+h/2)))
    # Appending the text into file
    file.write(text)
    file.write("\n")
     
    # Close the file
    file.close
cv2.imwrite('out.jpg',im2)