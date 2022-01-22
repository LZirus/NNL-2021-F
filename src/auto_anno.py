from tkinter import filedialog as fd
import cv2
import os
import re
from PIL import Image


mask = False
flipped = False

#dst_path = fd.askdirectory()
dst_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/image_dataset/test"

done = []
valid_images = [".jpg",".png",".jpeg",".JPG"]
for file in os.listdir(dst_path):
    end = os.path.splitext(file)[1]
    if end.lower() not in valid_images:
        continue
    if len(file.split('_')) > 2 :
        num = re.findall("\d+", file.split('_')[1])[0]
        done.append(num)

print("saving to", dst_path)

#src_path = fd.askdirectory()
src_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/image_dataset/missing"
print(src_path)


# get all file-names of image-files in the same folder
imgs = []
valid_images = [".jpg",".png",".jpeg",".JPG"]
for file in os.listdir(src_path):
    end = os.path.splitext(file)[1]
    if end.lower() not in valid_images:
        continue
    if len(file.split('_')) < 2 or not re.findall("\d+", file.split('_')[1])[0] in done:
        imgs.append(os.path.join(src_path,file))

print(len(imgs), "images found")



# https://medium.com/@saurabh.shaligram/face-mask-detection-simple-opencv-based-program-417bbcf0abd8
# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

cascPath = "haarcascade_frontalface_default.xml" 

faceCascade = cv2.CascadeClassifier(cascPath)

for img in imgs :
    image = cv2.imread(img)
    image_crop = Image.open(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(100, 100)
    )
    
    faces_bw = faceCascade.detectMultiScale(
        black_and_white, 
        scaleFactor=1.1,
        minNeighbors=4,
        )

    print("Found {0} faces!".format(len(faces)))
    print("\tFound {0} 'mask' faces!".format(len(faces_bw)))
    
    
    if not mask:
        if len(faces) > 0 :
            (x, y, w, h) = faces[0]
        else:
            continue
    else :
        if len(faces_bw) > 0 :
            (x, y, w, h) = faces_bw[0]
        else:
            continue
    
    if flipped : image_crop = image_crop.rotate(180)
    cropped = image_crop.crop((x, y, (x+w), (y+h)))
    path = dst_path+os.sep+os.path.basename(img)+"_"+str((x,y,w,h))+".png"
    
    if not os.path.exists(path):
        cropped.save(path, "PNG")
    else:
        print("found")