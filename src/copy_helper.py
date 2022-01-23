from tkinter import filedialog as fd
import os
import re
import shutil, os

#src_path = fd.askdirectory()
comp_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/image_dataset/test"
print(comp_path)

# get all file-names of image-files in the same folder
imgs = []
valid_images = [".jpg",".png",".jpeg",".JPG"]
for file in os.listdir(comp_path):
    end = os.path.splitext(file)[1]
    if end.lower() not in valid_images:
        continue
    num = re.findall("\d+", file.split('_')[1])[0]
    imgs.append(num)

print(len(imgs), "images found")
print(imgs[:10])

original_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/image_dataset/missing"

# get all file-names of image-files in the same folder
missing = []
for file in os.listdir(original_path):
    end = os.path.splitext(file)[1]
    if end.lower() not in valid_images:
        continue
    num = re.findall("\d+", file)[0]
    if not (num in imgs) :
        missing.append(file)

print(len(missing), "images missing")
print(missing)

missing_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/image_dataset/new_missing"

for file in missing:
    shutil.copy(original_path + os.sep + file, missing_path)