from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox

import cv2
from PIL import ImageTk, Image, ImageDraw

from shapely.geometry import box, Point

import json
import csv
import os
from os.path import exists

from re import compile, split
import traceback

import pandas as pd
import numpy as np

# import other self-writen script: mask_classifier.py
import mask_classifier

# ======== Global variables ========
# UI-related globals
root = Tk()
window = None
image_panel = None
size = 1000

# images save-points
image = None
image_save = None
image_clean = None

# main save-dictionary
images = {}

# globals for active image
active_name = ""
rectangles = []
rect_to_category = []
categories = ["None", "mask", "no_mask", "op_mask", "ffp2"]
dest_paths = ["None"]
act_src = ""
act_dst = 0

# needed for some errors with flipped pictures
flipped = False

# Config
DEBUG = True

# Class Window:
# - initializing top menu
# - taking care of popup fireing and value return
class Window(Frame):
    def __init__(self, master=None):
        # init tkinter frame and keep reference
        Frame.__init__(self, master)
        self.master = master

        # --- top menu - setup ---
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # init tabs
        fileMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=fileMenu)
        
        annoMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Annotations", menu=annoMenu)

        categoryMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Categories", menu=categoryMenu)
        
        learningMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Machine Learning", menu=learningMenu)
        
        modelMenu = Menu(learningMenu, tearoff=0)

        # add all menu-elements and bind functions to buttons
        # -> file menu for opening Images and maintaining save-locations of processed images
        fileMenu.add_command(label="Open Image", command=select_image)
        fileMenu.add_command(label="Load next Image", command=next_image)
        fileMenu.add_separator()
        fileMenu.add_command(label="Change this Save-Location", command=change_dst)
        fileMenu.add_command(label="Replace all Save-Locations", command=replace_dst)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        
        # -> annotation menu
        annoMenu.add_command(label="Save Annotations", command=save_annos)
        annoMenu.add_command(label="Import Annotations", command=load_annos)
        annoMenu.add_command(label="View Annotations", command=view_annos)

        # -> category menu
        categoryMenu.add_command(label="Add", command=addCategory)
        categoryMenu.add_command(label="Replace", command=replaceCategory)
        categoryMenu.add_separator()
        categoryMenu.add_command(label="Show All", command=listCategories)
        categoryMenu.add_separator()
        categoryMenu.add_command(label="Import", command=importCategories)
        categoryMenu.add_command(label="Export Categories as JSON", command=exportCategories)
        categoryMenu.add_command(label="Export Categories as CSV", command=exportCSV)
        categoryMenu.add_command(label="Export Categories as XLSX", command=exportXLSX)
        
        # -> machine learning menu
        learningMenu.add_command(label="Crop & Save Dataset", command=crop_and_save)
        
        modelMenu.add_command(label="Train model", command=trainModel)
        modelMenu.add_command(label="Load model", command=loadModel)
        modelMenu.add_command(label="Save model", command=saveModel)
        
        learningMenu.add_cascade(label="ML Model", menu=modelMenu)
        learningMenu.add_command(label="Classify current Image", command=classifyImage)
        learningMenu.add_command(label="Open Live-Classifier", command=liveClassify)
        
        
    # create PopUp-Window and wait for user
    def popup(self, text, type):
        self.w = popupWindow(self.master, text, type)
        self.master.wait_window(self.w.top)

    # creating PopUps with more settings
    def popup_select(self, text, type, rect, i):
        self.w = popupWindow(self.master, text, type, rect, i)
        self.master.wait_window(self.w.top)
    
    def popup_stdout(self):
        self.w = popupWindow(self.master, "text", "stdout")

    # get return values from PopUp if only one value needed
    def entryValue(self):
        return self.w.value

    # get return values from PopUp if two values are needed
    def entryValues(self):
        return self.w.value0, self.w.value1

    # function to exit program 
    def exitProgram(self):
        exit()

# class popupWindow to setup different types of popups, initiated by the Window-class
# Inspired by https://stackoverflow.com/questions/10020885/creating-a-popup-message-box-with-an-entry-field#10021242
class popupWindow(object):
    def __init__(self, master, text, type, rect=None, i=0):
        top = self.top = Toplevel(master)

        # setting popup size (if showing annotations more space is needed)
        if type == "all":
            top.geometry(str(700)+"x"+str(150))
            self.list_all(top, text)
        else:
            top.geometry(str(int((16/9)*150))+"x"+str(150))

        # switching between the functions to create PopUps
        if type == "one":
            self.oneInput(top, text)
        elif type == "two":
            self.twoInput(top, text)
        elif type == "list":
            self.list(top, text)
        elif type == "select":
            self.rect = rect
            self.select(top, text, i)
        elif type == "dst":
            self.pathSelect(top, text)
        elif type == "rep":
            self.pathReplace(top, text)
        elif type == "stdout":
            self.stdout_stuff(top)

    def stdout_stuff(self, top):
        top.wm_title("Output")
        top.geometry(str(int((16/9)*500))+"x"+str(500))
        
        text = ScrolledText(top)
        text.pack(expand=True, fill='both')
        
        # redirect stdout
        redir = RedirectText(text)
        sys.stdout = redir
        redir.write("This will take a while\r")

    # Single Text-Input Popup
    def oneInput(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()
        
        self.e = Entry(top)
        self.e.pack()
        
        self.b = Button(top, text='Ok', command=lambda: self.cleanup("one"))
        self.b.pack()

    # Select-Menu and Text-Input returning two values
    def twoInput(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        # Dropdown-Menu
        self.active = StringVar()
        self.active.set(categories[0])
        self.drop = OptionMenu(top, self.active, *categories)
        self.drop.pack()

        self.e1 = Entry(top)
        self.e1.pack()

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("two"))
        self.b.pack()

    # Callback for Folder-Selection-Button
    def selectFolder(self):
        global dest_paths
        
        # get dir-path with selector
        path = fd.askdirectory()
        
        # add path to list if not in there
        if path not in dest_paths:
            dest_paths.append(path)
            self.active_path.set(path)  # refresh active-menu-item
    
    # Callback for Folder-Selection-Button
    def selectFolderRep(self):
        # get dir-path with selector
        path = fd.askdirectory()
        
        # refresh label for folder-change-popup
        self.rep.config(text=path)
    
    # Popup to replace existing dst-path
    def pathReplace(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        # Dropdown-Menu
        self.active = StringVar()
        self.active.set(dest_paths[0])
        self.drop = OptionMenu(top, self.active, *dest_paths)
        self.drop.pack()
        self.drop.config(width=13)

        # label to display selected path
        self.rep = Label(top, text='None')
        self.rep.pack()
        
        # Source-Select Button
        self.src = Button(top, text='Replace with', command=self.selectFolderRep)
        self.src.pack()
        self.src.config(width=13)

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("rep"))
        self.b.pack()

    # Select dst-path or add new one
    def pathSelect(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        # Dropdown-Menu
        self.active_path = StringVar()
        self.active_path.set(dest_paths[act_dst if act_dst != None else 0])
        self.drop = OptionMenu(top, self.active_path, *dest_paths)
        self.drop.pack(side=LEFT)
        self.drop.config(width=13)

        # Source-Select Button
        self.src = Button(top, text='Source', command=self.selectFolder)
        self.src.pack(side=RIGHT)
        self.src.config(width=13)

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("dst"))
        self.b.pack(side=BOTTOM)

    # popup to list existing categories
    # Inspired by https://www.geeksforgeeks.org/scrollable-listbox-in-python-tkinter/
    def list(self, top, intext):
        top.wm_title("Info")

        self.l = Label(top, text=intext)
        self.l.pack()

        # create listbox
        listbox = Listbox(top)
        listbox.pack(side=LEFT, fill=BOTH)

        # create scrollbar
        scrollbar = Scrollbar(top)
        scrollbar.pack(side=RIGHT, fill=BOTH)

        # add values to list
        for values in categories:
            listbox.insert(END, values)

        # conntect scrollbar to list
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

    # callback function for image list active-select
    def image_callback(self, _):

        # get active selections
        items = self.image_listbox.curselection()

        # if more or less than one is selected no change
        if len(items) != 1:
            return

        # reset rectangles list
        self.rectangles_listbox.delete(0, END)

        # get rectangles of selected image
        self.active_img = self.image_listbox.get(items[0])

        # add rectangles to list
        for rect in images[self.active_img]["rectangles"]:
            self.rectangles_listbox.insert(END, str(rect.bounds))

    # callback function for rectangle list active-select
    def rect_callback(self, _):

        # get active selections
        items = self.rectangles_listbox.curselection()

        # if more or less than one is selected no change
        if len(items) != 1:
            return

        # reset info-checkbox
        self.info_listbox.delete(0, END)

        # get selected rectangle
        active = self.rectangles_listbox.get(items[0])

        # print infos in info-list
        for rect in images[self.active_img]['rectangles']:
            if str(rect.bounds) == active:
                self.info_listbox.insert(END, "Area: " + str(rect.area))
                self.info_listbox.insert(END, "Bounds: " + str(rect.bounds))
                
                rect_i = images[self.active_img]['rectangles'].index(rect)
                cat_i = images[self.active_img]['rect_to_category'][rect_i]
                cat = categories[cat_i]
                self.info_listbox.insert(END, "Category: " + cat)

    # PopUp displaying all Annotations in three Lists
    def list_all(self, top, intext):
        top.wm_title("Info")

        l = Label(top, text=intext)
        l.pack()

        # create listboxes with fitting size
        max_size = 35
        self.image_listbox = Listbox(top, width=max_size)
        self.rectangles_listbox = Listbox(top, width=max_size)
        self.info_listbox = Listbox(top, width=max_size+5)

        # create scrollbars
        img_scrollbar = Scrollbar(top)
        rect_scrollbar = Scrollbar(top)

        # place listboxes
        self.image_listbox.pack(side=LEFT, fill=BOTH)
        img_scrollbar.pack(side=LEFT, fill=BOTH)

        self.rectangles_listbox.pack(side=LEFT, fill=BOTH)
        rect_scrollbar.pack(side=LEFT, fill=BOTH)

        self.info_listbox.pack(side=LEFT, fill=BOTH)

        # bind selection of list-elements to callback functions
        self.image_listbox.bind("<<ListboxSelect>>", self.image_callback)
        self.rectangles_listbox.bind("<<ListboxSelect>>", self.rect_callback)

        # add all images to image-list
        for img in images.keys():
            self.image_listbox.insert(END, img)

        # connect scrollbars
        self.image_listbox.config(yscrollcommand=img_scrollbar.set)
        img_scrollbar.config(command=self.image_listbox.yview)
        
        self.rectangles_listbox.config(yscrollcommand=rect_scrollbar.set)
        rect_scrollbar.config(command=self.image_listbox.yview)

    # PopUp displaying rectangle info and allowing category choosing/change
    def select(self, top, intext, i):
        top.wm_title("Info")

        self.l = Label(top, text=intext)
        self.l.pack()

        # infos to display
        text = "Area: " + str(self.rect.area) + \
            "\nBounds: " + str(self.rect.bounds)

        self.info = Label(top, text=text)
        self.info.pack()

        # dropdown menu, default category is given by parameter
        self.active = StringVar()
        self.active.set(categories[i])
        self.drop = OptionMenu(top, self.active, *categories)
        self.drop.pack()

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("select"))
        self.b.pack()

    # cleanup function, that sets the return values for class window to access
    def cleanup(self, type):
        if type == "one":
            self.value = self.e.get()
        elif type == "two":
            self.value0 = self.active.get()
            self.value1 = self.e1.get()
        elif type == "select":
            self.value = self.active.get()
        elif type == "rep":
            self.value0 = self.active.get()
            self.value1 = self.rep.cget('text')
        elif type == "dst":
            self.value = self.active_path.get()

        # destroy popup
        self.top.destroy()

# ========== Link to Machine Learning ==========

from tkinter.scrolledtext import ScrolledText
import sys

ml_model = None
labels = None
ml_size = None

def trainModel():
    global ml_model, labels, ml_size
    
    ml_model, ml_size = mask_classifier.select_model("basic_model")
    epochs = 10
        
    window.popup_stdout()
    
    train_ds, val_ds, labels, y_test, x_test = mask_classifier.load_dataset(imgs_path=fd.askdirectory())
    mask_classifier.train_model(ml_model, train_ds, val_ds)
    
    sys.stdout = sys.__stdout__
    return

def loadModel():
    global ml_model, labels, ml_size
    
    path = select_file(False, "ml", "model.h5", title='Load ML Model')
    
    with open(path+"_labels.csv", 'r') as openfile:
        data = list(csv.reader(openfile))
        labels = data[0]
        ml_size = (int(data[1][0]),int(data[1][1]))
    ml_model = mask_classifier.load_model_good(path)
    
    if DEBUG: print(ml_model)
    if DEBUG: print(labels)
    if DEBUG: print(ml_size)
    
    return

def saveModel():
    global ml_model
    if not ml_model:
        return
    
    path = select_file(True, "ml", "model.h5", title='Save ML Model')
    
    with open(path+"_labels.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(labels)
        writer.writerow([ml_size[0], ml_size[1]])
    
    mask_classifier.save_model(path, ml_model)
    return

def classifyImage():
    global ml_model, ml_size
    if not ml_model:
        return
    
    for rect in rectangles:
        i = rectangles.index(rect)
        cat = categories[rect_to_category[i]]
        
        if cat != "None":
            continue
        
        bounds = rect.bounds
        if DEBUG: print(bounds)

        height = int(bounds[3] - bounds[1])
        width = int(bounds[2] - bounds[0])
        length = max(height, width)
        diff = (length - min(height, width))/2
        
        left = (bounds[0] - diff) if height > width else bounds[0]
        up = bounds[1] if height > width else (bounds[1] - diff)
        right = (bounds[0] - diff) + length if height > width else (bounds[0] + length)
        bottom = (bounds[1] + length) if height > width else ((bounds[1] - diff) + length )
        
        cropped = image_clean.crop((left, up, right, bottom))
        cropped = cropped.resize(ml_size, Image.ANTIALIAS)
        # img_arr = np.array(cropped)
        # img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        cropped.save("act_img.png", "PNG")
        label = mask_classifier.predict('category', img_path="act_img.png", model=ml_model, labels=labels, img_size=ml_size)
        
        if DEBUG: print(label)
        rect_to_category[i] = categories.index(label)
        messagebox.showinfo('Result',str(bounds)+':\nClassified with '+label+"\nLabel was updated")
        #os.remove("act_img.png")
        
        # call classify with img_cv image

def liveClassify():
    if not ml_model:
        return
    # call live classify
    mask_classifier.predict('live_detection', model=ml_model)
    return

# https://www.blog.pythonlibrary.org/2014/07/14/tkinter-redirecting-stdout-stderr/
class RedirectText(object):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, text_ctrl):
        """Constructor"""
        self.output = text_ctrl
        self.text = ""
        
    #----------------------------------------------------------------------
    def write(self, string):
        """"""
        for c in string:
            if c == "\r":
                self.output.insert(END, self.text)
                self.text = ""
            if c != "\b":
                self.text = self.text + c
        self.output.insert(END, self.text)
        self.text = ""
    
    def flush(self):
        return

# ==============================================


# fire PopUp to add a Category
def addCategory():
    try:
        window.popup("Add Category:", "one")
        if window.entryValue() in categories:
            return
        categories.append(window.entryValue())
    except:
        return


# fire popup to replace a category
def replaceCategory():
    try:
        window.popup("Replace Category:", "two")
        c1, c2 = window.entryValues()
        if not c1 in categories:    # only replace existing ones
            return
        categories[categories.index(c1)] = c2
    except:
        return


# fire popup to list all annotations
def listCategories():
    window.popup("All Annotations:", "list")
    return


def importCategories():
    global categories
    # Opening JSON file
    with open(select_file(False, "json", "categories.json", title='Load Categories', initialdir='~/'), 'r') as openfile:
        if openfile.name.endswith('.csv'):
            data = list(csv.reader(openfile))
            categories = data[0]
        elif openfile.name.endswith('.json'):
            json_object = json.load(openfile)
            categories = json_object["categories"]
    return


def exportCategories():
    # Data to be written
    dic = {"categories": categories}
    # Writing to sample.json
    with open(select_file(True, "json", "categories.json", title='Save Categories', initialdir='~/categories.json'), "w") as outfile:
        json.dump(dic, outfile)
    return

def exportCSV():
    # Writing to csv file
    with open(select_file(True, "csv", "categories.csv", title='Save Categories as CSV', initialdir='~/categories.csv'), "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(categories)
    return

def exportXLSX():
    # Writing to csv file
    df = pd.Series(categories)

    with open(select_file(True, "xlsx", "categories.xlsx", title='Save Categories as XLSX', initialdir='~/categories.xlsx'), "w") as outfile:
        
        #writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
        df.to_excel("C://Users/Julie/OneDrive/Bilder/categories.xlsx")
        #writer.save()
    return


# function to save all annotations and important information
def save_annos():
    global images

    # save data for current image in dictionary
    if active_name != "":
        images[active_name] = {'rectangles': rectangles.copy(),
                               'rect_to_category': rect_to_category.copy(),
                               'dst': act_dst if act_dst != None else 0,
                               'src': act_src}

    # create copy of dic to change data to readable format
    save = {"categories": categories,
            "images": {}, 'destinations': dest_paths}
    
    # convert rectangle array to string array and add all data to save-dic
    for key in images :
        save['images'][key] = {'rectangles': [rect.bounds for rect in images[key]['rectangles']],
                               'rect_to_category': images[key]['rect_to_category'].copy(),
                               'dst': images[key]['dst'],
                               'src': images[key]['src']}

    # Writing to .json file selected by fileselector
    with open(select_file(True, "json", "annotations.json", title='Save Annotations', initialdir='~/annotations.json'), "w") as outfile:
        json.dump(save, outfile)
    return


# function to load annotation and additional data from file
def load_annos():
    global images, categories, image_save, rect_to_category, rectangles, dest_paths
    
    # Opening JSON file
    with open(select_file(False, "json", "annotations.json", title='Load Annotations', initialdir='~/info_uca/annotations.json'), 'r') as openfile:
        loaded = json.load(openfile)

    # copy data to work-lists and dictionary
    images = loaded['images'].copy()
    categories = loaded['categories'].copy()
    dest_paths = loaded['destinations'].copy()

    # transform string data of rectangles to objects
    for key in loaded['images']:
        images[key]['rectangles'] = [
            box(rect[0], rect[1], rect[2], rect[3]) for rect in images[key]['rectangles']]

    # refresh current image if any and draw rectangles
    if active_name in images:
        rectangles = images[active_name]['rectangles']
        rect_to_category = images[active_name]['rect_to_category']
        for rect in rectangles:
            bounds = rect.bounds
            draw_rectangle((bounds[0], bounds[1]),
                           (bounds[2], bounds[3]), (0, 0, 255))
            image_save = image.copy()

    return


# fire popup to view all made annotations
def view_annos():
    if active_name != "":
        images[active_name] = {'rectangles': rectangles.copy(),
                               'rect_to_category': rect_to_category.copy(),
                               'dst': act_dst if act_dst != None else 0,
                               'src': act_src}
    window.popup("Categories:", "all")
    return


# replace a destination-path from the list (for copying annotations to different computer)
def replace_dst():
    global dest_paths
    
    # get values from popup
    try:
        window.popup("Replace destination path:", 'rep')
        c1, c2 = window.entryValues()
    except:
        return
    
    # check if values exist and are correct
    if c2 == None : return
    if not c1 in dest_paths : return
    if not os.path.isdir(c2) : return

    # update path in dest_path array
    dest_paths[dest_paths.index(c1)] = c2
    
    # resave active image, all other images will need to be opened again, as src path can not automatically be updated
    if act_dst != None and dest_paths[act_dst] != 'None':
        dst_path = dest_paths[act_dst] + os.sep + active_name[:active_name.rindex('.')] + '.png'
        if not exists(dst_path):
            image_clean.save(dst_path, "PNG")


# change destination only of the active image
def change_dst():
    global act_dst
    
    # get values from popup
    try:
        window.popup("Select destination path:", 'dst')
        dst = window.entryValue()
    except:
        return

    # check validity of new dst_path
    if not dst == 'None':
        act_dst = dest_paths.index(dst)
    else:
        act_dst = None

    # Save active Image
    if act_dst != None and dest_paths[act_dst] != 'None':
        dst_path = dest_paths[act_dst] + os.sep + active_name[:active_name.rindex('.')] + '.png'
        if not exists(dst_path):
            image_clean.save(dst_path, "PNG")


# get the after natural order next image and load it
def next_image():
    if act_src == "":
        return
    
    # get dir-name and image-file-name
    dir = os.path.dirname(act_src)
    base = os.path.basename(act_src)
    
    # get all file-names of image-files in the same folder
    imgs = []
    valid_images = [".jpg",".png",".jpeg"]
    for file in os.listdir(dir):
        end = os.path.splitext(file)[1]
        if end.lower() not in valid_images:
            continue
        imgs.append(file)
    
    # sort file-list after natural order
    dre = compile(r'(\d+)')
    imgs.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in split(dre, l)])
        
    # find index of open image in sorted list, if there is no next one, return
    i = imgs.index(base)
    if i >= len(imgs) - 1 :
        return
    
    # get path of next image and open it
    next = os.path.join(dir,imgs[i+1])
    select_image(next)
    
def crop_and_save():
    save_path = "/mnt/c/Users/Leonhard Zirus/Desktop/Semester UCA/NNL - Neural Network Learning/NNL-2021-F/img"
    save_path = fd.askdirectory();
    
    for img in images :
        path = images[img]["src"]
        if not path :
            continue
        
        image = Image.open(path)
        #image = cv2.imread(path)
        
        #asp_rat = image.shape[0] / image.shape[1]
        asp_rat = image.width / image.height
        
        image = image.resize((int(asp_rat*size), size), Image.ANTIALIAS)
        #image = cv2.resize(image, (size, (int(asp_rat*size))), interpolation= cv2.INTER_LINEAR)
        
        for rect in images[img]["rectangles"]:
            i = images[img]["rectangles"].index(rect)
            bounds = rect.bounds

            width = int(bounds[3] - bounds[1])
            height = int(bounds[2] - bounds[0])
            length = max(height, width)
            diff = (length - min(height, width))/2
            
            left = bounds[0] - diff if height > width else bounds[0]
            up = bounds[1] if height > width else bounds[1] - diff
            right = bounds[0] + length - diff if height > width else bounds[0] + length
            bottom = bounds[1] + length if height > width else bounds[1] + length - diff
            

            #cropped = image[int(bounds[1]):(int(bounds[1])+length), int(bounds[0]):(int(bounds[0])+length)]
            #cropped = image.crop((bounds[1], length, bounds[0], length))
            #cropped = image.crop((bounds[1], bounds[0], bounds[1]+length, bounds[0]+length))
            if flipped: image = image.rotate(180)
            cropped = image.crop((left, up, right, bottom))
            
            cat = categories[images[img]["rect_to_category"][i]]
            
            if not os.path.exists(save_path+os.sep+cat):
                os.makedirs(save_path+os.sep+cat)
            
            #if cropped.size > 0 : 
            if cropped :
                #cropped = cv2.resize(cropped, (240, 240), cv2.INTER_LINEAR)
                cropped = cropped.resize((240, 240), Image.ANTIALIAS)
                
                #cv2.imwrite(save_path+os.sep+categories[images[img]["rect_to_category"][i]]+os.sep+img+"_"+str(bounds)+".png", cropped, [int(cv2.IMWRITE_PNG_COMPRESSION), None])
                cropped.save(save_path+os.sep+cat+os.sep+img+"_"+str(bounds)+".png", "PNG")
    if DEBUG : print("done exporting")

# Inspired by https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
# loads image and edits the size
def select_image(path=None):
    global image_panel, image, active_name, rectangles, rect_to_category, image_save, dest_paths, act_dst, act_src, image_clean

    # if no parameter was specified, then open fileselecter
    if path == None :
        path = select_file(False, "img")

    if DEBUG : print(path)

    if len(path) > 0:
        # load image from file
        image = cv2.imread(path)
        name = os.path.basename(path)

        # BGR to RBG then OpenCV to PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Alternative:
        #image = Image.open(select_file())

        # Resize to a fixed hight
        asp_rat = image.width / image.height
        image = image.resize((int(asp_rat*size), size), Image.ANTIALIAS)
        image_clean = image.copy()

        # to ImageTk
        imagetk = ImageTk.PhotoImage(image)

        # if the panel was not created yet, create it
        if image_panel is None:
            image_panel = Label(image=imagetk)
            image_panel.image = imagetk         # anti garbage-collection
            image_panel.pack(padx=10, pady=10)
            image_panel.place(relx=.5, rely=.5, anchor="center")
        else:
            image_panel.configure(image=imagetk)
            image_panel.image = imagetk         # anti garbage-collection

        setupMouseListeners(image_panel)

        image_save = image.copy()
        check_save = False

        # save data of last image and if existing load from new image
        if active_name != "":
            # save working arrays to images-dictionary
            images[active_name] = {'rectangles': rectangles.copy(),
                                   'rect_to_category': rect_to_category.copy(),
                                   'dst': act_dst if act_dst != None else 0,
                                   'src': act_src}
            
            # clear working-arrays for next image
            rectangles.clear()
            rect_to_category.clear()
        
        if name in images:
            # if image has saved data, load it to working arrays
            rectangles = images[name]['rectangles']
            rect_to_category = images[name]['rect_to_category']
            act_dst = images[name]['dst']
            act_src = images[name]['src']
            
            # check wether the resized image was already saved
            if act_dst == None or act_dst == 'None':
                check_save = True
            
            # check if the save-path is even valid
            if not os.path.isdir(dest_paths[act_dst]):
                check_save = True
            
            # draw all loaded rectangles
            for rect in rectangles:
                bounds = rect.bounds
                draw_rectangle((bounds[0], bounds[1]),
                               (bounds[2], bounds[3]), (0, 0, 255))
                image_save = image.copy()
        else:
            # if images was not processed yet destination-path must be specified
            check_save = True
        
        # if the image was not saved yet, the destination path must be updated
        if check_save :
            # get save-path from popup
            try:
                raise Exception # in the final use case this functionality was not needed
                window.popup("Select destination path:", 'dst')
                dst = window.entryValue()
            except Exception:
                dst = 'None'

            # update working variable
            if not dst == 'None':
                act_dst = dest_paths.index(dst)
            else:
                act_dst = None

            act_src = path

        # Save Image
        if act_dst != None and dest_paths[act_dst] != 'None':
            dst_path = dest_paths[act_dst] + os.sep + name[:name.rindex('.')] + '.png'
            if not exists(dst_path):
                image.save(dst_path, "PNG")

        active_name = name


# use file-navigator to get file-path
def select_file(save, type, inifile='', title='Open a file', initialdir='~/info_uca/nnl/NNL-2021-F/img/test/with_mask/'):
    filetypes = (
        ('All files', '*.*')
    )
    if type == "img":
        filetypes = (
            ('img files', '*.jpg *.png *.JPEG'),
            ('All files', '*.*')
        )
    elif type == "json":
        filetypes = (
            ('json files', '*.json *.JSON'),
            ('csv files', '*.csv'),
            # ('xls files', '*.xlsx'),
            ('All files', '*.*')
        )

    elif type == "csv":
        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )
    elif type == "ml":
        filetypes = (
            ('h5 files', '*.h5 *.*'),
            ('All files', '*.*')
        )

    # switches between opening and saving the file
    if not save:
        path = fd.askopenfilename(
            title=title,
            initialdir=initialdir,
            filetypes=filetypes)
    else:
        file = fd.asksaveasfile(
            initialfile=inifile,
            title=title,
            initialdir=initialdir,
            filetypes=filetypes)
        path = file.name

    return path

# ================= Right click context menu =================

selected_rect = None

# show information of selected rectangle
def view_rect():
    # check if selection was made
    if selected_rect == None:
        return
    
    # fire popup and change category if need be
    try:
        window.popup_select("Select Category:", "select", selected_rect,
                            rect_to_category[rectangles.index(selected_rect)])
        cat = window.entryValue()
        rect_to_category[rectangles.index(
            selected_rect)] = categories.index(cat)
    except:
        return


# delete selected rectangle
def delete_rect():
    global rectangles, rect_to_category, image, image_save
    
    # check if rectangle was selected
    if selected_rect == None:
        return
    
    # delete rectangle from both arrays
    index = rectangles.index(selected_rect)
    del rectangles[index]
    del rect_to_category[index]

    # reset image-drawings and redraw all remaining rectangles
    image_save = image_clean.copy()
    for rect in rectangles:
        bounds = rect.bounds
        draw_rectangle((bounds[0], bounds[1]),
                       (bounds[2], bounds[3]), (0, 0, 255))
        image_save = image.copy()


def add_context_menu():
    # create menubar for click on rectangle as well as click anywhere else
    menu = Menu(root, tearoff=0)
    menu.add_command(label="Add Category", command=addCategory)
    menu.add_command(label="Replace Category", command=replaceCategory)
    menu.add_command(label="View Categories", command=listCategories)
    menu.add_separator()
    menu.add_command(label="Exit", command=root.destroy)
    
    rect_menu = Menu(root, tearoff=0)
    rect_menu.add_command(label="View", command=view_rect)
    rect_menu.add_command(label="Delete", command=delete_rect)

    # define intern callback function in order to access local variables
    def right_click(event):
        global selected_rect
        
        # find if click was on a rectangle
        for rect in rectangles:
            point = Point(event.x, event.y)
            
            if rect.contains(point):
                # change color of selected rectangle
                bounds = rect.bounds
                draw_rectangle((bounds[0], bounds[1]), (bounds[2], bounds[3]))

                selected_rect = rect

                # fire menu-popup
                try:
                    rect_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    # close popup and recolor rectangle
                    rect_menu.grab_release()
                    draw_rectangle((bounds[0], bounds[1]),
                                   (bounds[2], bounds[3]), (0, 0, 255))
                return
        
        # if click was not on rectangle fire other menu
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # binding right click button to root
    root.bind("<Button-3>", right_click)

# ================= Draw Rectangle =================

# draw rectangle on active-image
def draw_rectangle(xy0, xy1, color=(255, 0, 0)):
    global image_save, image
    
    # get last saved image to delete any not yet permanent rectangle
    image = image_save.copy()
    
    # draw rectangle on image
    rect = ImageDraw.Draw(image, 'RGBA')
    rect.rectangle([xy0, xy1], fill=None, outline=color, width=2)
    
    # add the new image to TK-Panel
    imagetk = ImageTk.PhotoImage(image)
    image_panel.configure(image=imagetk)
    image_panel.image = imagetk     # anti garbage-collection


# callback for left-press
def left_pressed(event):
    global x0, y0, x1, y1, image, image_save
    
    # new rectangle is started drawing, so image_save is updated
    image_save = image.copy()
    
    # both corner coordinates are set to this point
    x0, y0 = event.x, event.y
    x1, y1 = event.x, event.y
    return


# callback for left-pressed mouse that is moving
def left_moved(event):
    global x0, y0, x1, y1
    
    # second corner is updated and new rectangle drawn
    x1, y1 = event.x, event.y
    draw_rectangle((x0, y0), (x1, y1))


# callback for left-released
def left_released(event):
    global x0, y0, x1, y1, image, image_save

    # rectangle is final, so object is created
    rect = box(x0, y0, x1, y1)

    # check rectangle area (could be just a click) and validity
    if rect.area < 5 or not check_validity(rect, x0, y0, x1, y1):
        # image reset to before rectangle was drawn
        image = image_save.copy()

        # update image-panel
        imagetk = ImageTk.PhotoImage(image)
        image_panel.configure(image=imagetk)
        image_panel.image = imagetk

        return

    # rectangle was valid and is added to the list, image keeps drawn rectangle
    rectangles.append(rect)
    image_save = image.copy()

    # category select popup is fired
    try:
        window.popup_select("Select Category:", "select", rect, 0)
        cat = window.entryValue()
        rect_to_category.append(categories.index(cat))
    except:
        rect_to_category.append(0)

    # rectangle is drawn in final color
    draw_rectangle((x0, y0), (x1, y1), (0, 0, 255))


# check validity of new rectangle
def check_validity(rect, x0, y0, x1, y1):

    # check area, and side-lenghts
    if rect.area <= 40 or abs(x0-x1) <= 5 or abs(y0-y1) <= 5:
        if DEBUG:
            print("not big enough")
        messagebox.showinfo("Invalid Box",  "The Box is to small!")
        return False

    # check all intersections and overlap percentages
    for R in rectangles:
        inter = rect.intersection(R)
        if (inter.area / rect.area) > 0.2 or inter.area == R.area:
            if DEBUG:
                print("overlap")
            messagebox.showinfo("Invalid Box",  "Overlapping to much!")
            return False

    return True


# callback function for double click
def double_click(event):
    # get mouse-click location as point
    point = Point(event.x, event.y)
    
    # find first rectangle, that contains the point
    for rect in rectangles:
        if rect.contains(point):

            # change rectangle color to show it's selected
            bounds = rect.bounds
            draw_rectangle((bounds[0], bounds[1]), (bounds[2], bounds[3]))

            # fire category select popup with infos, change category if necessary
            try:
                window.popup_select(
                    "Select Category:", "select", rect, rect_to_category[rectangles.index(rect)])
                cat = window.entryValue()
                rect_to_category[rectangles.index(
                    rect)] = categories.index(cat)
            except:
                if DEBUG: print("no selection")

            # change color back to normal
            draw_rectangle((bounds[0], bounds[1]),
                           (bounds[2], bounds[3]), (0, 0, 255))

            break


def setupMouseListeners(element):
    element.bind("<ButtonPress-1>", left_pressed)
    element.bind("<B1-Motion>", left_moved)
    element.bind("<ButtonRelease-1>", left_released)
    element.bind('<Double-Button-1>', double_click)


def main():
    global window
    
    if DEBUG:
        print("Image Annotator started")

    # init tkinter
    window = Window(root)

    # window title
    root.wm_title("Image Annotator")

    # set window size
    root.geometry(str(int((16/9)*size))+"x"+str(size))

    # add right click menu
    add_context_menu()

    # show window
    root.mainloop()

    return


# call main function
main()