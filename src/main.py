from os import replace
from re import I
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import ImageTk, Image, ImageDraw
import cv2
from shapely.geometry import box, Polygon, Point

# Global
root = Tk()
image = None
image_save = None
image_panel = None
window = None
size = 1000
rectangles = []
rect_to_category = []
categories = []

# Config
DEBUG = True


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        # top menu
        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu)
        menu.add_cascade(label="File", menu=fileMenu)

        editMenu = Menu(menu)
        menu.add_cascade(label="Categories", menu=editMenu)

        fileMenu.add_command(label="Open Image", command=select_image)
        fileMenu.add_command(label="Save Annotations")
        fileMenu.add_command(label="View Annotations")
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        
        
        editMenu.add_command(label="Add", command=addCategory)
        editMenu.add_command(label="Replace", command=replaceCategory)
        editMenu.add_command(label="List", command=listCategories)
        editMenu.add_command(label="Import", command=importCategories)
        editMenu.add_command(label="Export", command=exportCategories)

    def popup(self, text, type):
        self.w=popupWindow(self.master, text, type)
        self.master.wait_window(self.w.top)

    def entryValue(self):
        return self.w.value
    
    def entryValues(self):
        return self.w.value0, self.w.value1

    def exitProgram(self):
        exit()

# Inspired by https://stackoverflow.com/questions/10020885/creating-a-popup-message-box-with-an-entry-field#10021242
class popupWindow(object):
    def __init__(self,master, text, type):
        top=self.top=Toplevel(master)
        top.geometry(str(int((16/9)*100))+"x"+str(100))
        
        if type == "one" :
            self.oneInput(top, text)
        if type == "two" :
            self.twoInput(top, text)
        if type == "list" :
            self.list(top, text)
    
    def oneInput(self, top, intext) :
        top.wm_title("Input")
        
        self.l=Label(top,text=intext)
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=lambda: self.cleanup("one"))
        self.b.pack()
    
    def twoInput(self, top, intext) :
        top.wm_title("Input")
        
        self.l=Label(top,text=intext)
        self.l.pack()
        self.e0=Entry(top)
        self.e1=Entry(top)
        self.e0.pack()
        self.e1.pack()
        self.b=Button(top,text='Ok',command=lambda: self.cleanup("two"))
        self.b.pack()
    
    # Inspired by https://www.geeksforgeeks.org/scrollable-listbox-in-python-tkinter/
    def list(self, top, intext) :
        top.wm_title("Info")
        
        self.l=Label(top,text=intext)
        
        listbox = Listbox(top)
        listbox.pack(side = LEFT, fill = BOTH)
        
        scrollbar = Scrollbar(top)
        scrollbar.pack(side = RIGHT, fill = BOTH)
        
        for values in categories:
            listbox.insert(END, values)
            
        listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = listbox.yview)
    
    def cleanup(self, type):
        if type == "one" :
            self.value=self.e.get()
        elif type == "two" :
            self.value0 = self.e0.get()
            self.value1 = self.e1.get()
        
        self.top.destroy()

def addCategory() :
    window.popup("Add Category:", "one")
    if window.entryValue() in categories :
        return
    categories.append(window.entryValue())
    print(categories)
    return

def replaceCategory() :
    window.popup("Replace Category:", "two")
    c1, c2 = window.entryValues()
    if not c1 in categories :
        return
    categories[categories.index(c1)] = c2
    print(categories)
    return

def listCategories() :
    window.popup("Categories:", "list")
    return

def importCategories() :
    return

def exportCategories() :
    return

# Inspired by https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
# loads image and edits the size
def select_image():
    global image_panel, image

    path = select_file()

    if len(path) > 0:
        image = cv2.imread(path)
        print(path)

        # BGR to RBG then OpenCV to PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image)
        
        # Alternative: 
        #image = Image.open(select_file())
        
        # Resize to a fixed hight
        asp_rat = image.width / image.height
        image = image.resize((int(asp_rat*size), size), Image.ANTIALIAS)

        # to ImageTk
        imagetk = ImageTk.PhotoImage(image)

        if image_panel is None :
            image_panel = Label(image=imagetk)
            image_panel.image = imagetk         # anti garbage-collection
            image_panel.pack(padx=10, pady=10)
            image_panel.place(relx=.5, rely=.5, anchor="center")
        else:
            image_panel.configure(image=imagetk)
            image_panel.image = imagetk         # anti garbage-collection
        
        setupMouseListeners(image_panel)


# use file-navigator to get file-path
def select_file():
    filetypes = (
        ('img files', '*.jpg *.png *.JPEG'),
        ('All files', '*.*')
    )
    
    path = fd.askopenfilename(
        title='Open a file',
        initialdir='~/info_uca/nnl/NNL-2021-F/img/test/with_mask/',
        filetypes=filetypes)

    return path

# ================= Right click context menu =================
def cut_text():
    print("cut")
    
def copy_text():
    print("copy")

def paste_text():
    print("paste")

def add_context_menu():
    # create menubar
    menu = Menu(root, tearoff=0)
    menu.add_command(label="Cut", command=cut_text)
    menu.add_command(label="Copy", command=copy_text)
    menu.add_command(label="Paste", command=paste_text)
    menu.add_separator()
    menu.add_command(label="Exit", command=root.destroy)
    # define function to popup the
    # context menu on right button click

    def context_menu(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    # binding right click button to root
    root.bind("<Button-3>", context_menu)

# ================= Draw Rectangle =================

def draw_rectangle(xy0, xy1) :
    global image_save, image
    image = image_save.copy()
    rect = ImageDraw.Draw(image, 'RGBA')
    color = (0, 0, 255)
    rect.rectangle([xy0, xy1], fill=None, outline=color, width=2)
    imagetk = ImageTk.PhotoImage(image)
    image_panel.configure(image=imagetk)
    image_panel.image = imagetk

def left_pressed(event) :
    global x0, y0, x1, y1, image, image_save
    image_save = image.copy()
    x0, y0 = event.x, event.y
    x1, y1 = event.x, event.y
    return

def left_moved(event) :
    global x0, y0, x1, y1
    x1, y1 = event.x, event.y
    draw_rectangle((x0, y0), (x1, y1))
    return

def check_validity(rect, x0, y0, x1, y1) :
    
    if rect.area <= 40 or abs(x0-x1) <= 5 or abs(y0-y1) <= 5 :
        if DEBUG : print("not big enough")
        messagebox.showinfo("Invalid Box",  "The Box is to small!")
        return False
    
    for R in rectangles :
        inter = rect.intersection(R)
        if (inter.area / rect.area) > 0.2 or inter.area == R.area:
            if DEBUG : print("overlap")
            messagebox.showinfo("Invalid Box",  "Overlapping to much!")
            return False
    
    return True

def left_released(event) :
    global x0, y0, x1, y1, image, image_save
    
    rect = box(x0, y0, x1, y1)
    
    if rect.area < 5 :
        return

    if not check_validity(rect, x0, y0, x1, y1) :
        image = image_save.copy()
        
        imagetk = ImageTk.PhotoImage(image)
        image_panel.configure(image=imagetk)
        image_panel.image = imagetk
        
        return
    
    rectangles.append(rect)
    image_save = image.copy()
    
    
    
    # TODO: Add Rect to List, chose Category, etc.
    # - Choose Category/Tags for Rect and save in rect_to_category with index
    # - 
    return

def double_click(event) :
    point = Point(event.x, event.y)
    for rect in rectangles :
        if rect.contains(point):
            print("clicked "+str(rect))
            
            # TODO: Info-Popup
            # - Fire Popup with info and category-change
            # - Maybe highlight rectangle
            
            break
    return

def setupMouseListeners(element) :
    element.bind("<ButtonPress-1>", left_pressed)
    element.bind("<B1-Motion>", left_moved)
    element.bind("<ButtonRelease-1>", left_released)
    element.bind('<Double-Button-1>', double_click)


def main():
    global window
    if DEBUG : print("Image Annotator started")
    
    # init tkinter
    window = Window(root)

    # window title
    root.wm_title("Image Annotator")

    # set window size
    root.geometry(str(int((16/9)*size))+"x"+str(size))
    
    add_context_menu()

    # show window
    root.mainloop()

    return


main()
