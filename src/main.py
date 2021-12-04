from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageDraw
#import tkFileDialog
import cv2

# Global
root = Tk()
image = None
img_label = None
panelA = None
size = 1000


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        # top menu
        self.master = master
        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu)
        menu.add_cascade(label="File", menu=fileMenu)

        editMenu = Menu(menu)
        menu.add_cascade(label="Edit", menu=editMenu)

        fileMenu.add_command(label="OpenImage", command=select_image)
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        editMenu.add_command(label="Draw", command=draw)
        editMenu.add_command(label="Redo")

    def exitProgram(self):
        exit()

def draw_rectangle(xy0, xy1) :
    global image
    draw = ImageDraw.Draw(image)
    color = (0, 0, 255)
    draw.rectangle([xy0, xy1], fill=None, outline=color, width=2)
    imagetk = ImageTk.PhotoImage(image)
    panelA.configure(image=imagetk)
    panelA.image = imagetk

def draw() :
    global image
    draw_rectangle((20,20), (100,100))
    draw = ImageDraw.Draw(image)
    draw.line((0, 0) + image.size, fill=128)
    draw.line((0, image.size[1], image.size[0], 0), fill=128)
    imagetk = ImageTk.PhotoImage(image)
    panelA.configure(image=imagetk)
    panelA.image = imagetk

def select_image():
    # https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/

    global panelA, panelB, image

    path = select_file()

    if len(path) > 0:
        image = cv2.imread(path)
        print(path)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # convert the images to PIL format...
        image = Image.fromarray(image)
        
        # Alternative: 
        #image = Image.open(select_file())
        
        asp_rat = image.width / image.height
        #image = image.resize((int(asp_rat*size), size), Image.ANTIALIAS)

        # ...and then to ImageTk format
        imagetk = ImageTk.PhotoImage(image)

        if panelA is None :
            panelA = Label(image=imagetk)
            panelA.image = imagetk
            panelA.pack(padx=10, pady=10)
            panelA.place(relx=.5, rely=.5, anchor="center")

        else:
            panelA.configure(image=imagetk)
            panelA.image = imagetk


def select_file():
    filetypes = (
        ('img files', '*.jpg *.png *.JPEG'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='~/info_uca/nnl/NNL-2021-F/img/test/with_mask/',
        filetypes=filetypes)

    return filename

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


def main():
    print("Hello World!")

    # init tkinter
    app = Window(root)

    # window title
    root.wm_title("My Window")

    # set window size
    root.geometry(str(int((16/9)*size))+"x"+str(size))
    
    add_context_menu()

    # show window
    root.mainloop()

    return


main()
