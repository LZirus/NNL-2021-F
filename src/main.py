from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk,Image  

#Global
root = Tk()
images = []
img_label = Label()
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
        
        fileMenu.add_command(label="OpenImage", command=openImage)
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        editMenu.add_command(label="Undo")
        editMenu.add_command(label="Redo")
    
    def exitProgram(self):
        exit()

def openImage() :
    global img_label, size
    
    img_label.destroy()
    
    # Image
    image = Image.open(select_file())
    
    asp_rat = image.width / image.height
    resized = image.resize((int(asp_rat*size), size), Image.ANTIALIAS)
    
    img = ImageTk.PhotoImage(resized)  
    images.append(img)
    img_label = Label(root, image=img)
    img_label.pack(side=TOP)
    # canvas.create_image(
    #     10, 
    #     10, 
    #     anchor=NW, 
    #     image=images[0]
    #     )     

def select_file():
    filetypes = (
        ('img files', '*.jpg *.png *.JPEG'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='~/info_uca/nnl/NNL-2021-F/img',
        filetypes=filetypes)

    return filename

# https://www.codershubb.com/create-context-menu-in-tkinter-python/
# define function to cut 
# the selected text
def cut_text():
        print("cut")
# define function to copy 
# the selected text
def copy_text():
        print("copy")
# define function to paste 
# the previously copied text
def paste_text():
        print("paste")

def add_context_menu():
    # create menubar  
    menu = Menu(root, tearoff = 0) 
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

def main() :
    print("Hello World!")
    
    #init tkinter
    app = Window(root)
    
    #window title
    root.wm_title("My Window")
    
    #set window size
    root.geometry(str(int((16/9)*size))+"x"+str(size))
    
    
    #show window
    root.mainloop()
    
    return

main()