import cv2
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import ImageTk, Image, ImageDraw
from shapely.geometry import box, Polygon, Point
import json
import os
from os.path import exists
import csv
from re import compile, split

# Global
root = Tk()
image = None
active_name = ""
image_save = None
image_panel = None
image_clean = None
window = None
size = 1000
rectangles = []
rect_to_category = []
categories = ["None", "mask", "no mask", "baum"]
images = {}
# TODO: Save Dest-Paths for processed images
dest_paths = ["None"]
act_src = ""
act_dst = 0

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
        
        annoMenu = Menu(menu)
        menu.add_cascade(label="Annotations", menu=annoMenu)

        categoryMenu = Menu(menu)
        menu.add_cascade(label="Categories", menu=categoryMenu)

        fileMenu.add_command(label="Open Image", command=select_image)
        fileMenu.add_command(label="Load next Image", command=next_image)
        fileMenu.add_command(label="Change this Save-Location", command=change_dst)
        fileMenu.add_command(label="Replace all Save-Locations", command=replace_dst)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        
        annoMenu.add_command(label="Save Annotations", command=save_annos)
        annoMenu.add_command(label="Import Annotations", command=load_annos)
        annoMenu.add_command(label="View Annotations", command=view_annos)

        categoryMenu.add_command(label="Add", command=addCategory)
        categoryMenu.add_command(label="Replace", command=replaceCategory)
        categoryMenu.add_command(label="List", command=listCategories)
        categoryMenu.add_command(label="Import", command=importCategories)
        # TODO: csv support for file export
        categoryMenu.add_command(label="Export", command=exportCategories)
        # TODO: save resized images
        # TODO: add fileformat support

    def popup(self, text, type):
        self.w = popupWindow(self.master, text, type)
        self.master.wait_window(self.w.top)

    def popup_select(self, text, type, rect, i):
        self.w = popupWindow(self.master, text, type, rect, i)
        self.master.wait_window(self.w.top)

    def entryValue(self):
        return self.w.value

    def entryValues(self):
        return self.w.value0, self.w.value1

    def exitProgram(self):
        exit()

# Inspired by https://stackoverflow.com/questions/10020885/creating-a-popup-message-box-with-an-entry-field#10021242


class popupWindow(object):
    def __init__(self, master, text, type, rect=None, i=0):
        top = self.top = Toplevel(master)

        if type == "all":
            top.geometry(str(700)+"x"+str(150))
            self.list_all(top, text)
        else:
            top.geometry(str(int((16/9)*150))+"x"+str(150))

        if type == "one":
            self.oneInput(top, text)
        if type == "two":
            self.twoInput(top, text)
        if type == "list":
            self.list(top, text)
        if type == "select":
            self.rect = rect
            self.select(top, text, i)
        if type == "dst":
            self.pathSelect(top, text)
        if type == "rep":
            self.pathReplace(top, text)

    def oneInput(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()
        self.e = Entry(top)
        self.e.pack()
        self.b = Button(top, text='Ok', command=lambda: self.cleanup("one"))
        self.b.pack()

    def twoInput(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        self.active = StringVar()
        self.active.set(categories[0])
        self.drop = OptionMenu(top, self.active, *categories)
        self.drop.pack()

        self.e1 = Entry(top)
        self.e1.pack()

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("two"))
        self.b.pack()

    def selectFolder(self):
        global dest_paths
        path = fd.askdirectory()
        if path not in dest_paths:
            dest_paths.append(path)
            self.active.set(path)
    
    def selectFolderRep(self):
        path = fd.askdirectory()
        self.rep.config(text=path)
            
    def pathReplace(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        self.active = StringVar()
        self.active.set(dest_paths[0])
        self.drop = OptionMenu(top, self.active, *dest_paths)
        self.drop.pack()
        self.drop.config(width=13)

        self.rep = Label(top, text='None')
        self.rep.pack()
        self.src = Button(top, text='Replace with', command=self.selectFolderRep)
        self.src.pack()
        self.src.config(width=13)

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("rep"))
        self.b.pack()

    def pathSelect(self, top, intext):
        top.wm_title("Input")

        self.l = Label(top, text=intext)
        self.l.pack()

        self.active = StringVar()
        self.active.set(dest_paths[act_dst])
        self.drop = OptionMenu(top, self.active, *dest_paths)
        self.drop.pack(side=LEFT)
        self.drop.config(width=13)

        self.src = Button(top, text='Source', command=self.selectFolder)
        self.src.pack(side=RIGHT)
        self.src.config(width=13)

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("select"))
        self.b.pack(side=BOTTOM)

    # Inspired by https://www.geeksforgeeks.org/scrollable-listbox-in-python-tkinter/
    def list(self, top, intext):
        top.wm_title("Info")

        self.l = Label(top, text=intext)
        self.l.pack()

        listbox = Listbox(top)
        listbox.pack(side=LEFT, fill=BOTH)

        scrollbar = Scrollbar(top)
        scrollbar.pack(side=RIGHT, fill=BOTH)

        for values in categories:
            listbox.insert(END, values)

        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

    def image_callback(self, _):

        items = self.image_listbox.curselection()

        if len(items) != 1:
            return

        self.rectangles_listbox.delete(0, END)

        self.active_img = self.image_listbox.get(items[0])

        for rect in images[self.active_img]["rectangles"]:
            self.rectangles_listbox.insert(END, str(rect.bounds))

        return

    def rect_callback(self, _):

        items = self.rectangles_listbox.curselection()

        if len(items) != 1:
            return

        self.info_listbox.delete(0, END)

        active = self.rectangles_listbox.get(items[0])

        for rect in images[self.active_img]['rectangles']:
            if str(rect.bounds) == active:
                self.info_listbox.insert(END, "Area: " + str(rect.area))
                self.info_listbox.insert(END, "Bounds: " + str(rect.bounds))
                rect_i = images[self.active_img]['rectangles'].index(rect)
                cat_i = images[self.active_img]['rect_to_category'][rect_i]
                cat = categories[cat_i]
                self.info_listbox.insert(END, "Category: " + cat)

        return

    def list_all(self, top, intext):
        top.wm_title("Info")

        l = Label(top, text=intext)
        l.pack()

        max_size = 35
        self.image_listbox = Listbox(top, width=max_size)
        self.rectangles_listbox = Listbox(top, width=max_size)
        self.info_listbox = Listbox(top, width=max_size+5)

        img_scrollbar = Scrollbar(top)
        rect_scrollbar = Scrollbar(top)

        self.image_listbox.pack(side=LEFT, fill=BOTH)
        img_scrollbar.pack(side=LEFT, fill=BOTH)

        self.rectangles_listbox.pack(side=LEFT, fill=BOTH)
        rect_scrollbar.pack(side=LEFT, fill=BOTH)

        self.info_listbox.pack(side=LEFT, fill=BOTH)

        img_scrollbar = Scrollbar(top)

        self.image_listbox.bind("<<ListboxSelect>>", self.image_callback)
        self.rectangles_listbox.bind("<<ListboxSelect>>", self.rect_callback)

        for img in images.keys():
            self.image_listbox.insert(END, img)

        self.image_listbox.config(yscrollcommand=img_scrollbar.set)
        img_scrollbar.config(command=self.image_listbox.yview)

    def select(self, top, intext, i):
        top.wm_title("Info")

        self.l = Label(top, text=intext)
        self.l.pack()

        text = "Area: " + str(self.rect.area) + \
            "\nBounds: " + str(self.rect.bounds)

        self.info = Label(top, text=text)
        self.info.pack()

        self.active = StringVar()
        self.active.set(categories[i])
        self.drop = OptionMenu(top, self.active, *categories)
        self.drop.pack()

        self.b = Button(top, text='Ok', command=lambda: self.cleanup("select"))
        self.b.pack()

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

        self.top.destroy()


def addCategory():
    try:
        window.popup("Add Category:", "one")
        if window.entryValue() in categories:
            return
        categories.append(window.entryValue())
    except:
        return


def replaceCategory():
    try:
        window.popup("Replace Category:", "two")
        c1, c2 = window.entryValues()
        if not c1 in categories:
            return
        categories[categories.index(c1)] = c2
    except:
        return


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


def save_annos():
    global images

    if active_name != "":
        images[active_name] = {'rectangles': rectangles.copy(),
                               'rect_to_category': rect_to_category.copy(),
                               'dst': act_dst if act_dst != None else 0,
                               'src': act_src}

    save = {"categories": categories,
            "images": {}, 'destinations': dest_paths}
    
    for key in images :
        save['images'][key] = {'rectangles': [rect.bounds for rect in images[key]['rectangles']],
                               'rect_to_category': images[key]['rect_to_category'].copy(),
                               'dst': images[key]['dst'],
                               'src': images[key]['src']}


    # Writing to sample.json
    with open(select_file(True, "json", "annotations.json", title='Save Annotations', initialdir='~/annotations.json'), "w") as outfile:
        json.dump(save, outfile)
    return


def load_annos():
    global images, categories, image_save, rect_to_category, rectangles, dest_paths
    # Opening JSON file
    with open(select_file(False, "json", "annotations.json", title='Load Annotations', initialdir='~/info_uca/annotations.json'), 'r') as openfile:
        loaded = json.load(openfile)

    images = loaded['images'].copy()
    categories = loaded['categories'].copy()
    dest_paths = loaded['destinations'].copy()

    for key in loaded['images']:
        images[key]['rectangles'] = [
            box(rect[0], rect[1], rect[2], rect[3]) for rect in images[key]['rectangles']]

    if active_name in images:
        #categories = images[name]['categories']
        rectangles = images[active_name]['rectangles']
        rect_to_category = images[active_name]['rect_to_category']
        for rect in rectangles:
            bounds = rect.bounds
            draw_rectangle((bounds[0], bounds[1]),
                           (bounds[2], bounds[3]), (0, 0, 255))
            image_save = image.copy()

    return


def view_annos():
    if active_name != "":
        images[active_name] = {'rectangles': rectangles.copy(),
                               'rect_to_category': rect_to_category.copy(),
                               'dst': act_dst if act_dst != None else 0,
                               'src': act_src}
    window.popup("Categories:", "all")
    return


def replace_dst():
    global dest_paths
    
    try:
        window.popup("Replace destination path:", 'rep')
        c1, c2 = window.entryValues()
    except:
        return
    
    if c2 == None : return
    if not c1 in dest_paths : return
    if not os.path.isdir(c2) : return

    dest_paths[dest_paths.index(c1)] = c2
    
    for img in images:
        act_dst = images[img]['dst']
        active_name = img
        # Save Image
        if act_dst != None and dest_paths[act_dst] != 'None':
            dst_path = dest_paths[act_dst] + os.sep + active_name[:active_name.rindex('.')] + '.png'

            if not exists(dst_path):
                image_clean.save(dst_path, "PNG")
    
    return


def change_dst():
    global act_dst
    # setup save-path and
    try:
        window.popup("Select destination path:", 'dst')
        dst = window.entryValue()
    except:
        return

    if not dst == 'None':
        act_dst = dest_paths.index(dst)
    else:
        act_dst = None

    # Save Image
    if act_dst != None and dest_paths[act_dst] != 'None':
        dst_path = dest_paths[act_dst] + os.sep + active_name[:active_name.rindex('.')] + '.png'

        if not exists(dst_path):
            image_clean.save(dst_path, "PNG")


def next_image():
    if act_src == "":
        return
    
    dir = os.path.dirname(act_src)
    base = os.path.basename(act_src)
    
    imgs = []
    valid_images = [".jpg",".png",".JPEG"]
    for f in os.listdir(dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(f)
    
    dre = compile(r'(\d+)')
    imgs.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in split(dre, l)])
        
    i = imgs.index(base)
    if i >= len(imgs) - 1 :
        return
    
    next = os.path.join(dir,imgs[i+1])
    select_image(next)
    
    return


# Inspired by https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
# loads image and edits the size
def select_image(path=None):
    global image_panel, image, active_name, rectangles, rect_to_category, image_save, dest_paths, act_dst, act_src, image_clean

    if path == None :
        path = select_file(False, "img")

    if DEBUG : print(path)

    if len(path) > 0:
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

        if active_name != "":
            images[active_name] = {'rectangles': rectangles.copy(),
                                   'rect_to_category': rect_to_category.copy(),
                                   'dst': act_dst if act_dst != None else 0,
                                   'src': act_src}
            rectangles.clear()
            rect_to_category.clear()
        if name in images:
            #categories = images[name]['categories']
            rectangles = images[name]['rectangles']
            rect_to_category = images[name]['rect_to_category']
            act_dst = images[name]['dst']
            act_src = images[name]['src']
            
            if act_dst == 'None' or act_dst == None:
                check_save = True
            
            if not os.path.isdir(dest_paths[act_dst]):
                check_save = True
            
            for rect in rectangles:
                bounds = rect.bounds
                draw_rectangle((bounds[0], bounds[1]),
                               (bounds[2], bounds[3]), (0, 0, 255))
                image_save = image.copy()
        else:
            check_save = True
            
        if check_save :
            # setup save-path and
            try:
                window.popup("Select destination path:", 'dst')
                dst = window.entryValue()
            except Exception as e:
                print(e)
                dst = 'None'

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
            ('json files', '*.json *.j *.JSON'),
            ('csv files', '*.csv'),
            ('xls files', '*.xlsx'),
            ('All files', '*.*')
        )

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


def view_rect():
    if selected_rect == None:
        return
    try:
        window.popup_select("Select Category:", "select", selected_rect,
                            rect_to_category[rectangles.index(selected_rect)])
        cat = window.entryValue()
        rect_to_category[rectangles.index(
            selected_rect)] = categories.index(cat)
    except:
        return


def delete_rect():
    global rectangles, rect_to_category, image, image_save
    if selected_rect == None:
        return
    index = rectangles.index(selected_rect)
    del rectangles[index]
    del rect_to_category[index]

    image_save = image_clean.copy()
    for rect in rectangles:
        bounds = rect.bounds
        draw_rectangle((bounds[0], bounds[1]),
                       (bounds[2], bounds[3]), (0, 0, 255))
        image_save = image.copy()

    return


def add_context_menu():
    # create menubar
    menu = Menu(root, tearoff=0)
    menu.add_command(label="Add Cat", command=addCategory)
    menu.add_command(label="Replace Cat", command=replaceCategory)
    menu.add_command(label="View Cats", command=listCategories)
    menu.add_separator()
    menu.add_command(label="Exit", command=root.destroy)
    # define function to popup the
    # context menu on right button click

    rect_menu = Menu(root, tearoff=0)
    rect_menu.add_command(label="View", command=view_rect)
    rect_menu.add_command(label="Delete", command=delete_rect)

    def right_click(event):
        global selected_rect
        for rect in rectangles:
            point = Point(event.x, event.y)
            if rect.contains(point):

                bounds = rect.bounds
                draw_rectangle((bounds[0], bounds[1]), (bounds[2], bounds[3]))

                selected_rect = rect

                try:
                    rect_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    rect_menu.grab_release()
                    draw_rectangle((bounds[0], bounds[1]),
                                   (bounds[2], bounds[3]), (0, 0, 255))

                return
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # binding right click button to root
    root.bind("<Button-3>", right_click)

# ================= Draw Rectangle =================


def draw_rectangle(xy0, xy1, color=(255, 0, 0)):
    global image_save, image
    image = image_save.copy()
    rect = ImageDraw.Draw(image, 'RGBA')
    rect.rectangle([xy0, xy1], fill=None, outline=color, width=2)
    imagetk = ImageTk.PhotoImage(image)
    image_panel.configure(image=imagetk)
    image_panel.image = imagetk


def left_pressed(event):
    global x0, y0, x1, y1, image, image_save
    image_save = image.copy()
    x0, y0 = event.x, event.y
    x1, y1 = event.x, event.y
    return


def left_moved(event):
    global x0, y0, x1, y1
    x1, y1 = event.x, event.y
    draw_rectangle((x0, y0), (x1, y1))
    return


def check_validity(rect, x0, y0, x1, y1):

    if rect.area <= 40 or abs(x0-x1) <= 5 or abs(y0-y1) <= 5:
        if DEBUG:
            print("not big enough")
        messagebox.showinfo("Invalid Box",  "The Box is to small!")
        return False

    for R in rectangles:
        inter = rect.intersection(R)
        if (inter.area / rect.area) > 0.2 or inter.area == R.area:
            if DEBUG:
                print("overlap")
            messagebox.showinfo("Invalid Box",  "Overlapping to much!")
            return False

    return True


def left_released(event):
    global x0, y0, x1, y1, image, image_save

    rect = box(x0, y0, x1, y1)

    if rect.area < 5 or not check_validity(rect, x0, y0, x1, y1):
        image = image_save.copy()

        imagetk = ImageTk.PhotoImage(image)
        image_panel.configure(image=imagetk)
        image_panel.image = imagetk

        return

    rectangles.append(rect)
    image_save = image.copy()

    try:
        window.popup_select("Select Category:", "select", rect, 0)
        cat = window.entryValue()
        rect_to_category.append(categories.index(cat))
    except:
        rect_to_category.append(0)

    draw_rectangle((x0, y0), (x1, y1), (0, 0, 255))

    return


def double_click(event):
    point = Point(event.x, event.y)
    for rect in rectangles:
        if rect.contains(point):

            bounds = rect.bounds
            draw_rectangle((bounds[0], bounds[1]), (bounds[2], bounds[3]))

            try:
                window.popup_select(
                    "Select Category:", "select", rect, rect_to_category[rectangles.index(rect)])
                cat = window.entryValue()
                rect_to_category[rectangles.index(
                    rect)] = categories.index(cat)
            except:
                print("no selection")

            draw_rectangle((bounds[0], bounds[1]),
                           (bounds[2], bounds[3]), (0, 0, 255))

            break
    return


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

    add_context_menu()

    # show window
    root.mainloop()

    return


main()