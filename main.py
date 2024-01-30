
# Easy Quadrat v1.0


# Easy Quadrat is nothing but the same quadrat traditionally used in agronomy and crop science. Now, it has been brought
# into the field of modern phenotyping, for cropping and sampling; not from the plants themselves, but rather from the
# images of plants.


# Author:
# Abbas Haghshenas
# Shiraz, Iran


# Note:
# This code is written with the assistance of Chat GPT 3.5 (OpenAI).
# The third method, titled "Auto (classic I)" is suggested by Peter Suter (for more information on this specific method,
# see: https://discuss.python.org/t/how-can-i-detect-and-crop-the-rectangular-frame-in-the-image/32378/25)

# -----------------------------------------------------------------------

# Copyright (C) 2024 Abbas Haghshenas
#
# This software, Easy Quadrat version 1.0 (MIT license), includes the YOLOv8n-seg and YOLOv8m-seg models, which are
# subject to the terms of the AGPL-3.0 license.
#
# -----------------------------------------------------------------------
#
# MIT LICENSE
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# -----------------------------------------------------------------------
#
# AGPL-3.0 LICENSE
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
#
# For more information about the YOLOv8n-seg and YOLOv8m-seg models, visit [https://docs.ultralytics.com/tasks/segment].
#
# Contact: Abbas Haghshenas
# Email: haqueshenas@gmail.com

# -----------------------------------------------------------------------

from tkinter import *
from tkinter import ttk
import tkinter as tk
import tkinter
import tkinter.filedialog
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
import cv2
import skimage.color
import skimage.io
import skimage.measure
import skimage.morphology
from skimage import segmentation
import skimage.morphology as morphology
from skimage.morphology import convex_hull_image
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os.path
from os import path
import matplotlib.colors
from tqdm import tqdm
import datetime
import math
import json
from ultralytics import YOLO
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

import sys, os
def resource(relative_path):
    base_path = getattr(
        sys,
        '_MEIPASS',
        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Splash Screen
root = tk.Tk()
root.overrideredirect(1)
root.wm_attributes('-alpha', 1, '-topmost', True)

# Background image
EQS = tk.PhotoImage(file=resource('EQS.png'))
tk.Label(root, bg='#57863C', image=EQS).pack()

root.eval('tk::PlaceWindow . Center')
root.after(2000, root.destroy)
root.update()
root.mainloop()

try:
    import pyi_splash
    pyi_splash.close()
except:
    pass


# Easy Quadrat GUI

w0 = Tk()
w0.geometry("870x500")
w0.title("Easy Quadrat")
w0.iconbitmap(resource("EQ.ico"))
w0.configure(bg='light gray')
w0.resizable(width=False, height=False)  # Disables window resizing

logo_image = PhotoImage(file=resource('Quadrat.png'))
logo_frame = Frame(w0, bg='light gray')
logo_label = Label(logo_frame, image=logo_image, bg='light gray')
logo_label.grid(row=0, column=1, padx=30, pady=3, sticky=W)

main_frame = Frame(w0, bg='light gray', padx=0, pady=70)

class FolderSelect(Frame):
    def __init__(self, parent=None, folderDescription="", **kw):
        Frame.__init__(self, master=parent, **kw)
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription, fg='dark blue', bg='light gray',
                             font=('label_font', 10, 'bold'))
        self.lblName.grid(row=1, column=0, padx=0, pady=0, rowspan=2, sticky=NSEW)
        self.entPath = Entry(self, borderwidth=3, width=24, textvariable=self.folderPath)
        self.entPath.grid(row=1, column=1, padx=0, pady=0, ipadx=0, ipady=0, sticky=NSEW)
        self.btnFind = ttk.Button(self, text="Browse", command=self.setFolderPath)
        self.btnFind.grid(row=1, column=2, padx=0, pady=0, ipadx=0, ipady=0, sticky=NSEW)

    def setFolderPath(self):
        folder_selected = filedialog.askdirectory()
        self.folderPath.set(folder_selected)

    @property
    def folder_path(self):
        return self.folderPath.get()


directory1Select = FolderSelect(main_frame, "Input ")
directory1Select.grid(row=0, column=0, columnspan=1, padx=10, pady=3, sticky=E)

directory2Select = FolderSelect(main_frame, "Output ")
directory2Select.grid(row=1, column=0, columnspan=1, rowspan=1, padx=10, pady=3, sticky=E)

cropping_frame = Frame(main_frame, bg='light gray')
cropping_frame.grid(row=2, column=0, columnspan=2, rowspan=2, padx=65, pady=30, sticky=W)

Cropping_method = Label(cropping_frame, text='Cropping Method', font='label_font 11 bold', fg="dark green",
                        bg='light gray')
Cropping_method.grid(row=0, column=0, columnspan=2, rowspan=2, padx=0, pady=5, sticky=W)

v = IntVar(value=1)
Radiobutton(cropping_frame, text='Manual', variable=v, value=1, bg='light gray').grid(row=2, column=0, columnspan=2,
                                                                                      rowspan=2, padx=15, pady=5,
                                                                                      sticky=W)
Radiobutton(cropping_frame, text='Fixed Frame', variable=v, value=2, bg='light gray').grid(row=4, column=0,
                                                                                           columnspan=2, rowspan=2,
                                                                                           padx=15, pady=5, sticky=W)
Radiobutton(cropping_frame, text='Auto (Classic I)', variable=v, value=3, bg='light gray').grid(row=6, column=0,
                                                                                              columnspan=2, rowspan=2,
                                                                                              padx=15, pady=5, sticky=W)

Radiobutton(cropping_frame, text='Auto (Classic II)', variable=v, value=4, bg='light gray').grid(row=8, column=0,
                                                                                              columnspan=2, rowspan=2,
                                                                                              padx=15, pady=5, sticky=W)

Radiobutton(cropping_frame, text='Auto (AI assisted)', variable=v, value=5, bg='light gray').grid(row=10, column=0,
                                                                                              columnspan=2, rowspan=2,
                                                                                              padx=15, pady=5, sticky=W)


buttons_frame = Frame(main_frame, bg='light gray')
buttons_frame.grid(row=4, column=0, columnspan=2, padx=60, pady=0, sticky=W)

def settings():
    global v, in_path, out_path
    in_path = directory1Select.folder_path
    out_path = directory2Select.folder_path

    if in_path == '':
        tkinter.messagebox.showwarning(title='Error!', message='Please select the input path')
    else:
        if out_path == '':
            tkinter.messagebox.showwarning(title='Error!', message='Please select the output path')
        else:
            v = v.get()
            w0.destroy()

folderPath = StringVar()

OK = ttk.Button(buttons_frame, text="OK", command=settings, style='TButton')
OK.grid(row=0, column=0, padx=10, pady=0, sticky=W)

Cancel = ttk.Button(buttons_frame, text="Cancel", command=w0.destroy, style='TButton')
Cancel.grid(row=0, column=1, padx=10, pady=0, sticky=W)


logo_frame.grid(row=0, column=2, rowspan=1, padx=0, pady=0, sticky=E)
main_frame.grid(row=0, column=0, columnspan=2, rowspan=11, padx=10, pady=5, sticky=NSEW)

def show_about_window():
    about_window = Toplevel(w0)
    about_window.title("Easy Quadrat")
    about_window.geometry("800x480")

    about_window.update_idletasks()
    width = about_window.winfo_width()
    height = about_window.winfo_height()
    x = (about_window.winfo_screenwidth() - width) // 2
    y = (about_window.winfo_screenheight() - height) // 2
    about_window.geometry(f"+{x}+{y}")

    about_text = """

Easy Quadrat is nothing but the same quadrat traditionally used in agronomy and crop science. Now, it has been brought into the field of modern phenotyping, for cropping and sampling; not from the plants themselves but rather from the images of plants.





 Developed by:

 Abbas Haghshenas
 Shiraz, Iran
 Contact: haqueshenas@gmail.com




 Copyright (C) 2024 Abbas Haghshenas

This software, Easy Quadrat version 1.0 (MIT license), includes the YOLOv8n-seg and YOLOv8m-seg models, which are subject to the terms of the AGPL-3.0 license.
-----------------------------------------------------------------------

MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-----------------------------------------------------------------------

AGPL-3.0 LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
-----------------------------------------------------------------------
For more information about the YOLOv8n-seg and YOLOv8m-seg models, visit [https://docs.ultralytics.com/tasks/segment].

"""

    header_label = Label(about_window, text="Easy Quadrat\nVersion 1.0", font=('Arial', 16, 'bold'), fg="dark green",
                         justify=CENTER)
    header_label.pack(pady=5)

    about_text_widget = scrolledtext.ScrolledText(about_window, wrap="word", width=80, height=20, font=('Arial', 10))
    about_text_widget.insert("1.0", about_text)

    about_text_widget.tag_configure("center", justify="left")
    about_text_widget.tag_add("center", "1.0", "end")

    about_text_widget.pack(pady=15, padx=30, fill="both", expand=True)

    ok_button = ttk.Button(about_window, text="OK", command=about_window.destroy, style='TButton')
    ok_button.pack(side=BOTTOM, pady=10)

    about_window.iconbitmap(resource("EQ.ico"))

w0.style = ttk.Style()
w0.style.configure('TButton.About.TButton', background='light gray', foreground='grey')
About_button = ttk.Button(main_frame, text="About", command=show_about_window, style='TButton.About.TButton')
About_button.grid(row=5, column=0, padx=5, pady=30, sticky=W)

w0.grid_rowconfigure(0, weight=1)
w0.grid_columnconfigure(0, weight=1)
w0.eval('tk::PlaceWindow . center')
w0.mainloop()

# ----------------------

# Method: Manual

# ----------------------

if v == 1:

    Segmented_images = out_path + '/Segmented images'
    Cropped_images = out_path + '/Cropped images'
    Labels = out_path + '/Labels'

    try:
        os.mkdir(Cropped_images)
        os.mkdir(Segmented_images)
        os.mkdir(Labels)
    except:
        pass

    class Manual:
        def __init__(self, w1):
            self.root = w1
            self.root.title(" Easy Quadrat _ Manual")
            w1.iconbitmap(resource("EQ.ico"))
            window_width = 1000
            window_height = 750
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = int((screen_width / 2) - (window_width / 2))
            y = int((screen_height / 2) - (window_height / 2))
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.canvas = None
            self.in_path = in_path
            self.num_points = 4  # Default number of points
            self.image_files = []
            self.current_image = 0
            self.total_images = 0
            self.image = None
            self.points = []
            self.handle_orientation = tk.BooleanVar()
            self.handle_orientation.set(False)
            self.markers = []
            self.rectangles = []
            self.lines = []

            self.top_frame = tk.Frame(self.root)
            self.top_frame.pack(side=tk.TOP, padx=5, pady=0, fill=tk.BOTH)

            self.reset_button = tk.Button(self.top_frame, text=" Reset ", command=self.reset, state=tk.DISABLED)
            self.reset_button.pack(side=tk.LEFT, padx=5, pady=0)

            self.skip_button = tk.Button(self.top_frame, text=" Skip ", command=self.skip_image)
            self.skip_button.pack(side=tk.LEFT, padx=5, pady=0)

            self.jump_button = tk.Button(self.top_frame, text=" Jump ", command=self.jump_to_image)
            self.jump_button.pack(side=tk.LEFT, padx=5, pady=0)

            self.crop_button = tk.Button(self.top_frame, text="  Crop  ", command=self.crop_image, state=tk.DISABLED)
            self.crop_button.pack(side=tk.LEFT, padx=5, pady=0)

            self.zoom_scale = tk.Scale(self.top_frame, from_=1, to=10, orient=tk.HORIZONTAL, resolution=0.1,
                                       command=self.zoom_image)
            self.zoom_scale.set(1)
            self.zoom_scale.pack(side=tk.RIGHT, padx=(0, 20))

            self.zoom_label = tk.Label(self.top_frame, text="Zoom")
            self.zoom_label.pack(side=tk.RIGHT, padx=0)

            # HLO Checkbutton
            self.handle_orientation_checkbutton = tk.Checkbutton(self.top_frame, variable=self.handle_orientation)
            self.handle_orientation_checkbutton.pack(side=tk.RIGHT, padx=0)

            self.hlo_label = tk.Label(self.top_frame, text="HLO:")
            self.hlo_label.pack(side=tk.RIGHT, padx=0)

            # Combobox to select the background color
            self.bg_color_combobox = ttk.Combobox(self.top_frame, values=["Color"], width=5,
                                                  state="readonly")
            self.bg_color_combobox.current(0)  # Set the default background color to Black
            self.bg_color_combobox.pack(side=tk.RIGHT, padx=5)
            self.bg_color_combobox.bind("<Button-1>", lambda event: self.open_custom_dropdown_bg())
            self.selected_bg_color = "Black"

            self.bg_label = tk.Label(self.top_frame, text="BG:")
            self.bg_label.pack(side=tk.RIGHT, padx=0)

            self.color_combobox = ttk.Combobox(self.top_frame, values=["Color"], width=5, state="readonly")
            self.color_combobox.current(0)  # Set the default color to black
            self.color_combobox.pack(side=tk.RIGHT, padx=5)
            self.color_combobox.bind("<Button-1>", lambda event: self.open_custom_dropdown())
            self.selected_color = "Red"

            self.point_combobox = ttk.Combobox(self.top_frame,
                                               values=["2-Sq."] + [str(i) for i in range(3, 21)],
                                               width=5, state="readonly")
            self.point_combobox.current(0)  # Set the default value to "2-Square"
            self.point_combobox.pack(side=tk.RIGHT, padx=5)
            self.point_combobox.bind("<<ComboboxSelected>>", self.update_num_points)

            self.point_label = tk.Label(self.top_frame, text="Points")
            self.point_label.pack(side=tk.RIGHT, padx=0)

            self.info_label = tk.Label(self.top_frame, text="", font="default 10 bold", fg="blue", anchor='w')
            self.info_label.pack(side=tk.LEFT, padx=20, pady=0, fill=tk.X, expand=True)

            self.canvas_frame = tk.Frame(self.root)
            self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.canvas_scrollbar_x = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
            self.canvas_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

            self.canvas_scrollbar_y = ttk.Scrollbar(self.canvas_frame)
            self.canvas_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

            self.canvas = tk.Canvas(self.canvas_frame, bg='#F0F0F0', xscrollcommand=self.canvas_scrollbar_x.set,
                                    yscrollcommand=self.canvas_scrollbar_y.set)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.canvas_scrollbar_x.config(command=self.canvas.xview)
            self.canvas_scrollbar_y.config(command=self.canvas.yview)

            self.root.bind("<Return>", lambda event: self.crop_image())
            self.root.bind("<space>", lambda event: self.skip_image())

            self.canvas.bind("<Control-Button-4>", lambda event: self.zoom_image(self.zoom_scale.get() + 0.1))
            self.canvas.bind("<Control-Button-5>", lambda event: self.zoom_image(self.zoom_scale.get() - 0.1))
            self.canvas.bind("<MouseWheel>", self.zoom_with_scroll)

            self.panning = False
            self.canvas.bind("<Button-3>", self.start_panning)
            self.root.bind("<Escape>", lambda event: self.reset())

            self.canvas.bind("<B3-Motion>", self.pan_image)
            self.canvas.bind("<ButtonRelease-3>", self.stop_panning)

        def check_image_list(self):
            if self.current_image >= self.total_images:
                self.canvas.delete("all")
                self.canvas.create_text(10, 10, anchor=tk.NW, text="No more image.", fill="black")

                self.info_label.config(text="No more image!")

                messagebox.showinfo("Completed!", "Completed! No more image.")

                self.root.quit()

        def zoom_with_scroll(self, event):
            if self.image:
                current_zoom = float(self.zoom_scale.get())
                delta = event.delta

                if event.state & 0x0004:  # Ctrl key is pressed
                    if delta > 0:
                        # Zoom in
                        new_zoom = current_zoom + 0.1
                    else:
                        # Zoom out
                        new_zoom = current_zoom - 0.1

                    self.zoom_scale.set(new_zoom)
                    self.zoom_image(new_zoom)

        def open_folder(self):
            if self.in_path:
                self.image_files = sorted(
                    [f for f in os.listdir(self.in_path) if os.path.isfile(os.path.join(self.in_path, f))])
                self.total_images = len(self.image_files)
                self.open_image()

        def open_image(self):
            self.check_image_list()
            if self.current_image < self.total_images:
                image_file = self.image_files[self.current_image]
                self.image = Image.open(os.path.join(self.in_path, image_file))
                self.zoom_scale.set(1)
                self.show_image(image_file, self.current_image + 1, self.total_images)
            else:
                self.canvas.delete("all")
                self.canvas.pack_forget()
                self.reset_button.config(state=tk.DISABLED)
                self.skip_button.config(state=tk.DISABLED)
                self.crop_button.config(state=tk.DISABLED)

        def show_image(self, image_file, current_image, total_images):
            if self.image:
                global image2
                width, height = self.calculate_image_size()

                image2 = self.image

                self.image = self.image.resize((width, height), Image.LANCZOS)

                self.tk_image = ImageTk.PhotoImage(self.image)

                self.canvas.config(scrollregion=(0, 0, width, height))
                self.canvas.delete("all")

                self.canvas.create_text(10, 10, anchor=tk.NW, text="Image...", fill="black")

                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

                if len(self.points) != self.num_points:
                    self.canvas.bind("<Button-1>", lambda event: self.get_point(event.x, event.y))
                    self.crop_button.config(state=tk.DISABLED)

                self.info_label.config(text="Image {0} / {1} : {2}".format(current_image, total_images, image_file))
            else:
                self.canvas.delete("all")
                self.canvas.create_text(10, 10, anchor=tk.NW, text="Completed! No more image.",
                                        fill="black")
                return

        def calculate_image_size(self):
            if self.image:
                window_width = self.root.winfo_width()
                window_height = self.root.winfo_height()
                border = 20
                max_width = window_width - (2 * border)
                max_height = window_height - (2 * border)

                image_ratio = self.image.width / self.image.height
                window_ratio = max_width / max_height

                if image_ratio > window_ratio:
                    width = max_width
                    height = int(max_width / image_ratio)
                else:
                    width = int(max_height * image_ratio)
                    height = max_height

                return width, height

        def zoom_image(self, value):
            if self.image:
                original_width, original_height = self.image.size
                zoom_level = float(value)

                # Calculate new width and height after zooming
                width = int(original_width * zoom_level)
                height = int(original_height * zoom_level)

                # Resize the image
                resized_image = self.image.resize((width, height), Image.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(resized_image)
                self.canvas.config(scrollregion=(0, 0, width, height))
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

                for i, point in enumerate(self.points):
                    original_x = int(point[0] * zoom_level)
                    original_y = int(point[1] * zoom_level)
                    self.canvas.coords(self.markers[i], original_x - 4, original_y - 4, original_x + 4, original_y + 4)
                self.reset()

        def update_selected_color(self, event):
            self.selected_color = self.color_combobox.get()
            self.close_custom_dropdown()
            self.open_custom_dropdown()

        def open_custom_dropdown(self):
            x, y = self.color_combobox.winfo_rootx(), self.color_combobox.winfo_rooty()
            height = self.color_combobox.winfo_height()

            self.color_dropdown = tk.Toplevel(self.root)
            self.color_dropdown.wm_overrideredirect(True)
            self.color_dropdown.wm_geometry(f"+{x}+{y + height}")

            color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Purple", "Black", "White"]
            colors = ["red", "green", "blue", "yellow", "magenta", "purple", "black", "white"]

            for i, color in enumerate(color_names):
                btn = tk.Button(self.color_dropdown, text=color, width=10, background=colors[i],
                                command=lambda c=colors[i]: self.change_color(c))
                btn.pack(fill=tk.X)

            self.color_dropdown.focus_set()

        def close_custom_dropdown(self):
            if hasattr(self, "color_dropdown"):
                self.color_dropdown.destroy()
                del self.color_dropdown

        def change_color(self, color):
            self.selected_color = color
            self.color_combobox.set(color.capitalize())
            self.close_custom_dropdown()

        def find_other_vertices(self, x1, y1, x2, y2):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.atan2(y2 - y1, x2 - x1)
            offset = distance / 2

            vertex2_x = center_x + offset * math.cos(angle - math.pi / 2)
            vertex2_y = center_y + offset * math.sin(angle - math.pi / 2)
            vertex3_x = center_x + offset * math.cos(angle + math.pi / 2)
            vertex3_y = center_y + offset * math.sin(angle + math.pi / 2)

            return (vertex2_x, vertex2_y), (vertex3_x, vertex3_y)

        def get_point(self, x, y):
            if len(self.points) == self.num_points:
                return

            canvas_x = self.canvas.canvasx(x)
            canvas_y = self.canvas.canvasy(y)

            original_width, original_height = self.image.size

            canvas_width, canvas_height = self.calculate_image_size()
            x_ratio = original_width / canvas_width
            y_ratio = original_height / canvas_height

            original_x = int(canvas_x * x_ratio)
            original_y = int(canvas_y * y_ratio)

            point_id = self.canvas.create_oval(canvas_x - 4, canvas_y - 4, canvas_x + 4, canvas_y + 4,
                                               outline=self.selected_color, width=2)

            self.points.append((original_x, original_y))
            self.markers.append(point_id)

            # For 2-Square selection
            if len(self.points) == 2 and self.point_combobox.get() == "2-Sq.":
                opposite_vertices = self.find_other_vertices(original_x, original_y, self.points[0][0],
                                                              self.points[0][1])
                vertex2, vertex3 = opposite_vertices

                vertex2_id = self.canvas.create_oval(vertex2[0] / x_ratio, vertex2[1] / y_ratio,
                                                     vertex2[0] / x_ratio, vertex2[1] / y_ratio,
                                                     outline=self.selected_color, width=2)
                vertex3_id = self.canvas.create_oval(vertex3[0] / x_ratio, vertex3[1] / y_ratio,
                                                     vertex3[0] / x_ratio, vertex3[1] / y_ratio,
                                                     outline=self.selected_color, width=2)

                self.points.append(vertex2)
                self.points.append(vertex3)
                self.markers.append(vertex2_id)
                self.markers.append(vertex3_id)

                self.points.sort(key=lambda p: p[0])
                min_y = min(self.points, key=lambda p: p[1])[1]
                self.points = [p for p in self.points if p[1] == min_y] + [p for p in self.points if p[1] != min_y]

                self.connect_points()

                self.crop_button.config(state=tk.NORMAL)
                self.canvas.unbind("<Button-1>")
                return

            self.reset_button.config(state=tk.NORMAL)

            if len(self.points) == self.num_points:
                self.canvas.unbind("<Button-1>")
                self.crop_button.config(state=tk.NORMAL)
                self.reset_button.config(state=tk.NORMAL)
                self.skip_button.config(state=tk.NORMAL)
                self.connect_points()
                self.reset_button.focus()

        def update_num_points(self, event):
            selected_option = self.point_combobox.get()
            if selected_option == "2-Sq.":
                self.num_points = 4
            else:
                self.num_points = int(selected_option)
            self.reset()

        def connect_points(self):
            for i in range(len(self.points) - 1):
                line = self.canvas.create_line(self.points[i][0], self.points[i][1], self.points[i + 1][0],
                                               self.points[i + 1][1], fill=self.selected_color, width=3)
                self.lines.append(line)
                self.lines.append(line)
            line = self.canvas.create_line(self.points[-1][0], self.points[-1][1], self.points[0][0], self.points[0][1],
                                           fill=self.selected_color, width=3)

            self.lines.append(line)

        def start_panning(self, event):
            self.panning = True
            self.prev_x = event.x
            self.prev_y = event.y

        def pan_image(self, event):
            if self.panning:
                current_x, current_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
                delta_x = event.x - self.prev_x
                delta_y = event.y - self.prev_y

                zoom_level = self.zoom_scale.get()

                scroll_amount_x = delta_x / (zoom_level * 500)  # Adjust divisor for sensitivity
                scroll_amount_y = delta_y / (zoom_level * 500)  # Adjust divisor for sensitivity

                new_x_pos = max(0, min(1, self.canvas.xview()[0] - scroll_amount_x))
                new_y_pos = max(0, min(1, self.canvas.yview()[0] - scroll_amount_y))

                self.canvas.xview_moveto(new_x_pos)
                self.canvas.yview_moveto(new_y_pos)

                self.prev_x = event.x
                self.prev_y = event.y

        def stop_panning(self, event):
            self.panning = False

        def reset(self):
            for marker in self.markers:
                self.canvas.delete(marker)

            for rectangle in self.rectangles:
                self.canvas.delete(rectangle)

            for line in self.lines:
                self.canvas.delete(line)

            self.points = []
            self.markers = []
            self.rectangles = []
            self.lines = []
            self.reset_button.config(state=tk.DISABLED)
            self.crop_button.config(state=tk.DISABLED)
            self.skip_button.config(state=tk.NORMAL)
            self.canvas.bind("<Button-1>", lambda event: self.get_point(event.x, event.y))

        def skip_image(self):
            self.current_image += 1
            self.check_image_list()
            if self.current_image < self.total_images:
                self.open_image()
            else:
                self.info_label.config(text="No more image!")  # Update info label

            markers_copy = list(self.markers)
            rectangles_copy = list(self.rectangles)
            lines_copy = list(self.lines)

            for marker in markers_copy:
                self.canvas.delete(marker)

            for rectangle in rectangles_copy:
                self.canvas.delete(rectangle)

            for line in lines_copy:
                self.canvas.delete(line)
            self.reset()

        def jump_to_image(self):
            def jump():
                try:
                    image_number = int(entry.get())
                    if 1 <= image_number <= self.total_images:
                        self.current_image = image_number - 1
                        self.open_image()
                        top.destroy()
                    else:
                        messagebox.showerror("Error", "Please enter a valid image number.")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number.")
                self.reset()

            def on_return(event):
                jump()

            top = tk.Toplevel(self.root)
            top.title("Jump to Image")
            top.iconbitmap(resource("EQ.ico"))

            label = tk.Label(top, text="Enter Image Number:", font=("Arial", 10, "bold"))
            label.pack(pady=10)

            entry = tk.Entry(top, font=("Arial", 12))
            entry.pack(ipady=5)

            button = tk.Button(top, text="Jump", command=jump, font=("Arial", 10), bg="green", fg="white")
            button.pack(pady=25)

            entry.bind("<Return>", on_return)

            entry.focus_set()

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = 300
            window_height = 150
            x = int((screen_width / 2) - (window_width / 2))
            y = int((screen_height / 2) - (window_height / 2))
            top.geometry(f"{window_width}x{window_height}+{x}+{y}")

        def update_bg_color(self, event):
            self.selected_bg_color = self.bg_color_combobox.get()
            self.close_custom_dropdown_bg()
            self.open_custom_dropdown_bg()

        def open_custom_dropdown_bg(self):
            x, y = self.bg_color_combobox.winfo_rootx(), self.bg_color_combobox.winfo_rooty()
            height = self.bg_color_combobox.winfo_height()

            self.color_dropdown_bg = tk.Toplevel(self.root)
            self.color_dropdown_bg.wm_overrideredirect(True)
            self.color_dropdown_bg.wm_geometry(f"+{x}+{y + height}")

            color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Purple", "Black", "White"]
            colors = ["red", "green", "blue", "yellow", "magenta", "purple", "black", "white"]

            for i, color in enumerate(color_names):
                btn = tk.Button(self.color_dropdown_bg, text=color, width=10, background=colors[i],
                                command=lambda c=colors[i]: self.change_bg_color(c))
                btn.pack(fill=tk.X)

            self.color_dropdown_bg.focus_set()

        def close_custom_dropdown_bg(self):
            if hasattr(self, "color_dropdown_bg"):
                self.color_dropdown_bg.destroy()
                del self.color_dropdown_bg

        def change_bg_color(self, color):
            self.selected_bg_color = color
            self.bg_color_combobox.set(color.capitalize())
            self.close_custom_dropdown_bg()

        def crop_image(self):
            if self.image:
                if len(self.points) == self.num_points:
                    display_width, display_height = self.calculate_image_size()

                    image2_width, image2_height = image2.size

                    width_scale = image2_width / display_width
                    height_scale = image2_height / display_height

                    zoom_level = self.zoom_scale.get()
                    left = int(min(self.points, key=lambda p: p[0])[0] * width_scale / zoom_level)
                    upper = int(min(self.points, key=lambda p: p[1])[1] * height_scale / zoom_level)
                    right = int(max(self.points, key=lambda p: p[0])[0] * width_scale / zoom_level)
                    lower = int(max(self.points, key=lambda p: p[1])[1] * height_scale / zoom_level)

                    cropped_image = image2.crop((left, upper, right, lower))

                    if Cropped_images:
                        original_filename = os.path.basename("C_" + self.image_files[self.current_image])
                        output_image_path = os.path.join(Cropped_images, original_filename)

                        cropped_image.save(output_image_path, format=self.image.format)

                    mask = Image.new('L', image2.size, color=0)

                    draw = ImageDraw.Draw(mask)
                    draw.polygon(
                        [(point[0] * width_scale / zoom_level, point[1] * height_scale / zoom_level) for point in
                         self.points],
                        fill=255)

                    masked_image = Image.composite(image2, Image.new('RGB', image2.size, color=self.selected_bg_color),
                                                   mask)

                    black_image = Image.new('RGB', image2.size, color='black')

                    black_image.paste(masked_image, (0, 0))

                    original_format = image2.format

                    if Segmented_images:
                        original_filename = os.path.basename("S_" + self.image_files[self.current_image])
                        output_image_path = os.path.join(Segmented_images, original_filename)

                        black_image.save(output_image_path, format=self.image.format)

                        zoom_level = self.zoom_scale.get()
                        scaled_points = [(int(p[0] * width_scale / zoom_level), int(p[1] * height_scale / zoom_level))
                                         for p in self.points]

                        scaled_points.append(scaled_points[0])

                        if self.handle_orientation.get():

                            orientation = 1  # Default value
                            try:
                                orientation = image2._getexif().get(0x0112, 1)
                            except:
                                pass

                            if orientation == 2:
                               # Flip horizontally
                               scaled_points = [(point[1], image2_width - point[0]) for point in scaled_points]
                            elif orientation == 3:
                               # Rotate 180 degrees
                               scaled_points = [(image2_width - point[0], image2_height - point[1]) for point in
                                                scaled_points]
                            elif orientation == 4:
                               # Flip vertically
                               scaled_points = [(point[0], image2_height - point[1]) for point in scaled_points]
                            elif orientation == 5:
                               # Rotate -90 degrees and flip horizontally
                               scaled_points = [(point[1], point[0]) for point in scaled_points]
                            elif orientation == 6:
                               # Rotate -90 degrees
                               scaled_points = [(image2_height - point[1], point[0]) for point in scaled_points]
                            elif orientation == 7:
                               # Rotate 90 degrees and flip horizontally
                               scaled_points = [(image2_height - point[1], image2_width - point[0]) for point in
                                                scaled_points]
                            elif orientation == 8:
                               # Rotate 90 degrees
                               scaled_points = [(point[1], image2_width - point[0]) for point in scaled_points]


                        segmentations = [list(point) for point in scaled_points]

                        filename_without_prefix = os.path.basename(self.image_files[self.current_image])

                        # Create the COCO JSON data
                        coco_data = {
                            "images": [
                                {
                                    "file_name": filename_without_prefix,
                                    "id": self.current_image + 1,  # Assuming image IDs start from 1
                                    "width": image2_width,
                                    "height": image2_height
                                }
                            ],
                            "annotations": [
                                {
                                    "segmentation": [segmentations],
                                    "iscrowd": 0,
                                    "image_id": self.current_image + 1,  # Assuming image IDs start from 1
                                    "bbox": [min(p[0] for p in scaled_points), min(p[1] for p in scaled_points),
                                             max(p[0] for p in scaled_points) - min(p[0] for p in scaled_points),
                                             max(p[1] for p in scaled_points) - min(p[1] for p in scaled_points)],
                                    "category_id": 1,  # Adjust category ID as needed
                                    "id": 1  # Assuming annotation IDs start from 1
                                }
                            ],
                            "categories": [
                                {
                                    "supercategory": "Quadrat",
                                    "id": 1,  # Adjust category ID as needed
                                    "name": "Quadrat"  # Adjust category name as needed
                                }
                            ]
                        }

                        # Convert the dictionary to JSON
                        coco_json = json.dumps(coco_data, indent=2)

                        # Saving the JSON data
                        json_filename = filename_without_prefix.replace(
                            filename_without_prefix.split(".")[-1], "json")
                        json_filepath = os.path.join(Labels, json_filename)
                        with open(json_filepath, "w") as json_file:
                            json_file.write(coco_json)

                        # Convert the segmentation points to YOLO format (.txt)
                        yolo_format = []
                        for point in scaled_points:
                            x, y = point
                            x /= image2_width  # Normalize x
                            y /= image2_height  # Normalize y
                            yolo_format.extend([x, y])

                        # Write labels file
                        txt_filename = filename_without_prefix.replace(filename_without_prefix.split(".")[-1], "txt")
                        txt_filepath = os.path.join(Labels, txt_filename)

                        class_id = 0  # Replace with your actual class ID

                        with open(txt_filepath, "w") as txt_file:
                            txt_file.write(f"{class_id} {' '.join(map(str, yolo_format))}\n")

                        self.current_image += 1
                        self.check_image_list()
                        if self.current_image < self.total_images:
                            self.open_image()
                        else:
                            self.info_label.config(text="No more image!")

                        for rectangle in self.rectangles:
                            self.canvas.delete(rectangle)
                        self.reset()


    w1 = tk.Tk()
    w1.resizable(width=False, height=False)
    app = Manual(w1)
    w1.update()

    window_width = 1000
    window_height = 750
    screen_width = w1.winfo_screenwidth()
    screen_height = w1.winfo_screenheight()
    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))
    w1.geometry(f"{window_width}x{window_height}+{x}+{y}")

    app.open_folder()
    w1.mainloop()

# ----------------------

# Method: Fixed frame

# ----------------------

elif v == 2:

    class ProgressWindow(tk.Toplevel):
        def __init__(self, parent, total, title="Progress"):
            super().__init__(parent)
            self.title(title)
            self.iconbitmap(resource("EQ.ico"))
            self.geometry("400x150")

            window_width = 400
            window_height = 120
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x_coordinate = int((screen_width - window_width) / 2)
            y_coordinate = int((screen_height - window_height) / 2)
            self.geometry(f"+{x_coordinate}+{y_coordinate}")

            self.init_label = tk.Label(self, text="Initializing... Please wait!", font=('default', 10, 'bold'))
            self.init_label.pack(pady=25)

            self.status_label = tk.Label(self, text="", font=('default', 10, 'bold'))
            self.status_label.pack(pady=25)

            self.progress_var = tk.IntVar()
            self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=total, length=300)
            self.progress_bar.pack(pady=10)
            self.resizable(width=False, height=False)

            self.update()

        def update_progress(self, value, current_file):
            if hasattr(self, 'init_label'):
                self.init_label.destroy()
                del self.init_label

            percentage = int((value / self.progress_bar["maximum"]) * 100)
            status_text = f"Processing image {value} / {self.progress_bar['maximum']}: {current_file} ({percentage}%)"
            self.status_label.config(text=status_text)
            self.progress_var.set(value)
            self.update()

    class FixedFrame:
        def __init__(self, root):
            self.root = root
            self.left = 0
            self.right = 100
            self.top = 0
            self.bottom = 100
            self.orientation_var = tk.StringVar(value="horizontal")
            self.create_widgets()

        def create_widgets(self):
            self.frame = tk.Frame(self.root, padx=10, pady=10)
            self.frame.pack()

            self.label_message = tk.Label(self.frame, text="Set the cropping frame",
                                          font="Default 11 bold", fg="dark green")
            self.label_message.grid(row=0, column=0, padx=10, sticky="W")

            self.label_left = tk.Label(self.frame, text="Left (%)")
            self.label_left.grid(row=2, column=0, columnspan=1, padx=1, sticky="W")

            self.left_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                        command=self.set_left)
            self.left_slider.grid(row=2, column=0, columnspan=1, padx=60, pady=0, sticky="NW")
            self.left_slider.set(0)

            self.left_entry = tk.Entry(self.frame, width=3, validate='key')
            self.left_entry.grid(row=2, column=0, columnspan=1, padx=166, pady=17, sticky="W")
            self.left_entry.insert(tk.END, self.left)
            self.left_entry.bind("<FocusOut>", lambda event: self.update_left_slider())

            self.label_right = tk.Label(self.frame, text="Right (%)")
            self.label_right.grid(row=3, column=0, sticky="W")

            self.right_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_right)
            self.right_slider.grid(row=3, column=0, columnspan=1, padx=60, sticky="NW")
            self.right_slider.set(100)

            self.right_entry = tk.Entry(self.frame, width=3, validate='key')
            self.right_entry.grid(row=3, column=0, columnspan=1, padx=166, pady=17, sticky="W")
            self.right_entry.insert(tk.END, self.right)
            self.right_entry.bind("<FocusOut>", lambda event: self.update_right_slider())

            self.label_top = tk.Label(self.frame, text="Top (%)")
            self.label_top.grid(row=4, column=0, columnspan=1, rowspan=1, padx=1, pady=5, sticky="WN")

            self.top_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.VERTICAL,
                                       command=self.set_top)
            self.top_slider.grid(row=4, column=0, columnspan=1, rowspan=1, padx=1, pady=60, sticky="WN")
            self.top_slider.set(0)

            self.top_entry = tk.Entry(self.frame, width=3, validate='key')
            self.top_entry.grid(row=4, column=0, columnspan=1, rowspan=1, padx=25, pady=35, sticky="WN")
            self.top_entry.insert(tk.END, self.top)
            self.top_entry.bind("<FocusOut>", lambda event: self.update_top_slider())

            self.label_bottom = tk.Label(self.frame, text="Bottom (%)")
            self.label_bottom.grid(row=4, column=0, columnspan=1, rowspan=1, padx=100, pady=5, sticky="WN")

            self.bottom_slider = tk.Scale(self.frame, from_=0, to=100, orient=tk.VERTICAL, command=self.set_bottom)
            self.bottom_slider.grid(row=4, column=0, columnspan=1, rowspan=1, padx=100, pady=60, sticky="NEW")
            self.bottom_slider.set(100)

            self.bottom_entry = tk.Entry(self.frame, width=3, validate='key')
            self.bottom_entry.grid(row=4, column=0, columnspan=1, rowspan=1, padx=122, pady=35, sticky="WN")
            self.bottom_entry.insert(tk.END, self.bottom)
            self.bottom_entry.bind("<FocusOut>", lambda event: self.update_bottom_slider())

            self.orientation_frame = tk.Frame(self.frame)
            self.orientation_frame.grid(row=4, column=0, padx=240, pady=200, sticky="W")

            tk.Label(self.orientation_frame, text="Adjust Initial Orientation:").grid(row=0, column=1, sticky="W")

            tk.Radiobutton(self.orientation_frame, text="Original", variable=self.orientation_var,
                           value="unchanged").grid(row=0, column=2, padx=5, sticky="W")
            tk.Radiobutton(self.orientation_frame, text="Horizontal", variable=self.orientation_var,
                           value="horizontal").grid(row=0, column=3, padx=5, sticky="W")
            tk.Radiobutton(self.orientation_frame, text="Vertical", variable=self.orientation_var,
                           value="vertical").grid(row=0, column=4, padx=5, sticky="W")

            # Green screen to simulate the image
            self.canvas = tk.Canvas(self.frame, background="black", width=400, height=300)
            self.canvas.grid(row=2, column=0, columnspan=100, rowspan=20, padx=235, pady=1, sticky="EN")
            self.btn_crop = tk.Button(self.frame, text="     Crop     ", font="default 12 bold", fg="dark green",
                                      command=self.crop_images)
            self.btn_crop.grid(row=4, column=0, columnspan=1, rowspan=5, padx=35, pady=200, sticky="WN")

            self.root.bind('<Return>', self.crop_images)

        def set_left(self, value):
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value > self.right:
                    value = self.right
                self.left = value
                self.update_slider(self.left_slider, self.left)
                self.update_canvas()
                self.update_left_entry()

        def update_left_entry(self):
            self.left_entry.delete(0, tk.END)
            self.left_entry.insert(tk.END, self.left)

        def update_left_slider(self):
            value = self.left_entry.get()
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value > self.right:
                    value = self.right
                self.left = value
                self.update_slider(self.left_slider, self.left)
                self.update_canvas()

        def set_right(self, value):
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value < self.left:
                    value = self.left
                self.right = value
                self.update_slider(self.right_slider, self.right)
                self.update_canvas()
                self.update_right_entry()

        def update_right_entry(self):
            self.right_entry.delete(0, tk.END)
            self.right_entry.insert(tk.END, self.right)

        def update_right_slider(self):
            value = self.right_entry.get()
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value < self.left:
                    value = self.left
                self.right = value
                self.update_slider(self.right_slider, self.right)
                self.update_canvas()

        def set_top(self, value):
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value > self.bottom:
                    value = self.bottom
                self.top = value
                self.update_slider(self.top_slider, self.top)
                self.update_canvas()
                self.update_top_entry()

        def update_top_entry(self):
            self.top_entry.delete(0, tk.END)
            self.top_entry.insert(tk.END, self.top)

        def update_top_slider(self):
            value = self.top_entry.get()
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value > self.bottom:
                    value = self.bottom
                self.top = value
                self.update_slider(self.top_slider, self.top)
                self.update_canvas()

        def set_bottom(self, value):
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value < self.top:
                    value = self.top
                self.bottom = value
                self.update_slider(self.bottom_slider, self.bottom)
                self.update_canvas()
                self.update_bottom_entry()

        def update_bottom_entry(self):
            self.bottom_entry.delete(0, tk.END)
            self.bottom_entry.insert(tk.END, self.bottom)

        def update_bottom_slider(self):
            value = self.bottom_entry.get()
            if value.isdigit() and len(value) <= 3:
                value = int(value)
                if value < self.top:
                    value = self.top
                self.bottom = value
                self.update_slider(self.bottom_slider, self.bottom)
                self.update_canvas()

        def update_slider(self, slider, value):
            slider.set(value)

        def update_canvas(self):
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, 400, 300, fill="black", outline="black")
            self.canvas.create_rectangle(self.left * 4, self.top * 3,
                                         self.right * 4, self.bottom * 3, fill="green")

        def crop_images(self, event=None):
            if self.left != self.right and self.top != self.bottom:
                progress_window = ProgressWindow(self.root, total=len(os.listdir(in_path)), title="Cropping Images...")

                for i, file in enumerate(os.listdir(in_path)):
                    try:
                        f_img = os.path.join(in_path, file)
                        img = Image.open(f_img)

                        original_orientation = "vertical" if img.height > img.width else "horizontal"

                        if self.orientation_var.get() == "vertical" and original_orientation == "horizontal":
                            img = img.rotate(90, expand=True)
                        elif self.orientation_var.get() == "horizontal" and original_orientation == "vertical":
                            img = img.rotate(90, expand=True)

                        w, h = img.size
                        cropped_img = img.crop((round(self.left * w / 100), round(self.top * h / 100),
                                                round(self.right * w / 100), round(self.bottom * h / 100)))
                        f_out = os.path.join(out_path, "C_" + file)
                        cropped_img.save(f_out)
                        progress_window.update_progress(i + 1, file)
                    except Exception as e:
                        messagebox.showerror("Error", f"An error occurred: {str(e)}")

                progress_window.destroy()
                messagebox.showinfo("Success!", "Images cropped and saved successfully!")
                self.root.destroy()
            else:
                messagebox.showwarning(title='Warning!', message='The cropping frame is empty!')

    w2 = tk.Tk()
    w2.geometry("700x400")
    w2.title(" Easy Quadrat _ Fixed frame")
    w2.iconbitmap(resource("EQ.ico"))

    screen_width = w2.winfo_screenwidth()
    screen_height = w2.winfo_screenheight()

    window_width = 700
    window_height = 400

    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2

    w2.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    app = FixedFrame(w2)
    w2.mainloop()


# ----------------------

# Method: Automated (Classic I)

# ----------------------

elif v == 3:

    class ProgressWindow(tk.Toplevel):
        def __init__(self, parent, total, title="Progress"):
            super().__init__(parent)
            self.title(title)
            self.iconbitmap(resource("EQ.ico"))
            self.geometry("400x150")

            window_width = 400
            window_height = 120
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x_coordinate = int((screen_width - window_width) / 2)
            y_coordinate = int((screen_height - window_height) / 2)
            self.geometry(f"+{x_coordinate}+{y_coordinate}")

            self.init_label = Label(self, text="Initializing... Please wait!", font=('default', 10, 'bold'))
            self.init_label.pack(pady=25)

            self.status_label = Label(self, text="", font=('default', 10, 'bold'))
            self.status_label.pack(pady=25)

            self.progress_var = IntVar()
            self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=total, length=300)
            self.progress_bar.pack(pady=10)
            self.resizable(width=False, height=False)

            self.update()

        def update_progress(self, value, current_file):
            if hasattr(self, 'init_label'):
                self.init_label.destroy()
                del self.init_label

            percentage = int((value / self.progress_bar["maximum"]) * 100)
            status_text = f"Processing image {value} / {self.progress_bar['maximum']}: {current_file} ({percentage}%)"
            self.status_label.config(text=status_text)
            self.progress_var.set(value)
            self.update()

    class Actual:
        def __init__(self, w1):
            self.w1 = w1
            self.w1.geometry("320x260")
            self.w1.title(" Easy Quadrat _ Classic I")
            self.w1.iconbitmap(resource("EQ.ico"))
            self.w1.resizable(width=False, height=False)  # Disables window resizing
            self.lblw1 = Label(self.w1, text="Set the values:", font=('default', 10, 'bold'))
            self.lblw1.grid(row=1, column=1, padx=40, pady=15, sticky="w")
            self.lblw1 = DoubleVar()
            self.ent1w1 = DoubleVar()
            self.ent2w1 = IntVar()
            self.ent3w1 = IntVar()
            self.ent1w1.set(0.1)
            self.ent2w1.set(40)
            self.ent3w1.set(220)

            self.p1w1_lable = tkinter.Label(self.w1, text='Saturation (0-1): ', anchor="w")
            self.p1w1_ent = tkinter.Entry(self.w1, textvariable=self.ent1w1, borderwidth=3, width=6)
            self.p2w1_lable = tkinter.Label(self.w1, text='Closing radius: ', anchor="w")
            self.p2w1_ent = tkinter.Entry(self.w1, textvariable=self.ent2w1, borderwidth=3, width=6)
            self.p3w1_lable = tkinter.Label(self.w1, text='Opening radius: ', anchor="w")
            self.p3w1_ent = tkinter.Entry(self.w1, textvariable=self.ent3w1, borderwidth=3, width=6)

            self.bg_color_combobox = ttk.Combobox(self.w1, values=["Color"], width=5, state="readonly")
            self.bg_color_combobox.current(0)
            self.bg_color_combobox.grid(row=7, column=2, columnspan=1, padx=1, pady=20, sticky="n")
            self.bg_color_combobox.bind("<Button-1>", lambda event: self.open_custom_dropdown_bg())
            self.selected_bg_color = "Black"

            self.bg_label = tk.Label(self.w1, text="Background Color:")
            self.bg_label.grid(row=7, column=1, columnspan=1, padx=40, pady=20, sticky="w")

            self.w1OK = ttk.Button(self.w1, text="OK", command=self.entw1)
            self.w1OK.grid(row=9, column=1, padx=15, pady=20, sticky="s")

            self.w1Cancel = ttk.Button(self.w1, text="Cancel", command=self.w1.quit)
            self.w1Cancel.grid(row=9, column=2, padx=0, pady=20, sticky="w")

            self.p1w1_lable.grid(row=4, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p1w1_ent.grid(row=4, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")
            self.p2w1_lable.grid(row=5, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p2w1_ent.grid(row=5, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")
            self.p3w1_lable.grid(row=6, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p3w1_ent.grid(row=6, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")

        def update_bg_color(self, event):
            self.selected_bg_color = self.bg_color_combobox.get()
            self.close_custom_dropdown_bg()
            self.open_custom_dropdown_bg()

        def open_custom_dropdown_bg(self):
            x, y = self.bg_color_combobox.winfo_rootx(), self.bg_color_combobox.winfo_rooty()
            height = self.bg_color_combobox.winfo_height()

            self.color_dropdown_bg = tk.Toplevel(self.w1)
            self.color_dropdown_bg.wm_overrideredirect(True)
            self.color_dropdown_bg.wm_geometry(f"+{x}+{y + height}")

            color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Purple", "Black", "White"]
            colors = ["red", "green", "blue", "yellow", "magenta", "purple", "black", "white"]

            for i, color in enumerate(color_names):
                btn = tk.Button(self.color_dropdown_bg, text=color, width=10, background=colors[i],
                                command=lambda c=colors[i]: self.change_bg_color(c))
                btn.pack(fill=tk.X)

            self.color_dropdown_bg.focus_set()

        def close_custom_dropdown_bg(self):
            if hasattr(self, "color_dropdown_bg"):
                self.color_dropdown_bg.destroy()
                del self.color_dropdown_bg

        def change_bg_color(self, color):
            self.selected_bg_color = color
            self.bg_color_combobox.set(color.capitalize())
            self.close_custom_dropdown_bg()

        def entw1(self):
            if not 0 <= self.ent1w1.get() <= 1:
                tkinter.messagebox.showwarning(title='Error!', message='Please enter a value between 0 and 1')
            else:
                self.settings_values = f"\n    Saturation: {self.ent1w1.get()}\n    Closing radius: {self.ent2w1.get()}" \
                                       f" "f"\n    Opening radius: {self.ent3w1.get()}"

                self.w1.withdraw()

                progress_window = ProgressWindow(self.w1, total=len(os.listdir(in_path)), title="Processing Images")

                log_file_path = os.path.join(out_path, "Log.txt")
                log_file = open(log_file_path, "w")
                log_file.write("\nEasy Quadrat Version 1.0\n..........................\n\n"
                               "Cropping method: Automated (classic I)\n\n")
                log_file.write(f"Settings: {self.settings_values}\n\n")

                try:
                    self.Segmented_images = os.path.join(out_path, 'Segmented images')
                    self.Cropped_images = os.path.join(out_path, 'Cropped images')

                    os.makedirs(self.Cropped_images)
                    os.makedirs(self.Segmented_images)
                except FileExistsError:
                    pass

                file_list = os.listdir(in_path)
                total_images = len(file_list)
                log_file.write(f"Total images: {total_images}\n\nStarting time: "
                               f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                               f"List of processed images:\n.........................\n")

                for i, file in enumerate(
                        tqdm(file_list, desc="Processing Images", unit="image", leave=False, disable=True)):
                    try:
                        f_img = os.path.join(in_path, file)
                        image_rgb = skimage.io.imread(f_img)

                        # Skip processing if the image size is too small
                        if image_rgb.size == 0 or min(image_rgb.shape[:2]) < 2:
                            continue

                        image_hsv = skimage.color.rgb2hsv(image_rgb)
                        seg = image_hsv[:, :, 1] < self.ent1w1.get()

                        seg_cleaned = skimage.morphology.isotropic_opening(seg, 1)
                        seg_cleaned = skimage.morphology.isotropic_closing(seg_cleaned, self.ent2w1.get())

                        def get_main_component(segments):
                            labels = skimage.measure.label(segments)
                            if labels.max() == 0:
                                return segments
                            return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

                        background = get_main_component(~seg_cleaned)
                        filled = ~background

                        mask = skimage.morphology.isotropic_opening(filled, self.ent3w1.get())
                        mask = skimage.segmentation.clear_border(mask)

                        masked_result = image_rgb.copy()
                        bgr_color = [int(c * 255) for c in matplotlib.colors.to_rgb(self.selected_bg_color)]
                        masked_result[~mask, :] = bgr_color

                        _, extension = os.path.splitext(file)

                        f_out1 = os.path.join(self.Segmented_images, "S_" + file)
                        cv2.imwrite(f_out1, cv2.cvtColor(masked_result, cv2.COLOR_RGB2BGR))
                        log_file.write(f"{file}\n")

                        mask_x = mask.max(axis=0)
                        mask_y = mask.max(axis=1)

                        if mask_x.any() and mask_y.any():
                            indices_x = mask_x.nonzero()[0]
                            indices_y = mask_y.nonzero()[0]
                            minx, maxx = int(indices_x[0]), int(indices_x[-1])
                            miny, maxy = int(indices_y[0]), int(indices_y[-1])

                            cropped_result = image_rgb[miny:maxy, minx:maxx, :]

                            f_out2 = os.path.join(self.Cropped_images, "C_" + file)
                            cv2.imwrite(f_out2, cv2.cvtColor(cropped_result, cv2.COLOR_RGB2BGR))

                    except Exception as e:
                        log_file.write(f"\nError: {str(e)}\n")
                        pass

                    progress_window.update_progress(i + 1, file)

                log_file.write(f".........................\n\nCompletion time: "
                               f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.close()
                progress_window.destroy()
                self.w1.destroy()
                messagebox.showinfo("Easy Quadrat", "Processing completed!")

    w1 = tk.Tk()
    w1.resizable(width=False, height=False)  # Disables window resizing
    app = Actual(w1)
    w1.update_idletasks()  # Ensure the window is fully created before centering
    w1.eval('tk::PlaceWindow . center')
    w1.mainloop()


# ----------------------

# Method: Automated (Classic II)

# ----------------------


elif v == 4:

    class ProgressWindow(tk.Toplevel):
        def __init__(self, parent, total, title="Progress"):
            super().__init__(parent)
            self.title(title)
            self.iconbitmap(resource("EQ.ico"))
            self.geometry("400x150")

            window_width = 400
            window_height = 120
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x_coordinate = int((screen_width - window_width) / 2)
            y_coordinate = int((screen_height - window_height) / 2)
            self.geometry(f"+{x_coordinate}+{y_coordinate}")

            self.init_label = Label(self, text="Initializing... Please wait!", font=('default', 10, 'bold'))
            self.init_label.pack(pady=25)

            self.status_label = Label(self, text="", font=('default', 10, 'bold'))
            self.status_label.pack(pady=25)

            self.progress_var = IntVar()
            self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=total, length=300)
            self.progress_bar.pack(pady=10)  # Adjusted padding
            self.resizable(width=False, height=False)  # Disables window resizing

            self.update()

        def update_progress(self, value, current_file):
            if hasattr(self, 'init_label'):
                self.init_label.destroy()
                del self.init_label

            percentage = int((value / self.progress_bar["maximum"]) * 100)
            status_text = f"Processing image {value} / {self.progress_bar['maximum']}: {current_file} ({percentage}%)"
            self.status_label.config(text=status_text)
            self.progress_var.set(value)
            self.update()

    class Actual:
        def __init__(self, w1):
            self.w1 = w1
            self.w1.geometry("325x260")
            self.w1.title(" Easy Quadrat _ Classic II")
            self.w1.iconbitmap(resource("EQ.ico"))
            self.w1.resizable(width=False, height=False)
            self.lblw1 = Label(self.w1, text="Set the values:", font=('default', 10, 'bold'))
            self.lblw1.grid(row=1, column=1, padx=40, pady=15, sticky="w")
            self.lblw1 = DoubleVar()
            self.ent1w1 = DoubleVar()
            self.ent2w1 = IntVar()
            self.ent3w1 = IntVar()
            self.ent1w1.set(0.1)
            self.ent2w1.set(40)
            self.ent3w1.set(300)

            self.p1w1_lable = tkinter.Label(self.w1, text='Saturation (0-1): ', anchor="w")
            self.p1w1_ent = tkinter.Entry(self.w1, textvariable=self.ent1w1, borderwidth=3, width=6)
            self.p2w1_lable = tkinter.Label(self.w1, text='Closing factor size: ', anchor="w")
            self.p2w1_ent = tkinter.Entry(self.w1, textvariable=self.ent2w1, borderwidth=3, width=6)
            self.p3w1_lable = tkinter.Label(self.w1, text='Opening factor size: ', anchor="w")
            self.p3w1_ent = tkinter.Entry(self.w1, textvariable=self.ent3w1, borderwidth=3, width=6)

            self.bg_color_combobox = ttk.Combobox(self.w1, values=["Color"], width=5, state="readonly")
            self.bg_color_combobox.current(0)
            self.bg_color_combobox.grid(row=7, column=2, columnspan=1, padx=1, pady=20, sticky="n")
            self.bg_color_combobox.bind("<Button-1>", lambda event: self.open_custom_dropdown_bg())
            self.selected_bg_color = "Black"

            self.bg_label = tk.Label(self.w1, text="Background color:")
            self.bg_label.grid(row=7, column=1, columnspan=1, padx=40, pady=20, sticky="w")

            self.w1OK = ttk.Button(self.w1, text="OK", command=self.entw1)
            self.w1OK.grid(row=9, column=1, padx=15, pady=20, sticky="s")

            self.w1Cancel = ttk.Button(self.w1, text="Cancel", command=self.w1.quit)
            self.w1Cancel.grid(row=9, column=2, padx=0, pady=20, sticky="w")

            self.p1w1_lable.grid(row=4, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p1w1_ent.grid(row=4, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")
            self.p2w1_lable.grid(row=5, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p2w1_ent.grid(row=5, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")
            self.p3w1_lable.grid(row=6, column=1, columnspan=1, rowspan=1, padx=40, pady=2, sticky="w")
            self.p3w1_ent.grid(row=6, column=2, columnspan=1, rowspan=1, padx=1, pady=2, sticky="n")

        def update_bg_color(self, event):
            self.selected_bg_color = self.bg_color_combobox.get()
            self.close_custom_dropdown_bg()
            self.open_custom_dropdown_bg()

        def open_custom_dropdown_bg(self):
            x, y = self.bg_color_combobox.winfo_rootx(), self.bg_color_combobox.winfo_rooty()
            height = self.bg_color_combobox.winfo_height()

            self.color_dropdown_bg = tk.Toplevel(self.w1)
            self.color_dropdown_bg.wm_overrideredirect(True)
            self.color_dropdown_bg.wm_geometry(f"+{x}+{y + height}")

            color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Purple", "Black", "White"]
            colors = ["red", "green", "blue", "yellow", "magenta", "purple", "black", "white"]

            for i, color in enumerate(color_names):
                btn = tk.Button(self.color_dropdown_bg, text=color, width=10, background=colors[i],
                                command=lambda c=colors[i]: self.change_bg_color(c))
                btn.pack(fill=tk.X)

            self.color_dropdown_bg.focus_set()

        def close_custom_dropdown_bg(self):
            if hasattr(self, "color_dropdown_bg"):
                self.color_dropdown_bg.destroy()
                del self.color_dropdown_bg

        def change_bg_color(self, color):
            self.selected_bg_color = color
            self.bg_color_combobox.set(color.capitalize())
            self.close_custom_dropdown_bg()

        def entw1(self):
            if not 0 <= self.ent1w1.get() <= 1:
                tkinter.messagebox.showwarning(title='Error!', message='Please enter a value between 0 and 1')
            else:
                self.settings_values = f"\n    Saturation: {self.ent1w1.get()}\n    Closing factor size: {self.ent2w1.get()}" \
                                       f" "f"\n    Opening factor size: {self.ent3w1.get()}"
                self.w1.withdraw()  # Hide the main window

                progress_window = ProgressWindow(self.w1, total=len(os.listdir(in_path)), title="Processing Images")

                log_file_path = os.path.join(out_path, "Log.txt")
                log_file = open(log_file_path, "w")
                log_file.write("\nEasy Quadrat Version 1.0\n..........................\n\n"
                               "Cropping method: Automated (classic II)\n\n")
                log_file.write(f"Settings: {self.settings_values}\n\n")

                try:
                    # Initialize paths as instance attributes
                    self.Segmented_images = os.path.join(out_path, 'Segmented images')
                    self.Cropped_images = os.path.join(out_path, 'Cropped images')

                    os.makedirs(self.Cropped_images)
                    os.makedirs(self.Segmented_images)
                except FileExistsError:
                    pass

                file_list = os.listdir(in_path)
                total_images = len(file_list)
                log_file.write(f"Total images: {total_images}\n\nStarting time: "
                               f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                               f"List of processed images:\n.........................\n")

                for i, file in enumerate(
                        tqdm(file_list, desc="Processing Images", unit="image", leave=False, disable=True)):
                    try:
                        f_img = os.path.join(in_path, file)
                        image_rgb = skimage.io.imread(f_img)

                        if image_rgb.size == 0 or min(image_rgb.shape[:2]) < 2:
                            continue

                        image_hsv = skimage.color.rgb2hsv(image_rgb)
                        seg = image_hsv[:, :, 1] < self.ent1w1.get()

                        seg_cleaned = skimage.morphology.isotropic_opening(seg, 1)
                        seg_cleaned = skimage.morphology.isotropic_closing(seg_cleaned, self.ent2w1.get())

                        def get_main_component(segments):
                            labels = skimage.measure.label(segments)
                            if labels.max() == 0:
                                return segments
                            return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

                        background = get_main_component(~seg_cleaned)
                        filled = ~background

                        mask_hor = morphology.binary_opening(filled, morphology.rectangle(self.ent3w1.get(), 1))
                        mask_hor = skimage.segmentation.clear_border(mask_hor)
                        if np.any(mask_hor):
                            mask_hor = convex_hull_image(mask_hor)
                        else:
                            mask_hor = np.zeros_like(mask_hor)

                        mask_ver = morphology.binary_opening(filled, morphology.rectangle(1, self.ent3w1.get()))
                        mask_ver = skimage.segmentation.clear_border(mask_ver)
                        if np.any(mask_ver):
                            mask_ver = convex_hull_image(mask_ver)
                        else:
                            mask_ver = np.zeros_like(mask_ver)

                        # Compare convex hull areas and choose the one with the higher area
                        area_hor = np.sum(mask_hor)
                        area_ver = np.sum(mask_ver)

                        if area_ver >= area_hor:
                            mask = mask_hor
                        else:
                            mask = mask_ver

                        masked_result = image_rgb.copy()
                        bgr_color = [int(c * 255) for c in matplotlib.colors.to_rgb(self.selected_bg_color)]
                        masked_result[~mask, :] = bgr_color

                        _, extension = os.path.splitext(file)

                        f_out1 = os.path.join(self.Segmented_images, "S_" + file)
                        cv2.imwrite(f_out1, cv2.cvtColor(masked_result, cv2.COLOR_RGB2BGR))
                        log_file.write(f"{file}\n")

                        mask_x = mask.max(axis=0)
                        mask_y = mask.max(axis=1)

                        if mask_x.any() and mask_y.any():
                            indices_x = mask_x.nonzero()[0]
                            indices_y = mask_y.nonzero()[0]
                            minx, maxx = int(indices_x[0]), int(indices_x[-1])
                            miny, maxy = int(indices_y[0]), int(indices_y[-1])

                            cropped_result = image_rgb[miny:maxy, minx:maxx, :]

                            f_out2 = os.path.join(self.Cropped_images, "C_" + file)
                            cv2.imwrite(f_out2, cv2.cvtColor(cropped_result, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        log_file.write(f"\nError: {str(e)}\n")
                        pass

                    progress_window.update_progress(i + 1, file)

                log_file.write(f".........................\n\nCompletion time: "
                               f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.close()
                progress_window.destroy()
                self.w1.destroy()
                messagebox.showinfo("Easy Quadrat", "Processing completed!")

    w1 = tk.Tk()
    w1.resizable(width=False, height=False)
    app = Actual(w1)
    w1.update_idletasks()
    w1.eval('tk::PlaceWindow . center')
    w1.mainloop()


# ----------------------

# Method: Auto (AI assisted)

# ----------------------


elif v == 5:

    class ProgressWindow(tk.Toplevel):
        def __init__(self, parent, total, title="Progress"):
            super().__init__(parent)
            self.title(title)
            self.iconbitmap(resource("EQ.ico"))
            self.geometry("400x150")

            window_width = 400
            window_height = 120
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x_coordinate = int((screen_width - window_width) / 2)
            y_coordinate = int((screen_height - window_height) / 2)
            self.geometry(f"+{x_coordinate}+{y_coordinate}")

            self.init_label = Label(self, text="Initializing... Please wait!", font=('default', 10, 'bold'))
            self.init_label.pack(pady=25)

            self.status_label = Label(self, text="", font=('default', 10, 'bold'))
            self.status_label.pack(pady=25)

            self.progress_var = IntVar()
            self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=total, length=300)
            self.progress_bar.pack(pady=10)
            self.resizable(width=False, height=False)

            self.update()

        def update_progress(self, value, current_file):
            if hasattr(self, 'init_label'):
                self.init_label.destroy()
                del self.init_label

            percentage = int((value / self.progress_bar["maximum"]) * 100)
            status_text = f"Processing image {value} / {self.progress_bar['maximum']}: {current_file} ({percentage}%)"
            self.status_label.config(text=status_text)
            self.progress_var.set(value)
            self.update()


    class Actual:
        def __init__(self, w1):
            self.w1 = w1
            self.w1.geometry("340x380")
            self.w1.title(" Easy Quadrat _ AI assisted")
            self.w1.iconbitmap(resource("EQ.ico"))
            self.w1.resizable(width=False, height=False)

            self.lblw1 = Label(self.w1, text="Model", font=('default', 11, 'bold'), foreground="#333333")
            self.lblw1.grid(row=1, column=1, padx=35, pady=15, sticky="w")

            self.model_path = None
            self.selected_bg_color = (0, 0, 0)

            self.model_var = StringVar()
            self.model_var.set("EQ1")  # Default model choice

            models = ["EQ1", "EQ2", "Custom"]

            row_counter = 3
            self.radiobtns = []
            for model in models:
                radiobtn = Radiobutton(self.w1, text=model, variable=self.model_var, value=model,
                                       command=self.toggle_model_btns)
                radiobtn.grid(row=row_counter, column=1, padx=60, pady=2, sticky="w")
                row_counter += 1
                self.radiobtns.append(radiobtn)

            self.model_btn = Button(self.w1, text="Model", command=self.browse_model, state="disabled", width=8)
            self.model_btn.grid(row=5, column=1, padx=140, pady=5, sticky="w")

            self.lblw1 = Label(self.w1, text="Settings", font=('default', 11, 'bold'), foreground="#333333")
            self.lblw1.grid(row=6, column=1, padx=35, pady=15, sticky="w")

            self.confidence_label = Label(self.w1, text="Confidence")
            self.confidence_label.grid(row=7, column=1, padx=60, pady=5, sticky="w")

            self.default_confidence = 0.25
            self.confidence_var = StringVar(value=str(self.default_confidence))
            self.confidence_entry = Entry(self.w1, textvariable=self.confidence_var, width=5)
            self.confidence_entry.grid(row=7, column=1, padx=190, pady=5, sticky="w")
            self.confidence_entry.bind("<FocusOut>", self.validate_confidence)

            self.maxnum_label = Label(self.w1, text="Max no. of quadrats")
            self.maxnum_label.grid(row=8, column=1, padx=60, pady=5, sticky="w")

            self.default_maxnum = 1
            self.maxnum_var = StringVar(value=str(self.default_maxnum))
            self.maxnum_entry = Entry(self.w1, textvariable=self.maxnum_var, width=5)
            self.maxnum_entry.grid(row=8, column=1, padx=190, pady=5, sticky="w")
            self.maxnum_entry.bind("<FocusOut>", self.validate_maxnum)

            self.bg_color_combobox = ttk.Combobox(self.w1, values=["Color"], width=5, state="readonly")
            self.bg_color_combobox.current(0)
            self.bg_color_combobox.grid(row=9, column=1, columnspan=1, padx=1, pady=5, sticky="n")
            self.bg_color_combobox.bind("<Button-1>", lambda event: self.open_custom_dropdown_bg())
            self.selected_bg_color = (0, 0, 0)

            self.bg_label = tk.Label(self.w1, text="Background color")
            self.bg_label.grid(row=9, column=1, columnspan=1, padx=60, pady=5, sticky="w")

            self.w1OK = ttk.Button(self.w1, text="OK", command=self.run_model)
            self.w1OK.grid(row=10, column=1, padx=90, pady=40, sticky="w")

            self.w1Cancel = ttk.Button(self.w1, text="Cancel", command=self.w1.quit)
            self.w1Cancel.grid(row=10, column=1, padx=180, pady=40, sticky="w")

        def toggle_model_btns(self):
            if self.model_var.get() == "Custom":
               self.model_btn.config(state="normal")
            else:
               self.model_btn.config(state="disabled")

        def browse_model(self):
            model_path = filedialog.askopenfilename(filetypes=[("Model", "*.pt")])
            self.model_path = model_path
            return

        def validate_confidence(self, event):
            try:
                confidence = float(self.confidence_var.get())
                if not 0 <= confidence <= 1:
                    messagebox.showerror("Error", "Confidence must be between 0 and 1")
                    self.confidence_var.set(str(self.default_confidence))
            except ValueError:
                messagebox.showerror("Error", "Invalid input for confidence. Please enter a valid number.")
                self.confidence_var.set(str(self.default_confidence))

        def validate_maxnum(self, event):
            try:
                maxnum = int(self.maxnum_var.get())
                if not maxnum > 1:
                    messagebox.showerror("Error", "Max no. must be between 0 and 1")
                    self.maxnum_var.set(str(self.default_maxnum))  # Reset to default
            except ValueError:
                messagebox.showerror("Error", "Invalid input for Max no. Please enter a valid number.")
                self.maxnum_var.set(str(self.default_maxnum))  # Reset to default

        def update_bg_color(self, event):
            self.selected_bg_color = self.bg_color_combobox.get()
            self.close_custom_dropdown_bg()
            self.open_custom_dropdown_bg()

        def open_custom_dropdown_bg(self):
            x, y = self.bg_color_combobox.winfo_rootx(), self.bg_color_combobox.winfo_rooty()
            height = self.bg_color_combobox.winfo_height()

            self.color_dropdown_bg = tk.Toplevel(self.w1)
            self.color_dropdown_bg.wm_overrideredirect(True)
            self.color_dropdown_bg.wm_geometry(f"+{x}+{y + height}")

            color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Purple", "Black", "White"]
            colors = ["red", "green", "blue", "yellow", "magenta", "purple", "black", "white"]

            for i, color in enumerate(color_names):
                btn = tk.Button(self.color_dropdown_bg, text=color, width=10, background=colors[i],
                                command=lambda c=colors[i]: self.change_bg_color(c))
                btn.pack(fill=tk.X)

            self.color_dropdown_bg.focus_set()

        def close_custom_dropdown_bg(self):
            if hasattr(self, "color_dropdown_bg"):
                self.color_dropdown_bg.destroy()
                del self.color_dropdown_bg

        def change_bg_color(self, color):
            color_dict = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
                          "yellow": (0, 255, 255), "magenta": (255, 0, 255),
                          "purple": (128, 0, 128), "black": (0, 0, 0), "white": (255, 255, 255)}

            self.selected_bg_color = color_dict[color]
            self.bg_color_combobox.set(color.capitalize())
            self.close_custom_dropdown_bg()

        def run_model(self):
            self.settings_values = f"\n\n    Model: {self.model_var.get()}    " \
                                   f"\n    Confidence: {self.confidence_entry.get()}\n    " \
                                   f"Max quadrats: {self.maxnum_entry.get()}\n"
            self.w1.withdraw()

            progress_window = ProgressWindow(self.w1, total=len(os.listdir(in_path)), title="Processing Images")

            log_file_path = os.path.join(out_path, "Log.txt")
            log_file = open(log_file_path, "w")
            log_file.write("\nEasy Quadrat Version 1.0\n..........................\n\n"
                           "Cropping method: Auto (AI assisted)\n\n")
            log_file.write(f"Settings: {self.settings_values}\n\n")

            try:
                self.Segmented_images = os.path.join(out_path, 'Segmented images')
                self.Cropped_images = os.path.join(out_path, 'Cropped images')

                os.makedirs(self.Cropped_images)
                os.makedirs(self.Segmented_images)
            except FileExistsError:
                pass

            file_list = os.listdir(in_path)
            total_images = len(file_list)
            log_file.write(f"Total images: {total_images}\n\nStarting time: "
                           f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                           f"List of processed images:\n.........................\n")

            model_type = self.model_var.get()
            confidence = float(self.confidence_var.get())
            max_num = int(self.maxnum_var.get())

            if model_type == "EQ1":
                model = YOLO('easyquadrat.pt')
            elif model_type == "EQ2":
                model = YOLO('easyquadrat2.pt')
            elif model_type == "Custom":
                model = YOLO(self.model_path)

        # Instead, use the below lines for PyInstaller:

            # if model_type == "EQ1":
            #     model_path = path.join(sys._MEIPASS, 'easyquadrat.pt')
            #     model = YOLO(model_path)
            # elif model_type == "EQ2":
            #     model_path = path.join(sys._MEIPASS, 'easyquadrat2.pt')
            #     model = YOLO(model_path)
            # elif model_type == "Custom":
            #     model = YOLO(self.model_path)

            for i, file in enumerate(
                    tqdm(file_list, desc="Processing Images", unit="image", leave=False, disable=True)):
                try:
                    f_img = os.path.join(in_path, file)
                    img = cv2.imread(f_img)
                    cropped_img = img.copy()

                    # Run model
                    results = model(source=f_img,
                                    task='segment',
                                    conf=confidence,
                                    max_det=max_num,
                                    save=False,
                                    retina_masks=True,
                                    show_labels=False,
                                    show_boxes=False)

                    # Create a mask for background
                    combined_background_mask = np.ones_like(img[:, :, 0], dtype=np.uint8)

                    for result in results:
                        masks = result.masks

                        # Check if masks is not None before iterating
                        if masks is not None:
                            for mask in masks.data:
                                mask_np = mask.cpu().numpy().astype(np.uint8)

                                # Resize the mask to match image dimensions
                                mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

                                # Invert the mask to get background
                                background_mask = 1 - mask_np

                                # Combine background masks
                                combined_background_mask = cv2.bitwise_and(combined_background_mask, background_mask)

                    # Apply the mask to the image with the correct color mapping
                    img[combined_background_mask > 0] = [int(val) for val in
                                                         self.selected_bg_color]  # No need to reverse the order

                    f_out1 = os.path.join(self.Segmented_images, "S_" + file)
                    cv2.imwrite(f_out1, img)

                    # Iterate over the masks and save each detected segment
                    for idx, mask in enumerate(masks.data):
                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

                        # Find the bounding box of the detected segment
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            x, y, w, h = cv2.boundingRect(contours[0])

                            # Crop the segment from the image
                            cropped_segment = cropped_img[y:y + h, x:x + w]

                            _, file_extension = os.path.splitext(file.lower())
                            f_out2 = os.path.join(self.Cropped_images,
                                                  "C_" + file.replace(file_extension, f"_{idx}{file_extension}"))
                            cv2.imwrite(f_out2, cropped_segment)

                    log_file.write(f"{file}\n")

                except Exception as e:
                    log_file.write(f"\nError: {str(e)}\n")
                    pass

                progress_window.update_progress(i + 1, file)

            log_file.write(f".........................\n\nCompletion time: "
                           f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.close()

            progress_window.destroy()
            self.w1.destroy()
            messagebox.showinfo("Easy Quadrat", "Processing completed!")


    w1 = tk.Tk()
    w1.resizable(width=False, height=False)
    app = Actual(w1)
    w1.update_idletasks()
    w1.eval('tk::PlaceWindow . center')
    w1.mainloop()