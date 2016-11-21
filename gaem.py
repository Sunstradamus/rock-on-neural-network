"""
Super janky ass rock paper scissors game
"""

#importing modules required
from ttk import *
import Tkinter as tk
from Tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import time
import threading
stop = False

last_frame = np.zeros((720, 1080, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

# # Sets image size to 1280x720 (closest to 1080x720)
cap.set(3, 1280)
cap.set(4, 720)


def back():
    global stop
    main_frame.tkraise()
    stop = True


def take_photo():
    # Shitty non blocking timer hack
    t = threading.Timer(1.5, display_rock)
    t.start()
    if mode == 0:
        start_normal.configure(state=DISABLED)
        reset_normal.configure(state=DISABLED)
        back_normal.configure(state=DISABLED)
    else:
        start_unfair.configure(state=DISABLED)
        reset_unfair.configure(state=DISABLED)
        back_unfair.configure(state=DISABLED)


def display_rock():
    if mode == 0:
        norm_text_label.configure(text="ROCK")
    else:
        unfair_text_label.configure(text="ROCK")

    t = threading.Timer(1.5, display_paper)
    t.start()

def display_paper():
    if mode == 0:
        norm_text_label.configure(text="PAPER")
    else:
        unfair_text_label.configure(text="PAPER")

    t = threading.Timer(1.5, stop_video)
    t.start()

def stop_video():
    global stop

    # TODO: Randomize or select class
    class_img = Image.open("rock.jpg")
    class_img = ImageTk.PhotoImage(class_img)
    stop = True

    # Shitty race condition fix for video to stop first
    time.sleep(0.3)

    if mode == 0:
        norm_text_label.configure(text="SCISSORS!")
        norm_main_label.configure(image=class_img)
        norm_main_label.image = class_img

    else:
        unfair_text_label.configure(text="SCISSORS!")
        unfair_main_label.configure(image=class_img)
        unfair_main_label.image = class_img

    t = threading.Timer(1.5, calculate_results)
    t.start()

def calculate_results():
    if mode == 0:
        norm_text_label.configure(text="Calculating...!")

        reset_normal.configure(state=NORMAL)
        back_normal.configure(state=NORMAL)
    else:
        unfair_text_label.configure(text="Calculating...!")

        reset_unfair.configure(state=NORMAL)
        back_unfair.configure(state=NORMAL)

    # TODO: Classify last_frame image


# Starts the normal game
def start_normal_game():
    global stop
    global norm_main_label
    global norm_text_label
    global mode

    # Set game mode to normal
    mode = 0
    # Set video flag to false
    stop = False

    start_normal.configure(state=NORMAL)
    back_normal.configure(state=NORMAL)
    reset_normal.configure(state=DISABLED)

    normal_frame.tkraise()

    norm_main_label = tk.Label(master=normal_frame, width=1080, height=720)
    norm_main_label.grid(row=0, sticky=W+E)
    norm_text_label = tk.Label(master=normal_frame, font=('', 20,), text="GET READY", foreground="WHITE", background="RED")
    norm_text_label.grid(row=1, sticky=W+E)

    show_vid()


# Starts the unfair game
def start_unfair_game():
    global stop
    global unfair_main_label
    global unfair_text_label
    global mode

    # Set game mode to unfair
    mode = 1
    # Set video flag to false
    stop = False

    start_unfair.configure(state=NORMAL)
    back_unfair.configure(state=NORMAL)
    reset_unfair.configure(state=DISABLED)

    unfair_frame.tkraise()

    unfair_main_label = tk.Label(master=unfair_frame, width=1080, height=720)
    unfair_main_label.grid(row=0, sticky=W+E)
    unfair_text_label = tk.Label(master=unfair_frame, font=('', 20,), text="GET READY", foreground="WHITE", background="RED")
    unfair_text_label.grid(row=1, sticky=W+E)

    show_vid()


def show_vid():
    global stop
    global last_frame

    if stop:
        # Save last picture for more data later i guess
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (1080, 720))
        img = Image.fromarray(pic)
        img.save('img.jpg') # TODO: randomize name
        return

    if not cap.isOpened():
        print("cant open the camera")
        return

    flag, frame = cap.read()
    if flag is None:
        print("cant open the camera")
        return
    elif flag:
        last_frame = frame.copy()
        height, width, channels = last_frame.shape

    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic, (1080, 720))
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)

    if mode == 0:
        norm_main_label.imgtk = imgtk
        norm_main_label.configure(image=imgtk)
        norm_main_label.after(5, show_vid) # Callback loop to redisplay img every 10ms
    else:
        unfair_main_label.imgtk = imgtk
        unfair_main_label.configure(image=imgtk)
        unfair_main_label.after(5, show_vid)  # Callback loop to redisplay img every 10ms

if __name__ == '__main__':
    main_window = tk.Tk()

    # Creates the 3 main screens, stacked on top of each other
    main_frame = Frame(master=main_window)
    normal_frame = Frame(master=main_window)
    unfair_frame = Frame(master=main_window)

    main_frame.grid(row=0, column=0, sticky='NEWS')
    normal_frame.grid(row=0, column=0, sticky='NEWS')
    unfair_frame.grid(row=0, column=0, sticky='NEWS')

    """ Main screen """
    # Displays the main img on the main screen
    main_img = Image.open("rps.jpg")
    main_img = ImageTk.PhotoImage(main_img)
    main_label = tk.Label(main_frame, image=main_img, width=1080, height=760)
    main_label.image = main_img # So not garbage collected
    main_label.grid(row=0, rowspan=2, sticky=W+E)

    # Displays the two game mode buttons on the main screen
    normal_button = tk.Button(main_frame, font=('', 20,), pady=2, text="Normal", command=start_normal_game)
    normal_button.grid(row=2, sticky=W+E)
    unfair_button = tk.Button(main_frame, font=('', 20,), pady=2, text="Unfair", command=start_unfair_game)
    unfair_button.grid(row=3, sticky=W+E)

    """ Normal screen """
    # Displays the game mode buttons on the normal screen
    start_normal = tk.Button(normal_frame, font=('', 15,), height=1, width=22, text="Start", command=take_photo)
    start_normal.grid(row=2, sticky=W+E)
    reset_normal = tk.Button(normal_frame, font=('', 15,), height=1, width=22, text="Reset", command=start_normal_game)
    reset_normal.grid(row=3, sticky=W+E)
    back_normal = tk.Button(normal_frame, font=('', 15,), height=1, width=22, text="Back", command=back)
    back_normal.grid(row=4, sticky=W+E)

    """ Unfair screen """
    # Displays the two game mode buttons on the unfair screen
    start_unfair = tk.Button(unfair_frame, font=('', 15,), height=1, width=22, text="Start", command=take_photo)
    start_unfair.grid(row=2, sticky=W+E)
    reset_unfair = tk.Button(unfair_frame, font=('', 15,), height=1, width=22, text="Reset", command=start_unfair_game)
    reset_unfair.grid(row=3, sticky=W+E)
    back_unfair = tk.Button(unfair_frame, font=('', 15,), height=1, width=22, text="Back", command=back)
    back_unfair.grid(row=4, sticky=W+E)

    main_window.title("RPS")

    # Display the first screen
    main_frame.tkraise()

    main_window.mainloop()
    cap.release()
