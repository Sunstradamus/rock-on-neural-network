"""
Super janky ass rock paper scissors game
"""

#importing modules required
import Tkinter as tk
from Tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import threading

stop = False

# Latest frame is the most up to date image from webcam to display
latest_frame = np.zeros((640, 480, 3), dtype=np.uint8)
# Last frame is the last frame after game played (the user's hand)
last_frame = np.zeros((640, 480, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

# # Sets image size to 1280x720 (closest to 1080x720)
cap.set(3, 640)
cap.set(4, 480)


def back():
    global stop
    main_frame.tkraise()
    stop = True


# First function call after start game button pressed
def take_photo():

    # Disables all the buttons once the game has started
    start.configure(state=DISABLED)
    reset.configure(state=DISABLED)
    back.configure(state=DISABLED)

    # Starts countdown to display rock text on game screen
    # GOTTA START A NEW THREAD TO MAKE ASYNC OR ELSE SLEEPING WILL BLOCK THE MAIN THREAD SHOWING WEBCAM FEED AND IT DIES
    t = threading.Timer(1, display_rock)
    t.start()


# Second function call after start game button pressed
def display_rock():
    # Sets the text under the img to say rock
    game_text_label.configure(text="ROCK")

    # Starts countdown to display paper text on game screen
    time.sleep(1)
    display_paper()


# Third function call after start game button pressed
def display_paper():
    # Sets the text under the img to say paper
    game_text_label.configure(text="PAPER")

    # Starts countdown to display scissors text on game screen
    # and start the classification
    time.sleep(1.3)
    stop_video()


def display_scissors():
    # Sets the text under the img to say scissors
    game_text_label.configure(text="SCISSORS")


def stop_video():
    global stop
    global last_frame
    global game_main_img

    # If mode is normal, take last frame, stop video, display scissors text, then classify last_frame, and display result
    if mode == 0:
        # Copies the latest frame as the one to classify
        last_frame = latest_frame.copy()
        stop = True

        # Waits until video has stopped so we can display computer's choice (else it overrides it)
        while not stopped:
            time.sleep(0.05)
        # Displays the scissors text
        display_scissors()

        # TODO: RANDOMIZE A CHOICE OF ROCK, PAPER OR SCISSORS THEN CLASSIFY LAST_FRAME AND DISPLAY RESULTS
        # make choice

        # classify last frame
        classify(last_frame)

        # display choice
        emg = Image.open("rock.jpg")
        emg = ImageTk.PhotoImage(emg)
        # The displayed image on the main screen
        game_main_img.configure(image=emg)
        game_main_img.image = emg

    # If mode is unfair, take last frame (user will have hand rdy since each R, P, S comes after
    # 1s each, but we will look at hand first
    else:
        # Copies the latest frame as the one to classify
        last_frame = latest_frame.copy()

        # TODO: CLASSIFY LAST FRAME, THEN MAKE OPPOSITE CHOICE OF CLASS AND DISPLAY RESULTS
        # classify this last frame
        classify(last_frame)

        # test to see if image looks fine when we take photo before hand
        pic = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (640, 480))
        img = Image.fromarray(pic)
        img.save('cheat.jpg')  # TODO: randomize name

        time.sleep(1) # test with classification taking 1s
        # make choice
        # ?

        stop = True
        # Waits until video has stopped so we can display computer's choice (else it overrides it)
        while not stopped:
            time.sleep(0.05)
        # Displays the scissors text
        display_scissors()

        # test to see last image
        pic = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (640, 480))
        img = Image.fromarray(pic)
        img.save('normal.jpg')  # TODO: randomize name

        # display choice
        emg = Image.open("rock.jpg")
        emg = ImageTk.PhotoImage(emg)
        # The displayed image on the main screen
        game_main_img.configure(image=emg)
        game_main_img.image = emg


    # Reenables the reset and back the buttons once the game has finished
    reset.configure(state=NORMAL)
    back.configure(state=NORMAL)


# Starts the normal game
def start_game():
    global stop
    global stopped
    global game_text_label

    # Set video flag to false
    stop = False
    stopped = False

    game_text_label.configure(text="GET READY")

    # Enables the start and back buttons and disables the reset button
    start.configure(state=NORMAL)
    back.configure(state=NORMAL)
    reset.configure(state=DISABLED)

    game_frame.tkraise()


# Starts the normal game
def set_normal_game():
    global mode

    # Set game mode to normal
    mode = 0
    start_game()


# Starts the unfair game
def set_unfair_game():
    global mode

    # Set game mode to unfair
    mode = 1
    start_game()


def classify(frame):
    # TODO: Classify frame and return results
    return


def show_vid():
    global latest_frame
    global stopped

    if not cap.isOpened():
        print("cant open the camera")
        return

    flag, frame = cap.read()
    if flag is None:
        print("cant open the camera")
        return
    elif flag:
        latest_frame = frame.copy()

    pic = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic, (640, 480))
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)

    if not stop:
        game_main_img.imgtk = imgtk
        game_main_img.configure(image=imgtk)
    else:
        stopped = True

    game_main_img.after(5, show_vid) # Callback loop to redisplay img every 5ms


if __name__ == '__main__':
    main_window = tk.Tk()

    """ Sets up the initial GUI """
    # Creates the 3 main screens, stacked on top of each other
    main_frame = Frame(master=main_window)
    game_frame = Frame(master=main_window)

    # Frames to hold the screen's GUI elements
    main_frame.grid(row=0, column=0, sticky='NEWS')
    game_frame.grid(row=0, column=0, sticky='NEWS')

    """ Main screen """
    # Displays the main img on the main screen
    main_img = Image.open("rps.jpg")
    main_img = ImageTk.PhotoImage(main_img)
    # The displayed image on the main screen
    main_label = tk.Label(main_frame, image=main_img, width=640, height=480)
    main_label.image = main_img  # So not garbage collected
    main_label.grid(row=0, rowspan=2, sticky=W+E)

    # The two game mode buttons on the main screen
    normal_button = tk.Button(main_frame, font=('', 15,), text="Normal", command=set_normal_game)
    normal_button.grid(row=2, sticky=W+E)
    unfair_button = tk.Button(main_frame, font=('', 15,), text="Unfair", command=set_unfair_game)
    unfair_button.grid(row=3, sticky=W+E)

    """ Game screen """
    # The main image and text on the game screen
    game_main_img = tk.Label(master=game_frame, width=640, height=480)
    game_main_img.grid(row=0, columnspan=3, sticky=W+E)
    game_text_label = tk.Label(master=game_frame, font=('', 20,), text="GET READY", foreground="WHITE", background="RED")
    game_text_label.grid(row=1, columnspan=3, sticky=W+E)

    # The start, reset and back buttons on the game screen
    start = tk.Button(game_frame, font=('', 15,), height=1, width=15, text="Start", command=take_photo)
    start.grid(row=2, column=0, sticky=W+E)
    reset = tk.Button(game_frame, font=('', 15,), height=1, width=15, text="Reset", command=start_game)
    reset.grid(row=2, column=1, sticky=W+E)
    back = tk.Button(game_frame, font=('', 15,), height=1, width=15, text="Back", command=back)
    back.grid(row=2, column=2,  sticky=W+E)

    main_window.title("RPS")

    # Starts the video
    show_vid()

    # Display the first screen
    main_frame.tkraise()

    main_window.mainloop()
    cap.release()
