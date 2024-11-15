import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
import os
import sqlite3
import pyttsx3
from threading import Thread
import pickle

# Prepare and load our sign language model
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('C:\\Users\\Olivier Flament\\540-signlanguage\\Code\\cnn_model_keras2.keras')

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab * 100 > 70:
        text = get_pred_text_from_db(pred_class)
    return text

cap = cv2.VideoCapture(0)  # Use local cam to capture the vid
hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300

def update_text_display(text):
    label_text.config(text=text)

# Canvas and vid capture
def tkImage():
    ref, frame = cap.read()
    if not ref:
        print("Failed to capture frame")
        return None
    return frame

# Pre-proceed captured img
def get_contour_thresh(img):
    frame = cv2.flip(img, 1)  # Flip the cam
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([cvimage], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return frame, contours, thresh

# Transfer pre-processed img to the type that tk can process
def convert_to_tk_image(thresh):
    pilImage = Image.fromarray(thresh)
    pilImage = pilImage.resize((image_width, image_height), Image.LANCZOS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage

def text_mode(cam):
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        frame = cam.read()[1]
        frame = cv2.resize(frame, (640, 480))
        frame, contours, thresh = get_contour_thresh(frame)
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0
            elif cv2.contourArea(contour) < 1000:
                if word != '':
                    update_text_display(word)
                text = ""
                word = ""
        else:
            if word != '':
                update_text_display(word)
            text = ""
            word = ""
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        res = np.hstack((frame, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('c'):
            break
    if keypress == ord('c'):
        return 2
    else:
        return 0

# Page design
top = tk.Tk()
top.title('Sign Language Detection')
top.geometry('900x600')
image_width = 600
image_height = 500
canvas = Canvas(top, bg='blue-1', width=image_width, height=image_height)
Label(top, text='Ready to Detect your sign language!:)', width=50, height=1).place(x=260, y=20, anchor='nw')
canvas.place(x=150, y=50)
label_text = tk.Label(top, text="", font=("黑体", 14))
label_text.pack()
image_container = None

def update_frame():
    global image_container
    frame = tkImage()
    if frame is not None:
        pic = convert_to_tk_image(frame)
        canvas.create_image(0, 0, anchor='nw', image=pic)
        image_container = pic  # Save the vid img
    else:
        print("No image to display")
    top.after(10, update_frame)  # Update the image per 10ms

top.after(10, update_frame)  # Start the timer
top.mainloop()

# Release the cam
cap.release()