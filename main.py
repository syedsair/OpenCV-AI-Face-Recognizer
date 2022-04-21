import tkinter as tk
from tkinter import *
import cv2.cv2 as opencv
import numpy as np
import os, time
from PIL import Image


# aws_secret="AKIAIMNOJVGFDXXXE4OA"


def userAuthenticated():
    file = open("Registrations", 'r')
    strr = ""
    lines = file.readlines()
    for line in lines:
        strr += line

    data.configure(text=strr)


def isInteger(s):
    for i in range(len(s)):
        if 48 <= ord(s[i]) <= 57:
            pass
        else:
            return False
    return True


def idExists(id):
    file = open("Registrations", 'r')
    lines = file.readlines()
    for line in lines:
        line = line[0:len(line)-1]
        if line == id:
            return True
    return False


def addUserFunction():
    id = nameEntry.get()
    if idExists(id):
        response.configure(text="ID already Exists!")
        return
    if isInteger(id):
        cameraObject = opencv.VideoCapture(0)
        detectorPath = "haarcascade_frontalface_default.xml"
        face_detector = opencv.CascadeClassifier(detectorPath)
        collectedImages = 0
        while True:
            returned, returned_image = cameraObject.read()
            grayScaleImage = opencv.cvtColor(returned_image, opencv.COLOR_BGR2GRAY)
            captured_faces = face_detector.detectMultiScale(grayScaleImage, 1.3, 5)
            for x, y, w, h in captured_faces:
                opencv.rectangle(returned_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                collectedImages = collectedImages + 1
                opencv.imwrite("Database\ " + id + "_" + str(collectedImages) + ".jpg", grayScaleImage[y:y + h, x:x + w])
                opencv.imshow('Frame', returned_image)
            if opencv.waitKey(50) & 0xFF == ord('q'):
                break
            if collectedImages > 50:
                break
        cameraObject.release()
        opencv.destroyAllWindows()
        response.configure(text="Images Saved for " + id)
        file = open("Registrations", 'a')
        file.write(id + '\n')
        file.close()
    else:
        response.configure(text="ID is not correct!")


def loadImages(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        id = str(imagePath.split('_')[0])
        id = id.split('\\')[1]
        id = int(id[1:len(id)])
        faces.append(imageNp)
        ids.append(id)
    return faces, ids


def trainFunction():
    faces, names = loadImages("Database")
    if len(faces) == 0:
        os.remove('TrainedModel.yml')
        response.configure(text="Empty Database. Model Reset!")
    else:
        recognizer = opencv.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(names))
        recognizer.save("TrainedModel.yml")
        response.configure(text="Model Trained")


def unlockFunction():
    found = False
    recognizer = opencv.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainedModel.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = opencv.CascadeClassifier(harcascadePath)

    cam = opencv.VideoCapture(0)
    font = opencv.FONT_HERSHEY_SIMPLEX
    message = "Unknown"
    while True:
        ret, im = cam.read()
        gray = opencv.cvtColor(im, opencv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            opencv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                message = Id
                found = True
                break
            else:
                Id = 'Unknown'

            opencv.putText(im, str(Id), (x, y + h), font, 1, (255, 255, 255), 2)

        opencv.imshow('im', im)
        opencv.waitKey(50)
        if found:
            break

    cam.release()
    opencv.destroyAllWindows()
    response.configure(text="Person: " + str(message))
    userAuthenticated()


def clearNameFunction():
    nameEntry.delete(0, 'end')


GUI = tk.Tk()
GUI.title("Face Unlock System")
GUI.geometry("1000x600")

heading = tk.Label(GUI, text="File Security through Face Recognition", fg="black", width=31, height=2,font=('times', 40, 'bold'))
heading.place(x=0, y=0)

response = tk.Label(GUI, text="", fg="black", width=31, height=2, font=('times', 20))

response.place(x=250, y=100)

addUserButton = tk.Button(GUI, text="Add User", command=addUserFunction, fg="white", bg="black", width=10, height=1,font=('times', 20, ' bold '))
addUserButton.place(x=800, y=200)

trainButton = tk.Button(GUI, text="Train Model", command=trainFunction, fg="white", bg="black", width=10, height=1,font=('times', 20, ' bold '))
trainButton.place(x=800, y=300)

unlockButton = tk.Button(GUI, text="Unlock File", command=unlockFunction, fg="white", bg="black", width=10, height=1,font=('times', 20, ' bold '))
unlockButton.place(x=800, y=400)

nameLabel = tk.Label(GUI, text="Name:", fg="black", width=10, height=1, font=('times', 20, 'bold'))
nameLabel.place(x=35, y=250)

nameEntry = tk.Entry(GUI, width=20, fg="white", bg="black", font=('times', 20, ' bold '))
nameEntry.place(x=75, y=300)

nameClearButton = tk.Button(GUI, text="Clear", command=clearNameFunction, fg="white", bg="black", width=5, height=1,font=('times', 10, ' bold '))
nameClearButton.place(x=75, y=350)

data = tk.Label(GUI, text="", fg="black", width=10, height=10,font=('times', 20, 'bold'))
data.place(x=400, y=300)
GUI.mainloop()
