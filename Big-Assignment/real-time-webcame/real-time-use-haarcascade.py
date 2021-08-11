import cv2
import sys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import time
import imutils

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
cascPath = "D:/HK4/CS114.L21-Machine-Learning/final-project/video_tan/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

print("[INFO] loading face mask detector model...")
path = "D:/HK4/CS114.L21-Machine-Learning/final-project/train_cnn_Tan_model/Adam/mask_detector_imple-cnn_Adam.model"
model = load_model(path)

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        #face_frame =  preprocess_input(face_frame)
        faces_list.append(face_frame)

        if len(faces_list) > 0:
            #faces_list = np.array(faces_list, dtype="float32")
            preds = model.predict(faces_list)
        for pred in preds:
            (correct_mask, incorrect_mask, without_mask) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            max_prob = max(incorrect_mask, correct_mask, without_mask)
            
        label = "correct_mask"
        color = (0, 255, 0)
        if incorrect_mask >= max_prob:
           label = "incorrect mask"
           color = (255, 0, 0)
        elif without_mask >= max_prob:
           label = "without mask"
           color = (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max_prob * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
        
        
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()