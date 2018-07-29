import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

change_res(640,480)


while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (y,x,w,h) in faces:
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        cv2.rectangle(frame, (y,x), (y + h, x + w), (0,0,255), 2)
        eye = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for(ey,ex,ew,eh) in eye:
            cv2.rectangle(roi_color, (ey, ex), (ey+eh, ex+ew),(255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detect',frame)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
