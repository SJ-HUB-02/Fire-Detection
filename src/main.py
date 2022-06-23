import cv2
import numpy as np
import playsound
import threading
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

Alarm_Status = False
Fire_Reported = 0


def play_alarm_sound_function():
    while True:
        playsound.playsound('G:/Internship/Feynn Labs/Solo Projecct/Code/alarm-sound.mp3', True)


model = tf.keras.models.load_model('G:/Internship/Feynn Labs/Solo Projecct/Code/model.h5')
video = cv2.VideoCapture("G:/Internship/Feynn Labs/Solo Projecct/Code/2.mp4")
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (224, 224))
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)
    if prediction == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(probabilities[prediction])
        if Alarm_Status == False:
            threading.Thread(target=play_alarm_sound_function).start()
            Alarm_Status=True
            break
    else:
        print("No fire")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
