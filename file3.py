import cv2
# import datetime

from datetime import datetime

full_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
ret, Frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# name as date + hour

current_date = datetime.today().strftime('%Y-%m-%d')
filename = datetime.today().strftime('%Y-%m-%d') + ".avi"
print(filename)
out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

while 1:
    processing_date = datetime.today().strftime('%Y-%m-%d')
    print("Processing date "+processing_date)

    if processing_date != current_date:
        out.release()
        current_date = datetime.today().strftime('%Y-%m-%d')
        filename = datetime.today().strftime('%Y-%m-%d')+".avi"
        # start new Video writer with new date
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    ret, Frame = cap.read()
    gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    human = full_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in human:
        cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 0, 220), 3)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(Frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = Frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        # out.write(Frame)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        print("Detected Humans")
        out.write(Frame)

    cv2.imshow('video', Frame)
    # out.write(Frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()