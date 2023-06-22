from ultralytics import YOLO
import cv2
import math
import cvzone

# I used this to run inference to see if it was recognizing me
# It checks to see if it can recognize me as a human from my picture
# model = YOLO('yolov8n.pt')
# result = model('Sam.jpg',show=True)
# cv2.waitKey(0)


# Triggers real time feed from my webcam
# And also accesses the trained data model in YOLO
cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')


# Reading the classes
classnames = []
with open('classes.txt','r') as f:
   classnames = f.read().splitlines()


while 1:
   _,frame = cap.read()
   result = model(frame, stream= True)
   # Getting the box outline ,confidence and class names informations to work
   for info in result:
       boxes = info.boxes
       for box in boxes:
           x1,y1,x2,y2 = box.xyxy[0]
           conf = box.conf[0]
           names = box.cls[0]

           conf = math.ceil(conf * 100)
           if conf > 65:
               x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
               names = int(names)


               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
               cvzone.putTextRect(frame,f'{classnames[names]} {conf}%',[x1+8,y1-12],
                              scale=1.5,thickness=2)





# Show frame around selected object being identified
   cv2.imshow('frame',frame)
   cv2.waitKey(1)

