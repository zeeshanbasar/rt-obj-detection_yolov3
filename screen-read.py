import numpy as np
from PIL import ImageGrab
import cv2

while(True):
    screen=np.array(ImageGrab.grab(bbox=(10,50,800,640)))    # cv2.imshow('window',cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB),cv2.resize(image_np, (800,600)))
    cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))

    y1, x1, y2, x2 = 100,500,200,700
    y3, x3, y4, x4 = 200,700,300,900

    color=(0,0,0)
    thickness=2
    black='black'
    box='box'
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.rectangle(screen,(x1,y1),(x2,y2),color, thickness)
    cv2.putText(screen,(black+"A"+box),(x1,y1-10), font, 0.5,(255,255,255),0)
    cv2.rectangle(screen,(x3,y3),(x4,y4),(255,0,0), thickness)
    # cv2.imshow('window',image)
    # cv2.imshow('window',image2)
    cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
