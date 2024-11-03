import cv2
import imutils
vs=cv2.VideoCapture(0)
firstframe=None
area=900
while True:
    a,img=vs.read()
    text='normal'
    resize=imutils.resize(img,width=1000,height=1000)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian=cv2.GaussianBlur(gray,(21,21),0)
    if firstframe is None:
        firstframe=gaussian
        continue
    imgdiff=cv2.absdiff(firstframe,gaussian)
    threshold=cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    threshold=cv2.dilate(threshold,None,iterations=2)
    contours=cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=imutils.grab_contours(contours)
    for i in contours:
        if cv2.contourArea(i)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(i)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text='MovingobjectDetected'
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow('show',img)
    key=cv2.waitKey(10)
    if key == ord('x'):
        break
vs.release()
cv2.destroyAllWindows()
