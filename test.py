import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# cap = cv2.VideoCapture("eye_recording.mp4")
cap = cv2.VideoCapture(0)
#cap.set(CV_CAP_PROP_FRAME_WIDTH, 352)
#cap.set(CV_CAP_PROP_FRAME_HEIGHT, 288)

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    XSTART = 100
    XEND = 400
    roi = frame[XSTART: XEND, 250: 750]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurg_roi = cv2.GaussianBlur(gray_roi, (13, 13), 0)

    _, bthreshold = cv2.threshold(blurg_roi, 60, 255, cv2.THRESH_BINARY_INV)
    _,contours, _ = cv2.findContours(bthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    eyes = eye_cascade.detectMultiScale(blurg_roi)

    for eye in eyes:
        window = set()
        for i in range(3000):
    #     (x, y, w, h) = cv2.boundingRect(cnt) 
            (x, y, w, h) = eye
            lrc = (x*3)//(XEND - XSTART)
            window.add(lrc)
            print('eye x:{} w:{}'.format(x, w))
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray2 = blurg_roi[y:y+h, x:x+w]
            roi_color2 = roi[y:y+h, x:x+w]
            circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, 1, 46, param1=52,
	    	param2=16, minRadius=0,
            maxRadius=0)
            print(lrc)
        for i in window:
            print("set: {}".format(i))
       # try:
       #     for i in circles[0,:]:
      #          print('{} {} {}'.format(i[0], i[1], i[2]))
     #           xPosition = i[0]
    #            segmentWidth = w/3
             #   lrc = xPosition//segmentWidth
        #        cv2.circle(roi_color2,(i[0], i[1]), i[2], (255,255,255) ,2)
         #       cv2.circle(roi_color2, (i[0], i[1]), 2, (255, 255, 255), 3)
         #   print("circle w: {}, h:{} s:{}".format(w, h, lrc) )
#
       # except Exception as e:
        #    print("hello")
         #   print(e)
    #    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #    cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (255, 0, 0), 1)
    #    cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 1)
    #    break

    cv2.imshow("roi", roi)
    cv2.imshow("btresh", bthreshold)
    key = cv2.waitKey(30)
    if key == 27:
        cap.release()
        break

cv2.destroyAllWindows()
