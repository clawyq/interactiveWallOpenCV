import cv2
import threading
import serial

mutex = threading.Semaphore()
XSTART = 100
XEND = 400
YSTART = 250
YEND = 750
LOWER_BRIGHTNESS_THRESHOLD = 60
KERNEL_BLUR = 13
STANDARD_DEVIATION_X = 0
GREEN_BOX_RGB = (0, 255, 0)
RESOLUTION_SCALE = 1
MIN_DISTANCE = 50
EYE_CASCADE_CLASSIFIER = 'haarcascade_righteye_2splits.xml'
BAUD_RATE = 9600
SERIAL_PORT = "/dev/ttyACM0"
# value to be passed into Canny() higher values means less edges will be detected
EDGE_STRICTNESS = 52
# higher values means an object must have more edges to shape it like a circle to be considered a circle
CIRCLE_STRICTNESS = 16
LEFT = 0
CENTRE_OFFSET = 1
RIGHT = 2

global frameSet
global frameCounter
frameSet = set()
frameCounter = 0

eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_CLASSIFIER)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)


def messagePasser(sector):
    global frameSet
    global frameCounter
    mutex.acquire()
    frameCounter += 1
    if frameCounter % 10 == 0:
        # send information to arduino
        # redefine frame set
        for i in frameSet:
            ser.write(bytes(str(i).encode('utf-8')))
            print('sending {}'.format(i), end=" ")
        ser.write(b'9')
        frameSet = set()
    else:
        frameSet.add(sector)
        for i in frameSet:
            print(i, end=" ")
    mutex.release()


class camThread(threading.Thread):
    def __init__(self, cap, ID):
        threading.Thread.__init__(self)
        self.cap = cap
        self.ID = ID

    def run(self):
        driver(self.cap, self.ID)


def driver(cap, ID):
    global frameCounter
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        for i in range(1):

            roi = frame[XSTART: XEND, YSTART: YEND]
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurg_roi = cv2.GaussianBlur(gray_roi, (KERNEL_BLUR, KERNEL_BLUR), STANDARD_DEVIATION_X)

            _, bthreshold = cv2.threshold(blurg_roi, LOWER_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

            # RETR_TREE calculates full hierarchy of the contours for nested object detection
            _, contours, _ = cv2.findContours(bthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            eyes = eye_cascade.detectMultiScale(blurg_roi)

            for eye in eyes:
                (x, y, w, h) = eye
                lrc = (x * 3) // (XEND - XSTART)
                if ID == 0:
                    # From left, 2/3 of ID0's frame needs to be LEFT, 1/3 is CENTER
                    lrc //= 2
                else:
                    # From left, 1/3 OF ID1's frame needs to be center, 2/3 is RIGHT
                    if lrc == LEFT:
                        lrc += CENTRE_OFFSET
                    else:
                        lrc = RIGHT

                messagePasser(lrc)
                print('eye x:{} w:{} cameraID: {}'.format(x, w, str(ID)))
                cv2.rectangle(roi, (x, y), (x + w, y + h), GREEN_BOX_RGB, 2)
                roi_gray2 = blurg_roi[y: y + h, x: x + w]
                roi_color2 = roi[y: y + h, x: x + w]
                circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, RESOLUTION_SCALE, MIN_DISTANCE,
                                           param1=EDGE_STRICTNESS, param2=CIRCLE_STRICTNESS, minRadius=0, maxRadius=0)

            # cv2.imshow("roi_{}".format(cap), roi)
            # cv2.imshow("bthresh_{}".format(cap), bthreshold)
            key = cv2.waitKey(100)
            if key == 27:
                cap.release()
                break


# run
thread0 = camThread(cap0, 0)
thread1 = camThread(cap1, 1)
thread0.start()
thread1.start()

cv2.destroyAllWindows()
