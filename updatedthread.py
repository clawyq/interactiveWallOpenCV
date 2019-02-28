 ret, frame = cap.read()
    if ret is False:
        break

    #window = set()

    for i in range(1):

        roi = frame[XSTART: XEND, YSTART: YEND]
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurg_roi = cv2.GaussianBlur(gray_roi, (KERNEL_BLUR, KERNEL_BLUR), STANDARD_DEVIATION_X)

        _, bthreshold = cv2.threshold(blurg_roi, LOWER_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        # RETR_TREE calculates full hierarchy of the contours for nested object detection
        _,  contours, _ = cv2.findContours(bthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        eyes = eye_cascade.detectMultiScale(blurg_roi)

        for eye in eyes:
            (x, y, w, h) = eye
            lrc = (x * 3)//(XEND - XSTART)
            #window.add(lrc)
            messagePasser(lrc)
            frameCounter += 1
            print('eye x:{} w:{}'.format(x, w))
            cv2.rectangle(roi, (x, y), (x + w, y + h), GREEN_BOX_RGB , 2)
            roi_gray2 = blurg_roi[y : y + h, x : x + w]
            roi_color2 = roi[y: y + h, x: x + w]
            circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, RESOLUTION_SCALE, MIN_DISTANCE,
                                       param1=EDGE_STRICTNESS, param2=CIRCLE_STRICTNESS, minRadius=0, maxRadius=0)

        cv2.imshow("roi", roi)
        cv2.imshow("btresh", bthreshold)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            break
    #for i in window:
        #print("window content: {}".format(i))

