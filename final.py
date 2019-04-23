from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(0.1)

thresh = 0.25
frame_check = 2
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#cap=cv2.VideoCapture(0)
flag=0
blink = False
blinkList = []

counter = 0
inc = 0
total = 0
calibrate = True #calibrate, avgLEAR, and avgREAR all need to be initialized at once. Keep them together if we break this out.
avgLEAR  = 0
avgREAR = 0
avgEAR = 0
blinks = 0
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    #ret, frame=cap.read()
    frame = frame.array
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR)/2
        diff = ear - avgEAR
        thresh = avgEAR*(9/10)
        
        if calibrate == True:
            cv2.putText(frame, ("Calibrating  %.3f" %(leftEAR)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            #cv2.putText(frame, ("left eye: %.3f" %(leftEAR)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, ("right eye: %.3f" %(rightEAR)), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, ("Current EAR: %.3f" %(ear)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, ("Calibrated EAR: %.3f" %(avgEAR)), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, ("Diff: %.3f" %(abs(avgEAR - ear))), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, ("flag: %d" %(flag)), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, ("Blinks: %d" %(total)), (10, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if calibrate == True:
            inc += 1
            avgLEAR += leftEAR
            avgREAR += rightEAR
                #print("EAR: " + str(ear))
            print("EAR: " + str(ear))
            if inc == 15:
                avgLEAR = avgLEAR/inc
                avgREAR = avgREAR/inc
                avgEAR = (avgLEAR + avgREAR)/2
                calibrate = False
                
            
        if ((leftEAR < thresh) & (rightEAR < thresh)):
        #if diff >= 0.1:
            flag += 1
            blink = True
            counter += 1
            #blinkList.append()
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
        else:
            flag = 0
            if counter >= 1:
                total += 1
            counter = 0
    #print ("leftEAR: %.3f rightEAR: %.3f flag: %d \n" %(leftEAR, rightEAR, flag))
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()