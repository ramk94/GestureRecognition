'''
Ram Bhattarai
Credit to Sparsha Saha
ECE 544
This file loads the training model and predicts the gesture
Sends a update to firebase with the change
'''

import signal
import sys
from firebase import firebase
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils

# global variables
bg = None

#Creating an instance of the Firebase Class
firebase = firebase.FirebaseApplication('https://ece544-6703a.firebaseio.com',None)

#Resize the image before doing the prediction
#Trained in the same size so same size is needed for prediction
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

#Do the calibration of the environment so that if there is any noise
#Get it filtered
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#Give a segmented image with the background subtraction to avoid noisy background
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)


    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts,hiearchy = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#Handle what happens when the user presses control c on the keyboard
#Sends a signal to firebase that Gesture control is done
def sig_handler(signal,frame):
    firebase.put('GESTURE','GESTURE',0)
    print("Exiting\n")
    sys.exit(0)

#Main function where the prediction is done
def main():

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
  

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False
    
    gesture_firebase = firebase.get('GESTURE', 'GESTURE')
    while(gesture_firebase!=1):
        gesture_firebase = firebase.get('GESTURE', 'GESTURE')

    # keep looping, until interrupted
    while(True):

        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)


        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass(True)
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)
            else:
                    predictedClass, confidence = getPredictedClass(False)
                    showStatistics(predictedClass, confidence)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        
        signal.signal(signal.SIGINT, sig_handler)


        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        #if keypress == ord("s"):
        #    start_recording = True
        if gesture_firebase == 1:
            print("RECORDING\n")
            start_recording = True

#This function gives the prediction of the gesture
def getPredictedClass(flag):
    # Predict
    if(flag==True):
        image = cv2.imread('Temp.png')
    else:
        image = cv2.imread('temp2.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3]+prediction[0][4]))

#Based on what value is returned, updates the firebase accordingly
def showStatistics(predictedClass, confidence):


    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Fist"
        firebase.put('','MOTOR_SELECT',1); 
    elif predictedClass == 1:
        className = "Palm" 
        firebase.put('','MOTOR_SELECT',2);
    elif predictedClass == 2:
        className = "Swing"
        firebase.put('','MOTOR_SELECT',3);
    elif predictedClass == 3:
        className = "Peace"
        firebase.put('','MOTOR_SELECT',4);
    elif predictedClass == 4:
        className = "None"
        firebase.put('','MOTOR_SELECT',0);


    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)
    return className





# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,5,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("../TrainedModel/GestureRecogModel.tfl")

main()
