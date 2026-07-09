
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python?_gl=1*pkpspi*_up*MQ..*_ga*MTMzMzAxNTgzMS4xNzgwODY0MDE5*_ga_SM8HXJ53K2*czE3ODA4NjQwMTgkbzEkZzAkdDE3ODA4NjQwMTgkajYwJGwwJGgw
import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import requests
import threading     # to make custom callback function

from universal import logger

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


latest_results = None
image = None
lastest_timestamp = 0
url = "http://localhost:5000/"

hand_connections = [                    # Basically tupled indicies of connections between joints.
    (0,1),(1,2), (2,3), (3,4),          # Wrist to thumb
    (0,5), (5,6), (6,7), (7,8),         # wrist to forefinger
    (0,9), (9,10), (10,11), (11,12),    # wrist to middle finger
    (0,13), (13,14), (14,15), (15,16),  # wrist to ring finger
    (0,17), (17,18), (18,19), (19,20),  # wrist to pinky
    (5,9), (9,13), (13,17), (17,0)      # palm
]

data = None

def background_send_data(landmark):
    global data
    hand_landmark = []
    for i in landmark:
        logger("sending data",(i), debug= False)
        hand_landmark.append(i.x)
        hand_landmark.append(i.y)
        hand_landmark.append(i.z)
    data = requests.post(url,json=hand_landmark).json()                      # Freezes code

def my_result_callback(result, output_image, timestamp_ms):
    global latest_results
    global image
    global lastest_timestamp

    latest_results = result
    image = output_image
    lastest_timestamp = timestamp_ms

def draw_landmarks(hand, canvas):
    for first,second in hand_connections:
        x0 = int(hand[first].x*canvas.shape[1])  # Turn normalised (0-1) back into actual size of the canvas.
        y0 = int(hand[first].y*canvas.shape[0])  # canvas.shape[0] = height, canvas.shape[1] = width
        x1 = int(hand[second].x*canvas.shape[1])
        y1 = int(hand[second].y*canvas.shape[0])
        cv2.line(canvas,(x0,y0),(x1,y1),(0,255,0),2)

    for landmark in hand:
        x_pixel = int(landmark.x * canvas.shape[1])  
        y_pixel = int(landmark.y *canvas.shape[0])
                                                # Z (depth) is relative to the wrist
        cv2.circle(canvas, (x_pixel,y_pixel), 5, (255,255,0),-1)

# Create a hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback = my_result_callback)
with HandLandmarker.create_from_options(options) as landmarker:        
    cap = cv2.VideoCapture(0)                   # opens webcam
    with open('training_landmarks.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        frame_count = 0
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:      
            letter = cv2.waitKey(1) & 0xFF
            ret, frame = cap.read()                 # ret = if frame is there, frame is the frame itself
            blank_frame = np.zeros(frame.shape,dtype=np.uint8)   #changes to float32
            if ret == True:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        # Blu,Gre,Red to RGB converts to RGB as its needed for media pipe
            else:
                print("No, frames available")
                break
            last_timestamp = lastest_timestamp
            lastest_timestamp = int(time.time()*1000)
            if lastest_timestamp > last_timestamp+10:      # checks if time is always moving forward
                # print("time is moving forward")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb)
                landmarker.detect_async(mp_image,lastest_timestamp)   # mp.image as MP doesn't accept numpy, calculates finger vectors
                frame_count += 1                                      # async waits for resposnse from callback, without pausing  
            
            if latest_results and latest_results.hand_landmarks:        # Prevents async timing issues & if no hands are in frame(needs frame to start everything.)
                #data = latest_results.hand_landmarks[0][0]
                for hand in latest_results.hand_landmarks:
                    draw_landmarks(hand,blank_frame)
                    
        
            while True:
                if latest_results and latest_results.hand_landmarks:
                    for hand_frame in latest_results.hand_landmarks:
                        #after press, hold to capture that letter.
                        if frame_count % 5 == 0:
                            t = threading.Thread(target = background_send_data, args=(latest_results.hand_landmarks[0],)) #commma is make it args a tuple
                            t.start()
                    if data != None:
                        logger("data",data['response'], debug=False)
                        cv2.putText(blank_frame, data['response'],(10,30), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255), thickness=5)
                break
            if letter == ord('q'):
                break
            cv2.imshow("Hand Tracking", blank_frame)

    cap.release()                   # releases webcame
    cv2.destroyAllWindows()         # closes windows