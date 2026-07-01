
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python?_gl=1*pkpspi*_up*MQ..*_ga*MTMzMzAxNTgzMS4xNzgwODY0MDE5*_ga_SM8HXJ53K2*czE3ODA4NjQwMTgkbzEkZzAkdDE3ODA4NjQwMTgkajYwJGwwJGgw
import cv2
import mediapipe as mp
import time
import csv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_results = None
image = None
lastest_timestamp = 0


hand_connections = [                    # Basically tupled indicies of connections between joints.
    (0,1),(1,2), (2,3), (3,4),          # Wrist to thumb
    (0,5), (5,6), (6,7), (7,8),         # wrist to forefinger
    (0,9), (9,10), (10,11), (11,12),    # wrist to middle finger
    (0,13), (13,14), (14,15), (15,16),  # wrist to ring finger
    (0,17), (17,18), (18,19), (19,20),  # wrist to pinky
    (5,9), (9,13), (13,17), (17,0)      # palm
]

def training_data_writer(letter, landmark):
    print("Full landmarks")
    row_list = []
    row_list.append(chr(letter))
    for i in landmark:
        print(f"i: {i}")
        row_list.append(i.x)
        row_list.append(i.y)
        row_list.append(i.z)
    writer.writerow(row_list)

def my_result_callback(result, output_image, timestamp_ms):
    #print("Calling back")
    global latest_results
    global image
    global lastest_timestamp

    latest_results = result
    image = output_image
    lastest_timestamp = timestamp_ms

def draw_landmarks(hand, frame):

    for first,second in hand_connections:
        x0 = int(hand[first].x*frame.shape[1])  # Turn normalised (0-1) back into actual size of the frame.
        y0 = int(hand[first].y*frame.shape[0])  # frame.shape[0] = height, frame.shape[1] = width
        x1 = int(hand[second].x*frame.shape[1])
        y1 = int(hand[second].y*frame.shape[0])
        cv2.line(frame,(x0,y0),(x1,y1),(0,255,0),2)

    for landmark in hand:
        x_pixel = int(landmark.x * frame.shape[1])  
        y_pixel = int(landmark.y *frame.shape[0])
                                                # Z (depth) is relative to the wrist
        cv2.circle(frame, (x_pixel,y_pixel), 5, (255,255,0),-1)


# Create a hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/Mine/Coding_Projects/ASL_vision_NeuralNet/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback = my_result_callback)
with HandLandmarker.create_from_options(options) as landmarker:         

    cap = cv2.VideoCapture(0)                   # opens webcam
    with open('training_landmarks.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        while True:      
            letter = cv2.waitKey(1) & 0xFF
            ret, frame = cap.read()                 # ret = if frame is there, frame is the frame itself
            if ret == True:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        # Blu,Gre,Red to RGB converts to RGB as its needed for media pipe
            else:
                print("broken")
                break
            last_timestamp = lastest_timestamp
            lastest_timestamp = int(time.time()*1000)
            if lastest_timestamp > last_timestamp:      # checks if time is always moving forward
                # print("time is moving forward")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb)
                landmarker.detect_async(mp_image,lastest_timestamp)   # mp.image as MP doesn't accept numpy, calculates finger vectors
                                                                                                                    # async waits for resposnse from callback, without pausing  
            
            if latest_results and latest_results.hand_landmarks:        # Prevents async timing issues & if no hands are in frame(needs frame to start everything.)
                data = latest_results.hand_landmarks[0][0]
                #print(f"data: {data}")
                for hand in latest_results.hand_landmarks:
                    draw_landmarks(hand,frame)

            while letter != 255:       # if letter is pressed
                if latest_results.hand_landmarks:
                    for hand_frame in latest_results.hand_landmarks:
                        #after press, hold to capture that letter.
                        training_data_writer(letter, latest_results.hand_landmarks[0])
                break
            cv2.imshow("Hand Tracking", frame)

    cap.release()                   # releases webcame
    cv2.destroyAllWindows()         # closes windows