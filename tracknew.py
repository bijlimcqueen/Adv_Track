import cv2
import numpy as np
from cv2 import VideoCapture


def distancepoints(point1, point2) : 
    difference = 0
    difference = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return difference


def selectcolor():
    color = input('\nPlease select the colour of the object you want to track:\n 1. Black\n 2. Blue\n 3. Red\n 4. Green\n ===>Input: ')
    color = int(color)

    if color == 1 : #BLACK
        Lower = (0,0,0)
        Upper = (80, 255, 30)

    elif color == 2 : #BLUE
        Lower = (80, 50, 70)
        Upper = (120, 255, 252)

    elif color == 3 : #RED
        Lower = (0,100,50) #bruh
        Upper = (10,255,255)

    elif color == 4 : #GREEN
        print('Green')
        Lower = (36, 50, 70)
        Upper = (89, 255, 255)

    else : 
        print('please enter the number corresponding to the color you wish to select')

    return Lower, Upper


print('============== *** Obtaining Input from video Feed *** =====================')
count = 0
ptstracked = []
distance_threshold = 500  # Distance threshold for motion consistency
velocity_threshold = 500  # Velocity threshold for motion consistency
Lower, Upper = selectcolor()
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
best_center = None
best_distance = float('inf')

# define a video capture object
vid = cv2.VideoCapture(0)
while(True):
    #Begin Webcam video
    ok, frame = vid.read()
    frame2draw = frame.copy()
    if not ok:
        print('Error - Video not Obtained')
        break

    #convert frame to HSV, obtain color threshold, apply morphological operations and threshold
    framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    mask = cv2.inRange(framehsv, Lower, Upper)
    _,thresh = cv2.threshold(mask , 40, 255, 0)
    maskop1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (9,9))
    maskop = cv2.morphologyEx(maskop1, cv2.MORPH_CLOSE, (9,9))
    
    # obtaining contours from binary mask
    contours, hierarchy = cv2.findContours(maskop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contour_data = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 3000: 
            continue
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        contour_data.append((area, (x, y), radius))
        
    #sort contours based on area in descending order and slice for top 10 areas
    contour_data.sort(key=lambda x: x[0], reverse=True)
    top_contours = contour_data[:10]
    
    centers = []
    radiuslist = []
    for data in top_contours:
        (area, (x, y), radius) = data
        centers.append((x, y))
        radiuslist.append(int(radius))

    try:    
        if count > 1: 
            # Calculate optical flow for each center
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_centers, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, np.array(prev_centers, dtype=np.float32), None, **lk_params)

            # Calculate speed and distance traveled for each center
            for i in range(len(centers)):
                prev_center = prev_centers[i]
                next_center = next_centers[i]
                speed = np.sqrt((next_center[0] - prev_center[0]) ** 2 + (next_center[1] - prev_center[1]) ** 2)
                distance = np.sqrt((next_center[0] - centers[i][0]) ** 2 + (next_center[1] - centers[i][1]) ** 2)
                print(f"Center {i+1}: Speed - {speed}, Distance - {distance}")

                # Check motion consistency using distance and velocity thresholds
                if distance < distance_threshold and speed < velocity_threshold:
                    # ptstracked.append(next_center)
                    # cv2.circle(frame2draw, (int(next_center[0]), int(next_center[1])), radiuslist[i], (0, 0, 255), 2)
                    best_center = next_center
                    best_distance = distance
                    best_radius = i
            
            if best_center is not None:
                ptstracked.append(best_center) 
            cv2.circle(frame2draw, (int(best_center[0]), int(best_center[1])), radiuslist[i], (0, 0, 255), 2)
        
    except: print('No suitable contour')
    
    
    print("=================HERE========================")
    ptstracked = ptstracked[-20:]
    ptstrackedNP = np.array(ptstracked, dtype=np.int32)
    cv2.polylines(frame2draw, [ptstrackedNP], False, (0, 0, 255), 3)
    cv2.imshow('mask', maskop)
    cv2.imshow('final tracked frame', frame2draw)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    prev_frame = frame.copy()
    prev_centers = centers.copy() 
    count+=1

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()