
import cv2
import time
import numpy as np
from matplotlib.figure import Figure
import argparse
import pandas as pd
import math
from flask import Flask, render_template, Response
from io import BytesIO
import base64
import json
import random
from datetime import datetime

def find_angle(p0,p1,c):
    if p0 == None or p1 == None or c == None: 
        return None
    p0c = math.sqrt(math.pow(c[0]-p0[0],2) + math.pow(c[1]-p0[1],2))
    p1c = math.sqrt(math.pow(c[0]-p1[0],2) + math.pow(c[1]-p1[1],2))
    p0p1 = math.sqrt(math.pow(p1[0]-p0[0],2) + math.pow(p1[1]-p0[1],2))
    try:
        return math.acos((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c))* (180 / math.pi)
    except:
        return None

app = Flask(__name__)
right_leg_angles = []
left_leg_angles = []

def gen_frames():
    
    count =0
    state = False

    name = "live_cam"
    timeTaken = time.time()
    parser = argparse.ArgumentParser(description='Run keypoint detection')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    parser.add_argument("--video_file", default= name + ".mp4", help="Input Video")

    args = parser.parse_args()



    

    
    protoFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]




    inWidth = 368
    inHeight = 368
    threshold = 0.1


    
    cap = cv2.VideoCapture(0)
    hasFrame, frame = cap.read()


    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if args.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :  
                points.append(None)

        i=0
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                
            i+=1
        
        
        cv2.putText(frame, "Squats Count = {:.2f} ".format(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        

        x = points[8]
        y = points[9]
        z = points[10]
        xx = points[11]
        yy = points[12]
        zz = points[13]
        # print(points)
        left = find_angle(x,z,y)
        right = find_angle(xx,zz,yy)
        left_leg_angles.append(left)
        right_leg_angles.append(right)

        if right != None:
            if state and right >100:
                state= False
                count+=1
            elif not state and right < 100:
                # print('hello g')
                state = True
        
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()d
        # yield (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        json_data = json.dumps(
            {
                "time": count,
                "value": left,
            }
        )
        yield f"data:{json_data}\n\n"
        cv2.imshow('Output-Skeleton', frame)
        print(type(frame))
        

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_random_data():
        while True:
            json_data = json.dumps(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "value": random.random() * 100,
                }
            )
            yield f"data:{json_data}\n\n"
            time.sleep(0.5)
        


@app.route("/chart-data")
def chart_data():
    return Response(gen_frames(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host = '0.0.0.0',debug=True)
# print( str(count) + " Squats" )