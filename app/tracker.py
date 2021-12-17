import cv2
import time
from matplotlib.figure import Figure
import argparse
import math
import json
from datetime import datetime


from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for,Response
)
from werkzeug.exceptions import abort

from app.auth import login_required
from app.db import get_db

bp = Blueprint('tracker', __name__)
stopCount = False
count=0

@bp.route('/')
def index():
    db = get_db()
    # posts = db.execute(
    #     'SELECT p.id, title, body, created, author_id, username'
    #     ' FROM post p JOIN user u ON p.author_id = u.id'
    #     ' ORDER BY created DESC'
    # ).fetchall()
    posts = {}
    return render_template('tracker/index.html', posts=posts)


@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    global stopCount
    if request.method == 'POST':
        stopCount = True

        if count != 0:
            db = get_db()
            db.execute(
                'INSERT INTO count (user_id,squat)'
                'VALUES (?,?)',
                (g.user['id'],count)
            )
            db.commit()
            print("Commited to DB successfully")
        return redirect(url_for('tracker.index'))

    return render_template('tracker/create.html')



@bp.route("/chart-data")
def chart_data():
    global stopCount
    if stopCount:
        stopCount =False
    return Response(gen_frames(), mimetype="text/event-stream")


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



def gen_frames():
    
    global count 
    count=0
    state = False

    name = "live_cam"
    timeTaken = time.time()
    parser = argparse.ArgumentParser(description='Run keypoint detection')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    
    args, unknown = parser.parse_known_args()



    

    
    protoFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]




    inWidth = 368
    inHeight = 368
    threshold = 0.1


    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    hasFrame, frame = cap.read()


    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if args.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    
    while cv2.waitKey(1) < 0 and not stopCount:
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
        

        x = points[8]
        y = points[9]
        z = points[10]
        xx = points[11]
        yy = points[12]
        zz = points[13]
        # print(points)
        left = find_angle(x,z,y)
        right = find_angle(xx,zz,yy)
        

        if left != None:
            if state and left >100:
                state= False
                count+=1
            elif not state and left < 100:
                # print('hello g')
                state = True
        
        
        if left != None : 
            json_data = json.dumps(
                {
                    "count": count,
                    "angle": left,
                }
            )
            yield f"data:{json_data}\n\n"
        cv2.imshow('Output-Skeleton', frame)
    cv2.destroyAllWindows()
