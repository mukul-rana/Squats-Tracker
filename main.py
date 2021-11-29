
import cv2
import time
import numpy as np
import argparse
import progressbar
import pandas as pd
import math

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



right_leg_angles = []
left_leg_angles = []
count =0
state = False

name = "live_cam"
timeTaken = time.time()
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--video_file", default= name + ".mp4", help="Input Video")

args = parser.parse_args()



MODE = "COCO"

if MODE is "COCO":
    protoFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "D:/CODES/Python/OPENCV/POse/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]



inWidth = 368
inHeight = 368
threshold = 0.1


input_source = args.video_file
cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# vid_writer = cv2.VideoWriter(name + '_gpu.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

# widgets = ["--[INFO]-- Analyzing Video: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
# pbar = progressbar.ProgressBar(maxval = n_frames,
#                                widgets=widgets).start()

p =0

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
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
            # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

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
    # p +=1
    # pbar.update()
    
    cv2.putText(frame, "Squats Count = {:.2f} ".format(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)

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
    
    cv2.imshow('Output-Skeleton', frame)
    print(type(frame))
    # print( str(count)  +  " ho gaye")
    # vid_writer.write(frame)

# vid_writer.release()


# df = pd.DataFrame(np.array([left_leg_angles,right_leg_angles]).T)
# df.to_csv(name + '_point.csv')

print(time.time()-timeTaken)



def hello():
    print('hellog ')
    
    
    
    
print( str(count) + " Squats" )
