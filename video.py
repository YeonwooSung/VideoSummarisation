from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from detectedObject import DetectedObject, compareObjectLists
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module.
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "video.avi", type = str)
    
    return parser.parse_args()


args = arg_parse()

batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)

# constants
numOfFrames_output = 3
numOfTurns = 5

# global variables to check the cost time
numOfUnselectedTurns = 0
start = 0

# check if the CUDA is available
CUDA = torch.cuda.is_available()


num_classes = 80
classes = load_classes("data/coco.names")

# lists to store the detected objects
objectList = []
previousList = []
lastSelectedList = []

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


model.eval()  # Set the model in evaluation mode


def write(x, results):
    vertex1 = tuple(x[1:3].int())  # The first vertex
    vertex2 = tuple(x[3:5].int())  # The other vertex, which is opposite to c1

    cls = int(x[-1])

    color = random.choice(colors)
    label = "{0}".format(classes[cls])

    img = parseResult(x, results)

    cv2.rectangle(img, vertex1, vertex2, color, 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    vertex2 = vertex1[0] + t_size[0] + 3, vertex1[1] + t_size[1] + 4

    cv2.rectangle(img, vertex1, vertex2, color, -1)

    cv2.putText(img, label, (vertex1[0], vertex1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def parseResult(x, results):
    vertex1 = tuple(x[1:3].int()) # The first vertex
    vertex2 = tuple(x[3:5].int()) # The other vertex, which is opposite to c1

    img = results

    cls = int(x[-1])

    label = "{0}".format(classes[cls])
    
    obj = DetectedObject()
    obj.setLabel(label)
    obj.setVertices(vertex1, vertex2)

    f.write('\t{0}\r\n'.format(obj.getInfoString()))
    
    # push the detected object to the list
    objectList.append(obj)

    return img



# Detection phase

videofile = args.videofile  # args.videofile = path to the video file.

cap = cv2.VideoCapture(videofile)  # cap = cv2.VideoCapture(0) -> for webcam

# use assert statement to check if the VideoCpature object is available to use
assert cap.isOpened(), 'Cannot capture source'

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

vWriter = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), numOfFrames_output, (frame_width, frame_height))

frames = 0
start = time.time()

# open the file stream instance to write a file
f = open('testOutput.txt', 'w+')



# use while loop to iterate the frames of the target video
while cap.isOpened():
    ret, frame = cap.read() # read the new frame

    # use if-else statement to check if there is remaining frame.
    if ret:
        img = prep_image(frame, inp_dim)

        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

        # check the type of the output
        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)

            # check if the user pressed the 'q' button to quit the program
            if key & 0xFF == ord('q'):
                break
            continue


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)

        # rescale the output
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1)) / 2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1)) / 2
        output[:,1:5] /= scaling_factor

        #TODO ???
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes('data/coco.names')

        colors = pkl.load(open("pallete", "rb")) #load the binary data of colors from the pallete

        # write text to the file
        f.write('\ncurrent frame: %d\n' % frames)

        # use the lambda to draw rectangles on the frames
        list(map(lambda x: parseResult(x, frame), output))

        #TODO cv2.imshow("frame", frame)  # show the modified frame to the user

        # cv2.waitKey(time) waits for "time" miliseconds to get the value of the pressed key
        # "& 0xFF" is essential for 64bit OS - not necessary for 32bit OS
        key = cv2.waitKey(1) & 0xFF

        # check if the pressed key is 'q'
        if key == ord('q'):
            break

        frames += 1 # increase the number of frames that are processed

        timeCost = time.time() - start
        print(timeCost)
        print("FPS of the video is {:5.2f}".format( frames / (timeCost)))


        if (len(previousList) > len(objectList)):
            objectList = previousList
        else:
            # update the previousList to the current list
            previousList = objectList


        if (frames % numOfTurns == 0):
            # iterate the object lists, and check if the object
            if (compareObjectLists(objectList, lastSelectedList)):
                # write the frame
                vWriter.write(frame)
            else:
                numOfUnselectedTurns += 1

                if (numOfUnselectedTurns > 100):
                    vWriter.write(frame)  # write the frame
                    numOfUnselectedTurns = 0

    else:
        break


# close the file stream
f.close()

# When everything done, release the video capture object
cap.release()
vWriter.release()

# Closes all the frames
cv2.destroyAllWindows()
