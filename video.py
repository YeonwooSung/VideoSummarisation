from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from preprocess import prep_image, inp_to_image, letterbox_image
import argparse
import os
from darknet import Darknet
from detectedObject import DetectedObject, compareObjectLists, mergeDetectedObjectLists
import pickle as pkl
import pandas as pd
import random
import math


def arg_parse():
    """
    Parse arguements to the detect module.
    """

    parser = argparse.ArgumentParser(description='Object detection for Video Summarisation by using YOLOv3 models')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence_coco", dest = "confidence_coco", help = "Object Confidence to filter predictions - COCO model", default = 0.5)
    parser.add_argument("--confidence_food", dest = "confidence_food", help = "Object Confidence to filter predictions - Food100 model", default = 0.3)
    parser.add_argument("--nms_thresh_coco", dest = "nms_thresh_coco", help = "NMS Threshhold of YOLO COCO model", default = 0.3)
    parser.add_argument("--nms_thresh_food", dest = "nms_thresh_food", help = "NMS Threshhold of Food100 retrained YOLO model", default = 0.3)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", default = "video.avi", type = str)
    parser.add_argument('--logPath', dest="logPath", help="The file path of the text file that the object detection system logs the logging data", default='./output/testOutput.txt', type=str)

    return parser.parse_args()


args = arg_parse()

# constants for the YOLO model
batch_size = int(args.bs)
confidence_coco = float(args.confidence_coco)  # confidence limit of the YOLO COCO model
confidence_food = float(args.confidence_food)  # confidence limit of the YOLO Food100 model
nms_thresh_coco = float(args.nms_thresh_coco)  # threshold value of the YOLO COCO model
nms_thresh_food = float(args.nms_thresh_food)  # threshold value of the YOLO Food100 model
output_path = args.logPath  # file path of the log file


videofile = args.videofile  # args.videofile = path to the video file.

cap = cv2.VideoCapture(videofile)  # cap = cv2.VideoCapture(0) -> for webcam

# use assert statement to check if the VideoCpature object is available to use
assert cap.isOpened(), 'Cannot capture source'

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
fps = 0

# get the fps of the target video
if int(major_ver) < 3:
    fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else:
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


# constants
numOfFrames_output = 10
numOfTurns = fps #TODO 5? 10? 50? 100?

# global variables to check the cost time
start = 0

# check if the CUDA is available
CUDA = torch.cuda.is_available()

# load object names for COCO model
num_classes = 80
classes = load_classes("data/coco.names")

# load object names for Food100 model
num_classes_food = 100
classes_food = load_classes("data/food100.names")


#Set up the neural network
print("Loading network.....")

model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)

model_foodDomain = Darknet('cfg/yolov3-food100.cfg')
model_foodDomain.load_weights('./yolov3-food100.weights')

print("Network successfully loaded")

model.net_info["height"] = args.reso
model_foodDomain.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
    model_foodDomain.cuda()


# Set the model in evaluation mode
model.eval()
model_foodDomain.eval()


def parseResult(x, results, target_classes, detected_object_list):
    """
    Parse the results to get the detected object.

    :param x: Tensor that contains the information about the detected object.
    :param results: Current frame
    :param target_classes: list of classes
    """
    vertex1 = tuple(x[1:3].int()) # The first vertex
    vertex2 = tuple(x[3:5].int()) # The other vertex, which is opposite to c1

    img = results

    if vertex1 == vertex2:
        return img

    cls = int(x[-1])

    print('[DEBUG] cls = ' + str(cls))

    # to avoid the IndexError
    if cls >= len(target_classes) or cls < 0:
        return img

    label = "{0}".format(target_classes[cls]) # get the label name

    obj = DetectedObject()
    obj.setLabel(label)
    obj.setVertices(vertex1, vertex2)

    # push the detected object to the list
    detected_object_list.append(obj)

    return img



# Detection phase

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

vWriter = cv2.VideoWriter('output/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), numOfFrames_output, (frame_width, frame_height))

frames = 0
start = time.time()


# open the file stream instance to write a file
f = open(output_path, 'w+')


# use while loop to iterate the frames of the target video
while cap.isOpened():
    ret, frame = cap.read() # read the new frame

    # use if-else statement to check if there is remaining frame.
    if ret:

        #TODO process object detection only when (frames % num of turns == 0)
        if frames % numOfTurns != 0:
            frames += 1
            continue

        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()


        with torch.no_grad():
            output_general = model(Variable(img), CUDA)
        with torch.no_grad():
            output_food = model_foodDomain(Variable(img), CUDA)

        # gets the results of the object detection
        output_general = write_results(output_general, confidence_coco, num_classes, nms_conf=nms_thresh_coco)
        output_food = write_results(output_food, confidence_food, num_classes_food, nms_conf=nms_thresh_food)

        # check the type of the output
        if type(output_general) == int or type(output_food) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)

            # check if the user pressed the 'q' button to quit the program
            if key & 0xFF == ord('q'):
                break
            continue


        im_dim_origin = im_dim

        im_dim_general = im_dim.repeat(output_general.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim_general, 1)[0].view(-1, 1)

        # rescale the output - general YOLO
        output_general[:, [1,3]] -= (inp_dim - scaling_factor*im_dim_general[:,0].view(-1,1)) / 2
        output_general[:, [2,4]] -= (inp_dim - scaling_factor*im_dim_general[:,1].view(-1,1)) / 2
        output_general[:, 1:5] /= scaling_factor


        im_dim = im_dim_origin

        im_dim_food = im_dim.repeat(output_food.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim_food, 1)[0].view(-1, 1)

        # rescale the output - domain YOLO
        output_food[:, [1,3]] -= (inp_dim - scaling_factor*im_dim_food[:,0].view(-1,1)) / 2
        output_food[:, [2,4]] -= (inp_dim - scaling_factor*im_dim_food[:,1].view(-1,1)) / 2
        output_food[:, 1:5] /= scaling_factor


        # reshape the outputs by using the clamp function

        for i in range(output_general.shape[0]):
            output_general[i, [1,3]] = torch.clamp(output_general[i, [1,3]], 0.0, im_dim_general[i,0])
            output_general[i, [2,4]] = torch.clamp(output_general[i, [2,4]], 0.0, im_dim_general[i,1])
        
        for i in range(output_food.shape[0]):
            output_food[i, [1, 3]] = torch.clamp(output_food[i, [1, 3]], 0.0, im_dim_food[i, 0])
            output_food[i, [2,4]] = torch.clamp(output_food[i, [2,4]], 0.0, im_dim_food[i,1])

        detected_general = []
        detected_food = []

        # use the lambda to draw rectangles on the frames
        list(map(lambda x: parseResult(x, orig_im, classes, detected_general), output_general))
        list(map(lambda x: parseResult(x, orig_im, classes_food, detected_food), output_food))

        #TODO wordnet to filter the objects

        # merge detected_general and detected_food
        objectList = mergeDetectedObjectLists(detected_general, detected_food)

        # cv2.waitKey(time) waits for "time" miliseconds to get the value of the pressed key
        # "& 0xFF" is essential for 64bit OS - not necessary for 32bit OS
        key = cv2.waitKey(1) & 0xFF

        # check if the pressed key is 'q'
        if key == ord('q'):
            break


        f.write('current frame: %d\n' % frames)

        # iterate the object list, and write the objects in the text file via file stream
        for obj in objectList:
            f.write('\t{0}\r\n'.format(obj.getInfoString()))
            #f.write('\t{}\n'.format(obj.getLabel()))

        frames += 1  # increase the number of frames that are processed
        timeCost = time.time() - start
        print(timeCost)
        print("FPS of video processing is {:5.2f}".format(frames / (timeCost)))

    else:
        break


# close the file stream
f.close()

# When everything done, release the video capture object
cap.release()
vWriter.release()

# Closes all the frames
cv2.destroyAllWindows()
