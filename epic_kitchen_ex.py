import torch.hub
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
import re
import cv2
import argparse
import pandas as pd
from darknet import Darknet
from util import *
from PIL import Image
import transforms



def arg_parse():
    """
    Parse arguements to the detect module.
    """

    parser = argparse.ArgumentParser(description='EPIC Kitchen dataset testing')

    parser.add_argument('--video_file', type=str, default='')

    parser.add_argument('--frame_folder', type=str, default='')
    parser.add_argument('--modality', type=str, default='RGB',
                        choices=['RGB', 'Flow', 'RGBDiff'], )
    parser.add_argument('--dataset', type=str, default='jester',
                        choices=['something', 'jester', 'moments', 'somethingv2'])
    parser.add_argument('--rendered_output', type=str, default='test')
    parser.add_argument('--arch', type=str, default="InceptionV3")
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--test_segments', type=int, default=8)
    parser.add_argument('--img_feature_dim', type=int, default=256)
    parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
    parser.add_argument('--weights', type=str, default='pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')

    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weightsfile", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    
    return parser.parse_args()



repo = 'epic-kitchens/action-models'

class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'

tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens', force_reload=True)
trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')
mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                      base_model=base_model,
                      pretrained='epic-kitchens')
tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')



# Show all entrypoints and their help strings
for entrypoint in torch.hub.list(repo):
    print(entrypoint)
    print(torch.hub.help(repo, entrypoint))



# file path of the video file
videofile = '../Gordon_Ramsay_perfect_burger_tutorial.mp4'

cap = cv2.VideoCapture(videofile)


args = arg_parse()


# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupScale(tsn.scale_size),
    transforms.GroupCenterCrop(tsn.input_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(tsn.input_mean, tsn.input_std),
])


def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))



# check if CUDA is available
CUDA = torch.cuda.is_available()


frames = 0
start = time.time()

target_num_of_frames = 160

imgs = []

trn.eval()

# load annotation files as data frames
df_n = pd.read_csv('./epic_annotations/EPIC_noun_classes.csv')
df_v = pd.read_csv('./epic_annotations/EPIC_verb_classes.csv')


# use while loop to iterate the frames of the target video
while cap.isOpened():
    ret, frame = cap.read()  # read the new frame

    # use if-else statement to check if there is remaining frame.
    if ret:
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        imgs.append(pil_im)

        # check the number of stored frames
        if len(imgs) >= target_num_of_frames:
            input_frames = load_frames(imgs)
            print(input_frames)
            data = transform(input_frames)

            input = ''
            if CUDA:
                with torch.no_grad():
                    input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda())
            else:
                with torch.no_grad():
                    input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0))

            features = tsn.features(inputs) # extract features from the inputs
            verb_logits, noun_logits = tsn.logits(features) # extract logits

            # Get the probabilities and indices for the verb
            h_x_verbs = torch.mean(F.softmax(verb_logits, 1), dim=0).data
            probs_v, idx_v = h_x_verbs.sort(0, True)

            # Get the probabilities and indices for the noun
            h_x_nouns = torch.mean(F.softmax(noun_logits, 1), dim=0).data
            probs_n, idx_n = h_x_nouns.sort(0, True)


            for i in range(0, 5):
                #print('{:.3f} -> {}'.format(probs_v[i], categories[idx_v[i]]))
                #print('{:.3f} -> {}'.format(probs_n[i], categories[idx_n[i]]))
                print('{:.3f} -> {}'.format(probs_v[i], idx_v[i]))
                print('{:.3f} -> {}'.format(probs_n[i], idx_n[i]))


            #TODO improve the program by write some codes that do something with extracted probabilities and indices

            imgs = []

    frames += 1
