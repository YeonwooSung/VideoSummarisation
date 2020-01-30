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
import nltk
from nltk.corpus import wordnet as wn
from random import randrange



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


batch_size = 1
segment_count = 8
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224

inputs = torch.randn(
    [batch_size, segment_count, snippet_length, snippet_channels, height, width]
)


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
prev_idx = 0
start = time.time()

target_num_of_frames = 160

imgs = []

trn.eval()

# load annotation files as data frames
df_n = pd.read_csv('./epic_annotations/EPIC_noun_classes.csv')
df_v = pd.read_csv('./epic_annotations/EPIC_verb_classes.csv')

# convert the class_key column to the list
class_key_n = df_n['class_key'].tolist()
class_key_v = df_v['class_key'].tolist()


def extractProbsAndIndex(model, inputs):
    features = model.features(inputs) # extract features from the inputs
    verb_logits, noun_logits = model.logits(features) # extract logits

    # Get the probabilities and indexes for the verb
    h_x_verbs = torch.mean(F.softmax(verb_logits, 1), dim=0).data
    probs_v, idx_v = h_x_verbs.sort(0, True)

    # Get the probabilities and indexes for the noun
    h_x_nouns = torch.mean(F.softmax(noun_logits, 1), dim=0).data
    probs_n, idx_n = h_x_nouns.sort(0, True)

    return probs_v, idx_v, probs_n, idx_n



def printVerbsAndNounsWithProbs(probs_v, idx_v, probs_n, idx_n):
    noun_list = []
    noun_synset_list = []

    print('Top5 verbs')
    for i in range(0, 5):
        print('P(Verb) = {:.3f} -> verb = {}'.format(probs_v[i], class_key_v[idx_v[i]]))

    print('Top5 nouns')
    for i in range(0, 5):
        current_noun = class_key_n[idx_n[i]]
        print('P(Noun) = {:.3f} -> noun = {}'.format(probs_n[i], current_noun))

        try:
            if ':' in current_noun:
                temp_n = current_noun.split(':')[0]
                noun_synset_list.append(wn.synset('{}.n.1'.format(temp_n)))
                noun_list.append(current_noun)

            else:
                noun_synset_list.append(wn.synset('{}.n.1'.format(current_noun)))
                noun_list.append(current_noun)
        except:
            print('Error in wordnet!')

    similarity_list = []
    for cur_n, cur_synset in zip(noun_list, noun_synset_list):
        total_sim = 0
        
        for target_n, target_synset in zip(noun_list, noun_synset_list):
            if cur_n is target_n:
                continue

            cur_sim = cur_synset.path_similarity(target_synset)
            total_sim += cur_sim
        
        similarity_list.append(total_sim)
    

    # local variables to find the noun whose total sum of the similarity scores with other nouns is the lowest.
    min_val = similarity_list[0]
    min_idx = 0

    # use for loop to iterate similarity_list
    for i in range(1, len(similarity_list)):
        cur_val = similarity_list[i]

        if min_val > cur_val:
            min_idx = i
            min_val = cur_val

    # If the min_idx is 4 or 5, and probability value of the probs_n[min_idx] is less than (probs_n[0] / 2), remove this noun from the list.
    if min_idx > 3 and (probs_n[min_idx] < (probs_n[0] / 2)):
        noun_list.pop(min_idx)

    print('The noun with the higest probability = {}'.format(class_key_n[idx_n[0]]))
    print('The verb with the higest probability = {}'.format(class_key_v[idx_v[0]]))

    return class_key_v[idx_v[0]], probs_v[0], noun_list


def mergeSubResults_verb(verb1, verb2, verb3, p1, p2, p3):
    if verb1 is verb2:
        if verb2 is verb3:
            return verb1
        else:
            # If verb1== verb2 but verb1 != verb3, then check the probabilties to find the most suitable verb
            if p3 > 0.8 and p2 < 0.5 and p1 < 0.5:
                return verb3
            else:
                return verb2
    elif verb2 is verb3:
        #TODO
        if p1 > 0.8 and p2 < 0.5 and p3 < 0.5:
            return verb1
        else:
            return verb2
    else:
        # If all verbs all different, then compare the probabilities to find the most suitable verb
        if p1 > p2 and p1 > p3:
            return verb1
        elif p2 > p1 and p2 > p3:
            return verb2
        elif p3 > p1 and p3 > p2:
            return verb3
        else:
            # If there is no biggest probability value, then choose random one.
            v_list = [verb1, verb2, verb3]
            rand_val = randrange(3) + 1
            return v_list[rand_val]


def mergeLists(first_list, second_list):
    in_first = set(first_list)
    in_second = set(second_list)
    in_second_but_not_in_first = in_second - in_first
    result = first_list + list(in_second_but_not_in_first)
    return result


def mergeSubResults_nouns(n1, n2, n3):
    """
    Merge the elements of lists that contain the detected nouns.
    """
    temp_res = mergeLists(n1, n2)
    res = mergeLists(temp_res, n3)
    return res


def findBestNoun(n1, n2, n3 ,prob_n1, prob_n2, prob_n3):
    if n1 == n2:
        if n2 == n3:
            return n1
        else:
            if prob_n1 < 0.5 and prob_n2 < 0.5 and prob_n3 > 0.8:
                return n3
            return n1
    else:
        if n2 == n3:
            if prob_n2 < 0.5 and prob_n3 < 0.5 and prob_n1 > 0.8:
                return n1
            return n2
        else:
            if prob_n1 > prob_n2 and prob_n1 > prob_n3:
                return n1
            elif prob_n2 > prob_n1 and prob_n2 > prob_n3:
                return n2
            elif prob_n3 > prob_n1 and prob_n3 > prob_n2:
                return n3
            else:
                # If there is no biggest probability value, then choose random one.
                n_list = [n1, n2, n3]
                rand_val = randrange(3) + 1
                return n_list[rand_val]


def processActionDetection():
    input_frames = load_frames(imgs)
    print(input_frames)
    data = transform(input_frames)

    if CUDA:
        with torch.no_grad():
            inputs = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda())
    else:
        with torch.no_grad():
            inputs = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0))


    probs_v1, idx_v1, probs_n1, idx_n1 = extractProbsAndIndex(tsn, inputs)
    probs_v2, idx_v2, probs_n2, idx_n2 = extractProbsAndIndex(tsm, inputs)
    probs_v3, idx_v3, probs_n3, idx_n3 = extractProbsAndIndex(trn, inputs)

    # print out probabilities of detected verbs and nouns
    print('TSN')
    verb1, probs_v1, noun_list1 = printVerbsAndNounsWithProbs(probs_v1, idx_v1, probs_n1, idx_n1)
    print('\nTSM')
    verb2, probs_v2, noun_list2 = printVerbsAndNounsWithProbs(probs_v2, idx_v2, probs_n2, idx_n2)
    print('\nTRN')
    verb3, probs_v3, noun_list3 = printVerbsAndNounsWithProbs(probs_v3, idx_v3, probs_n3, idx_n3)

    # get final verb
    verb_final = mergeSubResults_verb(verb1, verb2, verb3, probs_v1, probs_v2, probs_v3)

    # get best noun
    best_noun = findBestNoun(noun_list1[0], noun_list2[0], noun_list3[0], probs_n1[0], probs_n2[0], probs_n3[0])

    print('[LOG] final verb = {0}\r\n[LOG] final noun = {1}'.format(verb_final, best_noun))

    # get final noun list
    noun_list_res = mergeSubResults_nouns(noun_list1, noun_list2, noun_list3)

    f.write('from {0} to {1} :\r\n'.format(prev_idx, frames))
    f.write('v={}\r\n'.format(verb_final))
    f.write('n={}\r\n'.format(len(noun_list_res)))

    for noun_final in noun_list_res:
        f.write('{}\r\n'.format(noun_final))


# open the file stream instance to write a file
f = open('actionDetection_output.txt', 'w+')


# use while loop to iterate the frames of the target video
while cap.isOpened():
    ret, frame = cap.read()  # read the new frame

    print('[DEBUG] frames={}'.format(frames))

    # use if-else statement to check if there is remaining frame.
    if ret:
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        imgs.append(pil_im)

        # check the number of stored frames
        if len(imgs) >= target_num_of_frames:
            processActionDetection()

            imgs = []
            prev_idx = frames + 1
    
    else:
        break

    frames += 1


if len(imgs) is not 0:
    processActionDetection()


f.close()
cap.release()
