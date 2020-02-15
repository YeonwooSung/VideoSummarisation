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
    parser.add_argument('--video_file', type=str, default='./video.avi')

    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help="Activate debug mode.")

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

args = arg_parse()           # argument parser
videofile = args.video_file  # file path of the video file
is_debug = args.debug        # to chceck if the user wants to activate the debugging messages

class_counts = (125, 352)  # num of verbs = 125, num of nouns = 352
segment_count = 8
base_model = 'resnet50'    # 'resnet50' or 'BNInception'
modality = args.modality

tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, modality,
                     base_model=base_model,
                     pretrained='epic-kitchens', force_reload=True)
trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, modality,
                     base_model=base_model,
                     pretrained='epic-kitchens')
mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, modality,
                      base_model=base_model,
                      pretrained='epic-kitchens')
tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, modality,
                     base_model=base_model,
                     pretrained='epic-kitchens')



# Show all entrypoints and their help strings
for entrypoint in torch.hub.list(repo):
    print(entrypoint)
    print(torch.hub.help(repo, entrypoint))



cap = cv2.VideoCapture(videofile)
P_VAL = 0.85

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

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
fps = 0

# get the fps of the target video
if int(major_ver) < 3:
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

#TODO
#target_num_of_frames = fps
#if fps < 8:
    #target_num_of_frames = 8
target_num_of_frames = 8
imgs = [] # a list to store frames

# change the models to the evaluation mode
trn.eval()
tsn.eval()
tsm.eval()

# open the file stream instance to write a file
f = open('actionDetection_output.txt', 'w+')

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

        # use try catch statement to catch the wordnet errors
        try:
            # some labels contain ':' (i.e. door:kitchen)
            # To avoid possible error, the system will check if the noun contains ':'
            # If so, the noun will be splitted, and use the first element of splitted words as a target noun.
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
    # use for loop to iterate a list of nouns
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
    # Check if the probability value of all 3 verbs are greater than P_VAL
    if p1 > P_VAL and p2 > P_VAL and p3 > P_VAL:
        # If the probability value of all 3 verbs are greater than P_VAL, all 3 of them could be the answer
        # So, rather than choosing the 1 best verb, the system will concatenate all verbs with "OR".
        #
        # i.e.
        # verb1 = "put", verb2 = "turn-on", verb3 = "take"
        # result = "put OR turn-on OR take"

        if verb1 is verb2:
            if verb2 is verb3:
                return verb1
            else:
                return '{} OR {}'.format(verb1, verb3)
        else:
            if verb2 is verb3:
                return '{} OR {}'.format(verb1, verb3)
            elif verb1 is verb3:
                return '{} OR {}'.format(verb1, verb2)
            else:
                return '{} OR {} OR {}'.format(verb1, verb2, verb3)

    # Choose the best verb by comparing each verbs and probability values.
    # If the system cannot choose the best verb, than it will just use the random method to
    # find the best verb from 3 chosen verbs.

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
            rand_val = randrange(3)  # rand_val = int(0..2)
            return v_list[rand_val]


def mergeLists(first_list, second_list):
    """
    Merge the list of nouns.

    :param first_list: The first noun list
    :param second_list: The second noun list
    """
    in_first = set(first_list)
    in_second = set(second_list)
    in_second_but_not_in_first = in_second - in_first
    result = first_list + list(in_second_but_not_in_first)
    return result


def mergeSubResults_nouns(n1, n2, n3):
    """
    Merge the elements of lists that contain the detected nouns.

    :param n1: The first noun list
    :param n2: The second noun list
    :param n3: The third noun list
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
                rand_val = randrange(3) #rand_val = int(0..2)
                return n_list[rand_val]


def removeUnnecessaryNouns(list_n, list_p):
    best_p = list_p[0]
    p_limit = 0

    if best_p > 0.5:
        p_limit = 0.3
    else:
        p_limit = best_p / 2
    index_list = []

    for i, (n, p) in enumerate(zip(list_n, list_p)):
        if p < p_limit:
            index_list.append(i)
    index_list.reverse()
    
    for i in index_list:
        list_n.pop(i)
    return list_n


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

    probs_v1, idx_v1, probs_n1, idx_n1 = extractProbsAndIndex(mtrn, inputs)
    probs_v2, idx_v2, probs_n2, idx_n2 = extractProbsAndIndex(tsm, inputs)
    probs_v3, idx_v3, probs_n3, idx_n3 = extractProbsAndIndex(tsn, inputs)

    # print out probabilities of detected verbs and nouns
    print('MTRN')
    verb1, probs_v1, noun_list1 = printVerbsAndNounsWithProbs(probs_v1, idx_v1, probs_n1, idx_n1)
    print('\nTSM')
    verb2, probs_v2, noun_list2 = printVerbsAndNounsWithProbs(probs_v2, idx_v2, probs_n2, idx_n2)
    print('\nTSN')
    verb3, probs_v3, noun_list3 = printVerbsAndNounsWithProbs(probs_v3, idx_v3, probs_n3, idx_n3)

    # get final verb
    verb_final = mergeSubResults_verb(verb1, verb2, verb3, probs_v1, probs_v2, probs_v3)
    # get best noun
    best_noun = findBestNoun(noun_list1[0], noun_list2[0], noun_list3[0], probs_n1[0], probs_n2[0], probs_n3[0])
    print('[LOG] final verb = {0}\r\n[LOG] final noun = {1}'.format(verb_final, best_noun))

    # remove unnecessary nouns from each noun list
    noun_list1 = removeUnnecessaryNouns(noun_list1, probs_n1)
    noun_list2 = removeUnnecessaryNouns(noun_list2, probs_n2)
    noun_list3 = removeUnnecessaryNouns(noun_list3, probs_n3)

    # get final noun list
    noun_list_res = mergeSubResults_nouns(noun_list1, noun_list2, noun_list3)

    #TODO set the best noun as a head of the noun_list_res

    f.write('from {0} to {1} :\r\n'.format(prev_idx, frames))
    f.write('v={}\r\n'.format(verb_final))
    f.write('n={}\r\n'.format(len(noun_list_res)))

    for noun_final in noun_list_res:
        f.write('{}\r\n'.format(noun_final))


# use while loop to iterate the frames of the target video
while cap.isOpened():
    ret, frame = cap.read()  # read the new frame

    # check if the debugging mode is activated
    if is_debug:
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
        ret2, frame2 = cap.read()
        if ret2:
            cv2_im = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            imgs.append(pil_im)
            frames += 1
        else:
            # check if the debugging mode is activated
            if is_debug:
                print('[DEBUG] Error while reading the frame!')
            break
    frames += 1


# check if there is any remaining frames that the action detection system did not check
if len(imgs) > segment_count:
    processActionDetection()


f.close() # close the file stream
cap.release() # release the VideoCapture object
