from __future__ import division
import argparse
from itertools import product, permutations
import nltk
from nltk.corpus import wordnet as wn


def arg_parse():
    """
    Generate argument parser.

    :return: The argument parser
    """
    parser = argparse.ArgumentParser(description='Choose objects for the Video Summarisation system')

    #TODO parser.add_argument('', dest='', help='', default='', type=str)
    parser.add_argument("--obj_result", dest="obj_result", help="file path of result file of the object detection system", default='./output/testOutput.txt', type=str)
    parser.add_argument('--action_result', dest='action_result', help='file path of result file of the action detection system', default='./output/actionDetection_output.txt', type=str)
    parser.add_argument('--mode', dest='mode', help='', choices=['debug', 'normal'], default='normal', type=str)

    return parser.parse_args()


def calculateSimilarity(word1, word2):
    """
    Calculate the similarity between 2 words by using wordnet.

    :param word1: The first word
    :param word2: The second word

    :return max_sims: The calculated similarity value.
    :return checker:  The flag value that shows if an error occurred while executing this function.
    """
    # get synsets for the object and noun
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)
    max_sims = None
    checker = False

    if syns1 and syns2:
        sims = []

        # iterate all elements in syns1 and syns2
        for sense1, sense2 in product(syns1, syns2):
            s = wn.wup_similarity(sense1, sense2)
            sims.append(s)
        max_sims = max(sims)
        checker = True
    else:
        print('calculateSimilarity::Error ==> word1=({}), word2=({})'.format(word1, word2))
    return max_sims, checker


def compareAndCalculateSimilarityList(list1, list2):
    """
    """
    result = []
    syns_res = {}

    for w2 in list2:
        for w1 in list1:
            max_sims, checker = calculateSimilarity(w2, w1)

            # check if the error occurred in the calculateSimilarity() function
            if not checker:
                # check if the syns_res contains the current word as an attribute
                if w2 not in syns_res:
                    syns_res[w2] = [max_sims]
                else:
                    syns_res[w2].append(max_sims)
    for key in syns_res:
        sims = syns_res[key]

        #TODO key = object, sims = list of similarity values
        #TODO get the average value of the similarity values, and find the least important obj ??


def readActionDetectionResult(f, exec_mode):
    info_line = f.readline().strip()
    v_line = f.readline().strip()
    n_line = f.readline().strip()

    if info_line == '' or v_line == '' or n_line == '':
        return None, None, None

    # check if the user select "debug" mode
    if exec_mode == 'debug':
        print(info_line)
        print(v_line)
        print(n_line)

    v = v_line.replace('v=', '')
    n_num = int(n_line.replace('n=', ''))
    n = []

    for i in range(n_num):
        n.append(f.readline().strip())

    return info_line, v, n

def readObjectDetectionResult(f, exec_mode):
    obj = None
    objList= []
    for l in f:
        line = l.strip()
        if 'current frame:' in line:
            if obj:
                objList.append(obj)
            frame_num = line.replace('current frame:', '').strip()
            obj = {}
            obj['frame_num'] = frame_num
            obj['objects'] = []
        else:
            obj['objects'].append(line)

            # check if the user select "debug" mode
            if exec_mode == 'debug':
                print(line)
    if obj:
        objList.append(obj)
    # check if the user select "debug" mode
    if exec_mode == 'debug':
        print('Length of objList= {}'.format(len(objList)))

    return objList


if __name__ == '__main__':
    args = arg_parse()  # generate argument parser

    exec_mode = args.mode

    # file path of result files
    obj_output_file_path = args.obj_result
    action_output_file_path = args.action_result

    f_obj = open(obj_output_file_path, 'r')     # file stream to read the result file of the object detection
    f_act = open(action_output_file_path, 'r')  # file stream to read the result file of the action detection

    # read object lists via file stream
    objList = readObjectDetectionResult(f_obj, exec_mode)

    while True:
        info_line, v, n = readActionDetectionResult(f_act, exec_mode)

        # The readActionDetectionResult returns None if it gets the EOF.
        # Thus, by checking if the readActionDetectionResult function returns None,
        # the program could check if it shold break the endless loop.
        if info_line is None or v is None or n is None:
            break

        # get start and end frame number from the info_line
        frame_nums = info_line.replace(':', '').replace('from', '').strip().split(' to ')
        start_frame = frame_nums[0].strip()
        end_frame = frame_nums[1].strip()

        #TODO 1) compare verb and nouns -> remove unnecessary nouns
        #TODO 2) compare verbs and objects -> remove unnecessary objects
        #TODO 3) compare nouns and objects -> remove unnecessary objects


    # close file stream objects
    f_obj.close()
    f_act.close()
