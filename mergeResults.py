from __future__ import division
import argparse
from itertools import product
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


def validateWord(word):
    if ':' in word:
        return word.split(':')[0].strip()
    elif '-' in word:
        return word.split('-')[0].strip()
    return word

def calculateSimilarity(w1, w2):
    """
    Calculate the similarity between 2 words by using wordnet.

    :param w1: The first word
    :param w2: The second word

    :return max_sims: The calculated similarity value.
    :return checker:  The flag value that shows if an error occurred while executing this function.
    """
    if w1 == w2:
        return None, False

    # validate the form of given words
    word1 = validateWord(w1)
    word2 = validateWord(w2)

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
            if s:
                sims.append(s)
        if len(sims) > 0:
            max_sims = max(sims)
            checker = True
    else:
        print('[DEBUG] calculateSimilarity::Error ==> word1=({}), word2=({})'.format(word1, word2))
    return max_sims, checker


def compareAndCalculateSimilarityList(comb_list):
    """
    """
    result = []
    syns_res = {}

    for comb in comb_list:
        w1 = comb[0]
        w2 = comb[1]

        max_sims, checker = calculateSimilarity(w2, w1)
        # check if the error occurred in the calculateSimilarity() function
        if checker:
            # check if the syns_res contains the current word as an attribute
            if w2 not in syns_res:
                syns_res[w2] = [max_sims]
            else:
                syns_res[w2].append(max_sims)

    resultList = []

    # iterate a dictionary to find the most irrelevant word that should be removed
    for key in syns_res:
        sims = syns_res[key]
        # calculate the sum of all similarity values in a list
        sum_sims = sum(sims)
        resultList.append((key, sum_sims))

    return resultList


def readActionDetectionResult(f, exec_mode):
    info_line = f.readline().strip()
    v_line = f.readline().strip()
    n_line = f.readline().strip()

    if info_line == '' or v_line == '' or n_line == '':
        return None, None, None

    # check if the user select "debug" mode
    if exec_mode == 'debug':
        print('[DEBUG] ' + info_line)
        print('[DEBUG] ' + v_line)
        print('[DEBUG] ' + n_line)

    v = v_line.replace('v=', '')
    n_num = int(n_line.replace('n=', ''))
    n = []

    for i in range(n_num):
        n.append(f.readline().strip())

    return info_line, v, n

def readObjectDetectionResult(f, exec_mode):
    obj = None
    objList= []
    objects = set()

    for l in f:
        line = l.strip()
        if 'current frame:' in line:
            if obj:
                objList.append(obj)
            frame_num = line.replace('current frame:', '').strip()

            # initialise the variables
            obj = {}
            objects = set()
            obj['frame_num'] = int(frame_num)
            obj['objects'] = []
        else:
            # check if the user select "debug" mode
            if exec_mode == 'debug':
                print('[DEBUG] object=' + line)

            # check if the current object is duplicating
            if not (line in objects):
                obj['objects'].append(line)
            else:
                if exec_mode == 'debug':
                    print('[DEBUG] "{}" is a duplicating word!'.format(line))

    # check if there is at least one object
    if obj:
        objList.append(obj)
    # check if the user select "debug" mode
    if exec_mode == 'debug':
        print('[DEBUG] Length of objList= {}'.format(len(objList)))

    return objList


def getAllCombinationsOf2Lists(list1, list2):
    all_combinations = [[i, j] for i in list1 for j in list2]
    return all_combinations


def filterWordsByThreshold(tuple_list, exec_mode):
    max_sim = 0
    max_w = None
    w_list = []

    for (w, s) in tuple_list:
        # check if the user select the debugging mode to print out the debugging message
        if exec_mode == 'debug':
            print('[DEBUG] word={} - sum(similarity_value)={}'.format(w, s))
        w_list.append(w)

        # check if the current similarity value is greater than previous max value
        if s > max_sim:
            max_w = w
            max_sim = s

    if exec_mode == 'debug':
        print('--------------------------------')

    # check if the program found the maximum similarity value
    if max_w is None:
        return w_list

    # threshold value for filtering the unnecessary words
    sim_thresh = max_sim / 2

    # use for loop to find unnecessary words.
    for (w, s) in tuple_list:
        if s < sim_thresh:
            # remove the unnecessary word, whose similarity value is less than threshold
            w_list.remove(w)

            # check if the user select the debugging mode to print out the debugging message
            if exec_mode == 'debug':
                print('[DEBUG] threshold_value={}, similarity_value={}, removed_word={}'.format(sim_thresh, s, w))

    return w_list


def checkFrameRangeForSimilarityCalculation(start_frame, end_frame, objList, v, n_list, index, exec_mode):
    while True:
        # to avoid IndexOutOfBounds error
        if index >= len(objList):
            break
        # get a object from the object list
        obj = objList[index]
        frame_num = obj['frame_num']

        # print out debug message if it is debugging mode
        if exec_mode == 'debug':
            print('[DEBUG] frame={}, start_frame={}, end_frame={}'.format(frame_num, start_frame, end_frame))

        # check the range of frame number
        if start_frame <= frame_num and end_frame >= frame_num:
            objects = obj['objects']
            v_list = [v]

            # get a list of all possible combinations of verbs and nouns
            combinations_v_n = getAllCombinationsOf2Lists(v_list, n_list)
            # get the list of tuples where each tuple contains the noun and the sum of similarity values
            res_list = compareAndCalculateSimilarityList(combinations_v_n)
            # filter the unnecessary nouns by calculating the threshold similarity value
            n_list = filterWordsByThreshold(res_list, exec_mode)

            # get a list of all possible combinations of verbs and objects
            combinations_v_obj = getAllCombinationsOf2Lists(v_list, objects)
            # get the list of tuples where each tuple contains the object and the sum of similarity values
            res_list = compareAndCalculateSimilarityList(combinations_v_obj)
            # filter the unnecessary objects by calculating the threshold similarity value
            objects = filterWordsByThreshold(res_list, exec_mode)

            # get a list of all possible combinations of nouns and objects
            combinations_n_obj = getAllCombinationsOf2Lists(n_list, objects)
            # get the list of tuples where each tuple contains the object and the sum of similarity values
            res_list3 = compareAndCalculateSimilarityList(combinations_n_obj)
            # filter the unnecessary objects by calculating the threshold similarity value
            objects = filterWordsByThreshold(res_list, exec_mode)

            #TODO store the updated v_list, n_list, and objects !!!

            index += 1
        else:
            break
    return index


if __name__ == '__main__':
    args = arg_parse()  # generate argument parser

    exec_mode = args.mode  # get the execution mode (either normal or debug)

    # file path of result files
    obj_output_file_path = args.obj_result
    action_output_file_path = args.action_result

    f_obj = open(obj_output_file_path, 'r')     # file stream to read the result file of the object detection
    f_act = open(action_output_file_path, 'r')  # file stream to read the result file of the action detection

    # read object lists via file stream
    objList = readObjectDetectionResult(f_obj, exec_mode)
    index = 0

    #TODO store the previous verbs and nouns -> compare with current verb and nouns
    prev_v = ''
    prev_n = []
    start = 0

    while True:
        info_line, v, n = readActionDetectionResult(f_act, exec_mode)

        # The readActionDetectionResult returns None if it gets the EOF.
        # Thus, by checking if the readActionDetectionResult function returns None,
        # the program could check if it shold break the endless loop.
        if info_line is None or v is None or n is None:
            break

        # get start and end frame number from the info_line
        frame_nums = info_line.replace(':', '').replace('from', '').strip().split(' to ')
        start_frame = int(frame_nums[0].strip())
        end_frame = int(frame_nums[1].strip())

        # check if the verb has been changed - use the verb to detect the change point
        if prev_v != v and start != start_frame:
            if exec_mode == 'debug':
                print('[DEBUG] change point detected - prev_v={}, v={}'.format(prev_v, v))

            #TODO change point

            # update values of the variables with new verb, noun list, and frame range number
            prev_v = v
            prev_n = n
            start = start_frame

        # To remove duplicating elements, convert list to set, and convert the set to list
        nouns = list(set(n))
        # calculate the similarity values to find the unnecessary words that should be removed
        index = checkFrameRangeForSimilarityCalculation(start_frame, end_frame, objList, v, nouns, index, exec_mode)

        #TODO 1) compare verb and nouns -> remove unnecessary nouns
        #TODO 2) compare verbs and objects -> remove unnecessary objects
        #TODO 3) compare nouns and objects -> remove unnecessary objects


    # close file stream objects
    f_obj.close()
    f_act.close()
