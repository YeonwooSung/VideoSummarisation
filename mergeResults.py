from __future__ import division
import argparse
from itertools import product
import math
import os
import cv2
from nltk.corpus import wordnet as wn
import skvideo.io
import skvideo.utils


def arg_parse():
    """
    Generate argument parser.

    :return: The argument parser
    """
    parser = argparse.ArgumentParser(description='Choose objects for the Video Summarisation system')

    parser.add_argument("--obj_result", dest="obj_result", help="file path of result file of the object detection system", default='./output/testOutput.txt', type=str)
    parser.add_argument('--action_result', dest='action_result', help='file path of result file of the action detection system', default='./output/actionDetection_output.txt', type=str)
    parser.add_argument('--mode', dest='mode', help='Execution mode - either debug or normal', choices=['debug', 'normal'], default='normal', type=str)
    parser.add_argument('--use_n', dest='use_n', help='y for using nouns, n for not using nouns', choices=['y', 'n'], default='n', type=str)
    parser.add_argument('--use_obj', dest='use_obj', help='y for using objects, n for not using objects', choices=['y', 'n'], default='n', type=str)

    return parser.parse_args()


def validateWord(word):
    """
    Validate the word by checking if the given word contains unexpected character.
    """
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
    Compare the words in the given combination list.
    Then, calculate the sum of similarity values for each word.
    Calcualted sum of similarity values will be appended to the result list.

    :param comb_list: Combination list that contains lists of words.
    :return resultList: A list that contains the sum of similarity values
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
    """
    Read lines from the result file of the action detection system.

    :param f: File stream object
    :param exec_mode: The exection mode - either normal or debug

    :return info_line: The information line
    :return v: The verb
    :return n: A list of nouns
    """
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
    """
    Read lines from the result file of the object detection system.

    :param f: File stream object
    :param exec_mode: The exection mode - either normal or debug
    :return objList: A list of objects
    """
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
    """
    Generate a list of all combinations of all elements in 2 lists.

    :param list1: The first list
    :param list2: The second list
    :return: A list of all combinations of all elements in 2 lists.
    """
    all_combinations = [[i, j] for i in list1 for j in list2]
    return all_combinations


def filterWordsByThreshold(tuple_list, exec_mode):
    """
    Calculate the threshold value, and filter the inrelevant words by comparing the similarity
    values with the threshold value.

    :param tuple_list: A list of tuple, where each tuple contains the word and similarity value of that word
    :param exec_mode: The exection mode - either normal or debug

    :return w_list: A filtered word list.
    """
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


def checkFrameRangeForSimilarityCalculation(f, start_frame, end_frame, objList, v, noun_list, index, results, exec_mode):
    """
    Merges the results of action detection system and object detection system.
    When merging results, this function removes the most inrelevant words by comparing similarity values.

    :param f: The file stream instance
    :param start_frame: The start frame number
    :param end_frame: The end frame number
    :param objList: A list of objects
    :param v: The current verb
    :param noun_list: The current noun list
    :param index: A number that helps the program to know the index of current iteration
    :param results: A list of results (each result contains verb, noun, frame_num, and object)
    :param exec_mode: The exection mode - either normal or debug

    :return index: Updated index
    """
    n_list = noun_list
    v_list = [v]

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

            # 1) compare verb and nouns -> remove unnecessary nouns
            # 2) compare verbs and objects -> remove unnecessary objects
            # 3) compare nouns and objects -> remove unnecessary objects


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

            # store the verb and updated n_list, and objects by writing the text via file stream object
            f.write('frame={}\nv={}\nn='.format(frame_num, v))

            n_str = ''
            # iterate the list of nouns
            for noun in n_list:
                n_str += '{} AND '.format(noun)
            if len(n_list) > 0:
                f.write('{}\nobj='.format(n_str[:-5]))
            else:
                f.write('\nobj=')

            obj_str = ''
            # iterate the list of objects
            for obj in objects:
                obj_str += '{} AND '.format(obj)

            # check if there is no objects in the list of objects
            if len(objects) > 0:
                f.write('{}\n'.format(obj_str[:-5]))
            else:
                f.write('\n')
            
            results.append({
                'v': v,
                'n': n_list,
                'obj': objects,
                'frame_num': frame_num
            })

            index += 1
        else:
            break
    return index


def compareObjectLists(objects1, objects2, threshold_n=0.4):
    """
    Compare the object lists to detect the change point by comparing the size of intersection set
    with the calculated threshold value.

    :param objects1: The first object list
    :param objects2: The second object list
    :param threshold_n: The value that is used for calculating threshold value

    :return: Returns True if it finds the change point. Otherwise, returns False.
    """
    if len(objects1) != len(objects2) and (objects1 == [] or objects2 == []):
        return True

    # threshold value to detect change point by comparing lists of objects
    threshold = math.ceil(threshold_n * len(objects1))
    total = 0

    # use nested for loops to compare objects in the object lists
    for obj1 in objects1:
        for obj2 in objects2:
            if obj1 == obj2:
                total += 1

    # compare the size of intersection set with the threshold value
    if threshold <= total:
        return False
    else:
        return True


def compareNounLists(n_list1, n_list2):
    """
    Compare noun lists to find the change point.
    If the size of intersection set is 0 (no intersection), then returns True.
    Otherwise, returns False.

    :param n_list1: The first noun list
    :param n_list2: The second noun list
    :return: Either True or False
    """
    if n_list1 == []:
        return True
    if n_list2 == []:
        return False

    hasIdentical = True

    # use nested for loops to compare all nouns in n_list1 and n_list2
    for n1 in n_list1:
        hasIdentical1 = False
        for n2 in n_list2:
            if n1 == n2:
                hasIdentical1 = True
        hasIdentical = hasIdentical or hasIdentical1

    # hasIdentical = True  -> There is at least one intersection between n_list1 and n_list2
    # hasIdentical = False -> There is no intersection between n_list1 and n_list2
    return not hasIdentical


def compressOutputs(results, use_n, use_obj, exec_mode):
    # open file stream object for compression
    f = open('output/compress_result.txt', 'w+')

    prev_v = ''
    prev_n = []
    prev_objects = []

    frames = []

    # iterate the list of results of merging process, and find the change points (key frames)
    for res in results:
        v = res['v']
        n_list = res['n']
        objects = res['obj']
        frame_num = res['frame_num']

        # check if the verb has been changed to detect the change point
        if prev_v != v:
            # check if it is debugging mode - if so, print out debugging message
            if exec_mode == 'debug':
                print(
                    '[DEBUG] change point detected - prev_v={}, v={}'.format(prev_v, v))

            # write result text
            f.write('{} -> prev_v={}, new_v={}\n'.format(frame_num, prev_v, v))
            # store the frame_num
            frames.append(frame_num)

            # update the values of variables
            prev_v = v
            prev_n = n_list
            prev_objects = objects


        # check if noun_list chagned
        elif compareNounLists(prev_n, n_list) and use_n:
            # check if it is debugging mode - if so, print out debugging message
            if exec_mode == 'debug':
                print('[DEBUG] noun list changed\n\tprev_n={}\n\tn_list={}'.format(prev_n, n_list))

            # write result text
            f.write('{} -> prev_n={}, new_n={}\n'.format(frame_num, prev_n, n_list))
            # store the frame_num
            frames.append(frame_num)

            # update the values of variables
            prev_v = v
            prev_n = n_list
            prev_objects = objects


        # check if objects chagned
        elif compareObjectLists(prev_objects, objects) and use_obj:
            # check if it is debugging mode - if so, print out debugging message
            if exec_mode == 'debug':
                print('[DEBUG] noun list changed\n\tprev_obj={}\n\tobjects={}'.format(prev_objects, objects))

            # write result text
            f.write('{} -> prev_objects={}, new_objects={}\n'.format(frame_num, prev_objects, objects))
            # store the frame_num
            frames.append(frame_num)

            # update the values of variables
            prev_v = v
            prev_n = n_list
            prev_objects = objects

    # close the file stream
    f.close()

    return frames


def mergeResults(obj_output_file_path, action_output_file_path, exec_mode):
    f_obj = open(obj_output_file_path, 'r')     # file stream to read the result file of the object detection
    f_act = open(action_output_file_path, 'r')  # file stream to read the result file of the action detection

    # read object lists via file stream
    objList = readObjectDetectionResult(f_obj, exec_mode)
    index = 0

    f = open('output/merge_result.txt', 'w+')
    results = []

    # use endless loop which will loop until the file stream gets the EOF of the result file
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

        # To remove duplicating elements, convert list to set, and convert the set to list
        nouns = list(set(n))
        # calculate the similarity values to find the unnecessary words that should be removed
        index = checkFrameRangeForSimilarityCalculation(f, start_frame, end_frame, objList, v, nouns, index, results, exec_mode)

    # close file stream objects
    f_obj.close()
    f_act.close()
    f.close()

    return results


def generateOutput(frames, video_name, output_video_name, exec_mode):
    # check if the video file exists
    if (not os.path.exists(video_name)) or (not os.path.isfile(video_name)):
        print('[Error] File not exist "{}"'.format(video_name))
        exit(1)

    videodata = skvideo.io.vread(video_name)
    vid = skvideo.utils.vshape(videodata)
    print('[DEBUG] video_format={}'.format(vid.shape))

    #TODO
    cap = cv2.VideoCapture(video_name)
    fps = 8

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # get the fps of the target video
    if int(major_ver) < 3:
        fps = math.ceil(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    vWriter = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    # initialise variables
    frame_num = 0
    end_num = fps
    f_flag = False
    f_index = 0
    result_frame_num = frames[f_index]
    final_index = len(frames) - 1

    # iterate video frames
    for frame in videodata:
        if frame_num == result_frame_num:
            if exec_mode == 'debug':
                print('[DEBUG] frame_num={}, result_frame_num={}'.format(frame_num, result_frame_num))

            # OpenCV uses BGR not RGB. 
            # Thus, we should convert the color space before write output frames
            # Convert the color space of the frame from RGB to BGR
            image_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vWriter.write(image_frame)

            if not f_flag:
                f_flag = True

            # to avoid IndexError
            if f_index < final_index:
                f_index += 1
                end_num = result_frame_num + fps
                result_frame_num = frames[f_index]

                if exec_mode == 'debug':
                    print('[DEBUG] new: result_frame_num={}, end_num={}'.format(result_frame_num, end_num))

        # write frames
        elif frame_num <= end_num and f_flag:
            # Convert the color space of the frame from RGB to BGR
            image_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vWriter.write(image_frame)
        frame_num += 1



if __name__ == '__main__':
    # generate argument parser
    args = arg_parse()
    # get the execution mode (either normal or debug)
    exec_mode = args.mode

    # file path of result files
    obj_output_file_path = args.obj_result
    action_output_file_path = args.action_result

    # merge results of action detection system and object detection system
    results = mergeResults(obj_output_file_path, action_output_file_path, exec_mode)

    # argparse arguments
    use_n = True if args.use_n == 'y' else False
    use_obj = True if args.use_obj == 'y' else False

    # Compression
    frames = compressOutputs(results, use_n, use_obj, exec_mode)

    # Generate output video by using the results of compression method
    generateOutput(frames, '../YouCook/Videos/0050.mp4', '../output.avi', exec_mode)
