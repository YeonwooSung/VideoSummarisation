import os
import sys


def readLineFromOutputFile(f, frame_num_line, model_type):
    """
    Read lines from output files.

    :param f: File stream object
    :param frame_num_line: The line that indicates the start frame number and end frame number
    :param model_tyep: 1 for TRN, 2 for TSN, and else for TSM

    :return v: Verb
    :return n: List of nouns
    """
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()

    if line1 == "" or line2 == ""  or line3 == "":
        print('readLineFromOutputFile :: Failed to read more lines from output file!')
        exit(0)
    if line1.strip() != frame_num_line.strip():
        print('readLineFromOutputFile :: Invalid frame number!')
        exit(0)

    v = line2.replace('v=', '').strip()
    num_of_nouns = line3.replace('n=', '')
    num_of_n = 0

    try:
        num_of_n = int(num_of_nouns)
    except ValueError:
        print('readLineFromOutputFile :: ValueError occurred!')
        if model_type == 1:
            print('ModelType = TRN')
        elif model_type == 2:
            print('ModelType = TSN')
        else:
            print('ModelType = TSM')
        exit(1)

    n = []
    for i in range(0, num_of_n):
        word = f.readline().strip()
        if word == "" or word.startswith('from'):
            print('readLineFromOutputFile :: Occurred in for loop\n{} is not the expected line'.format(word))
        n.append(word)
    return v, n

def compareWithAnswer(vTrue, nTrue, v_answer, n_answer, v, n):
    """
    Compare the results of prediction with the answer.
    
    :param vTrue: count of correct predicted verbs
    :param nTrue: count of correct predicted nouns
    :param v_answer: Answer verb
    :param n_answer: Answer noun
    :param v: The predicted verb
    :param n: The list of predicted nouns

    :return vTrue_new: Updated vTrue : vTrue + 1 if the v is equal to v_answer
    :return nTrue_new: Updated nTrue : nTrue + 1 if the n contains the n_answer
    """
    vTrue_new = vTrue
    nTrue_new = nTrue

    if v_answer == v:
        vTrue_new = vTrue + 1

    # use for loop to iterate list of predicted nouns
    for w in n:
        if w == n_answer:
            nTrue_new = nTrue + 1
            break
    return vTrue_new, nTrue_new

def readAndCompareOutputs(f, f_trn, f_tsn, f_tsm):
    """
    Read the answers from the answer file, and compare the answers with the predictions.

    :param f: File stream for answer file
    :param f_trn: File stream for the result of TRN
    :param f_tsn: File stream for the result of TSN
    :param f_tsm: File stream for the result of TSM

    :return total_count: Total counts
    :return vTrue_trn: The number of correct verbs that the TRN predicted
    :return nTrue_trn: The number of correct nouns that the TRN predicted
    :return vTrue_tsn: The number of correct verbs that the TSN predicted
    :return nTrue_tsn: The number of correct nouns that the TSN predicted
    :return vTrue_tsm: The number of correct verbs that the TSM predicted
    :return nTrue_tsm: The number of correct nouns that the TSM predicted
    """
    if f == None or f_trn == None or f_tsn == None or f_tsm == None:
        print('readAndCompareOutputs :: Invalid arguments!')
        exit(0)

    total_count = 0
    vTrue_trn = 0
    nTrue_trn = 0
    vTrue_tsn = 0
    nTrue_tsn = 0
    vTrue_tsm = 0
    nTrue_tsm = 0

    # Use endless loop to read until the file stream gets the eof
    while True:
        answer_line1 = f.readline()
        answer_line2 = f.readline()
        answer_line3 = f.readline()

        # Check the file stream objects get the EOF
        if answer_line1 == "" or answer_line2 == "" or answer_line3 == "":
            break

        if not answer_line1.startswith('from'):
            print('[Error] Invalid frame line!')
            print('actual = {}'.format(answer_line1))
            exit(1)

        answer_verb = answer_line2.replace('v=', '').strip()
        answer_noun = answer_line3.replace('n=', '').strip()

        v1, n1 = readLineFromOutputFile(f_trn, answer_line1, 1)
        v2, n2 = readLineFromOutputFile(f_tsn, answer_line1, 2)
        v3, n3 = readLineFromOutputFile(f_tsm, answer_line1, 3)

        total_count += 1
        vTrue_trn, nTrue_trn = compareWithAnswer(vTrue_trn, nTrue_trn, answer_verb, answer_noun, v1, n1)
        vTrue_tsn, nTrue_tsn = compareWithAnswer(vTrue_tsn, nTrue_tsn, answer_verb, answer_noun, v2, n2)
        vTrue_tsm, nTrue_tsm = compareWithAnswer(vTrue_tsm, nTrue_tsm, answer_verb, answer_noun, v3, n3)

    return total_count, vTrue_trn, nTrue_trn, vTrue_tsn, nTrue_tsn, vTrue_tsm, nTrue_tsm

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python3 accuracyTest.py <file_path_of_answer_file>')
        exit(1)
    answerFile = sys.argv[1]
    f_answer = open(answerFile, 'r')

    if not os.path.exists(answerFile):
        print('[Error] Invalid File Path :: Please recheck the file path of the answer file!\ninput = "{}"'.format(answerFile))
        exit(0)
    
    if not os.path.isfile(answerFile):
        print('[Error] Not a file :: The given name is a path of directory!\ninput = "{}"'.format(answerFile))
        exit(0)

    # input file stream objects for intermediate output files
    f_trn = open('trn_output.txt', 'r')
    f_tsn = open('tsn_output.txt', 'r')
    f_tsm = open('tsm_output.txt', 'r')

    # read and compare the results of predictions with answer
    total_count, vTrue_trn, nTrue_trn, vTrue_tsn, nTrue_tsn, vTrue_tsm, nTrue_tsm = readAndCompareOutputs(f_answer, f_trn, f_tsn, f_tsm)
    
    # calculate the accuracy scores
    vTrue_trn = vTrue_trn / total_count
    nTrue_trn = nTrue_trn / total_count
    vTrue_tsm = vTrue_tsm / total_count
    nTrue_tsm = nTrue_tsm / total_count
    vTrue_tsn = vTrue_tsn / total_count
    nTrue_tsn = nTrue_tsn / total_count

    print('TRN -> verb accuracy = {0:.4f}%, noun accuracy = {1:.4f}%'.format(vTrue_trn, nTrue_trn))
    print('TSN -> verb accuracy = {0:.4f}%, noun accuracy = {1:.4f}%'.format(vTrue_tsn, nTrue_tsn))
    print('TSM -> verb accuracy = {0:.4f}%, noun accuracy = {1:.4f}%'.format(vTrue_tsm, nTrue_tsm))

    # close the file stream objects
    f_trn.close()
    f_tsn.close()
    f_tsm.close()
