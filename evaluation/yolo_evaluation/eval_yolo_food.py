from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def readFile(file_path):
    f = open(file_path, 'r')
    firstLine = True
    detected = []
    answer = []

    for line in f:
        l = line.strip()
        if firstLine: 
            firstLine = False
        else:
            splitted = line.split(',')
            detected.append(splitted[1].strip())
            answer.append(splitted[2].strip())
    f.close()
    return detected, answer


def calculate_TP_TN_FP_FN(predicted, actual, label):
    tp, tn, fp, fn = 0, 0, 0, 0

    for predict, answer in zip(predicted, actual):
        p = predict.strip()
        a = answer.strip()

        if p == label:
            # TP or TN
            if p == a:
                tp += 1
            else:
                tn += 1
        else:
            # FP or FN
            if p == a:
                fp += 1
            else:
                fn += 1

    return tp, tn, fp, fn


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 eval_yolo_food.py [1 or 2]')
        exit(1)
    
    try:
        mode = int(sys.argv[1])
    except ValueError:
        print('Usage: python3 eval_yolo_food.py [1 or 2]')
        exit(1)

    detected, answer = readFile('detected_food_top10.csv')

    labels = [
        'rice', 'beef-curry', 'sushi',
        'fried-rice', 'toast', 'hamburger',
        'sandwiches', 'ramen-noodle', 'miso-soup',
        'egg-sunny-side-up', 'other'
    ]

    if mode == 1:
        cm = confusion_matrix(answer, detected)

        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(cm, cmap='Reds', ax=ax)
        plt.savefig('confusionMatrix_food_modified.png')

    else:
        print('Calculate TP, TN, FP, FN for all target labels\n')
        # calculate TP,TN, FP, FN table
        for label in labels:
            tp, tn, fp, fn = calculate_TP_TN_FP_FN(detected, answer, label)
            print('{}:\n\tTP = {}\n\tTN = {}\n\tFP = {}\n\tFN = {}'.format(label, tp, tn, fp, fn))
