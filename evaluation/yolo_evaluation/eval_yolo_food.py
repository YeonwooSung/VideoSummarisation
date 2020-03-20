from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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
            detected.append(splitted[1])
            answer.append(splitted[2])
    f.close()
    return detected, answer


if __name__ == '__main__':
    detected, answer = readFile('detected_food_top10.csv')
    cm = confusion_matrix(answer, detected)

    labels = [
        'rice', 'beef-curry', 'sushi',
        'fried-rice', 'toast', 'hamburger',
        'sandwiches', 'ramen-noodle', 'miso-soup', 
        'egg-sunny-side-up', 'others'
    ]

    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, cmap='Reds', ax=ax)
    plt.savefig('confusionMatrix_food_modified.png')
