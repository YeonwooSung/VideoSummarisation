from sklearn.metrics import confusion_matrix
import pandas as pd


def getObjectsAndAnswersFromFile(file_path):
    df = pd.read_csv(file_path)
    objects = df['object'].tolist()
    answers = df['answer'].tolist()

    return objects, answers


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    fp.close()
    return names


def print_cm(cm, labels, f_output, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    Pretty print for confusion matrixes.

    origin : <https://gist.github.com/zachguo/10296432>
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ", file=f_output)
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ", file=f_output)
    print(file=f_output)
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ", file=f_output)
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ", file=f_output)
        print(file=f_output)


def generateEncodedLabels(classes):
    labels = []
    index = 1
    for c in classes:
        labels.append(str(index))
        index += 1
    return labels


def evaluate_confusionMatrix_food100():
    y_pred, y_true = getObjectsAndAnswersFromFile('./detected_food.csv')
    classes = load_classes('./food100/food100.names')
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    encoded_labels = generateEncodedLabels(classes)

    f_output = open('./confusionMatrix_food.txt', "w")
    #print(cm, file=f_output)
    print_cm(cm, encoded_labels, f_output)
    f_output.close()


def evaluate_confusionMatrix_utensils():
    y_pred, y_true = getObjectsAndAnswersFromFile('./detected.csv')
    classes = load_classes('../utensils/utensils.names')
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    encoded_labels = generateEncodedLabels(classes)

    f_output = open('./confusionMatrix_utensils.txt', "w")
    #print(cm, file=f_output)
    print_cm(cm, encoded_labels, f_output)
    f_output.close()


if __name__ == '__main__':
    #evaluate_confusionMatrix_food100()
    evaluate_confusionMatrix_utensils()
