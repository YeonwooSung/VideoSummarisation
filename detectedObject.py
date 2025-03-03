from __future__ import division
import math


class DetectedObject():
    """
    The aim of this class is to store the information about the objects that the YOLOv3 model detected.
    """

    def __init__(self):
        super().__init__()
        self.vertex1 = []
        self.vertex2 = []
        self.midpoint = []
        self.size = 0
        self.label = ''

    def setVertices(self, vertex1, vertex2):
        """
        Set the vertices.
        :param vertex1: The first vertex of the object
        :param vertex2: The second vertex of the object
        """
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.calculateMidPointAndSize()

    def setLabel(self, label):
        """
        Set the label of the detected object.

        :param label: The label of the object
        """
        self.label = label

    def getLabel(self):
        """
        Getter for the label.

        :return label: The label of the detected object.
        """
        return self.label
    
    def getVertex1(self):
        """
        Getter for the vertex1.

        :return vertex1: The first vertex of the detected object.
        """
        return self.vertex1
    
    def getVertex2(self):
        """
        Getter for the vertex2.

        :return vertex2: The second vertex of the detected object.
        """
        return self.vertex2

    def calculateMidPointAndSize(self):
        """
        Calculate the mid point and the size of the bounding box that surrounds the object.
        The value of the size is calculated by computing the euclidean distance between vertex1 and vertex2.
        The middle point is calculated by computing the mean of x values and y values of 2 vertices.
        """
        mid_0 = (self.vertex1[0] + self.vertex2[0]) / 2
        mid_1 = (self.vertex1[1] + self.vertex2[1]) / 2
        self.midpoint = [mid_0, mid_1]
        # calculate the Euclidean distance between 2 vertices
        self.size = math.sqrt(pow((self.vertex2[0] - self.vertex1[0]), 2) + pow((self.vertex2[1] - self.vertex1[1]), 2))

    def getSizeAndMidPoint(self):
        """
        Returns the size and middle point of the bounding box.

        :return size:     The size of the bounding box.
        :return midpoint: The middle point of the bounding box
        """
        if (self.size != 0 and len(self.midpoint) != 0):
            return self.size, self.midpoint
        self.calculateMidPointAndSize()
        return self.size, self.midpoint

    def getInfoString(self):
        """
        Get the information string.
        The information string contains the label, and vertices.

        :return: Information string  ->  format = '{label}, {vertex1_x}, {vertex1_y}, {vertex2_x}, {vertex2_y}'
        """
        return '{0} ({1}, {2}) ({3}, {4})'.format(self.label, self.vertex1[0], self.vertex1[1], self.vertex2[0], self.vertex2[1])


def compareObjectLists(list1: list = [], list2: list = []) -> bool:
    """
    Compare the 2 lists of DetectedObject instances.

    :param list1: the first list of DetectedObject type instances.
    :param list2: the other list of DetectedObject type instances.
    :return: (bool) - If there are some differences between 2 lists, then returns True. Otherwise, returns False.
    """
    if (len(list2) is 0):
        return False
    if (len(list1) is 0):
        return True


    checker = True

    # use nested for loop to compare object lists
    for obj1 in list1:
        label1 = obj1.getLabel()
        checker2 = False

        for obj2 in list2:
            label2 = obj2.getLabel()
            if (compareDetectedObjects(obj1, obj2)):
                # compare the label to check if they are same object
                if (label1 == label2):
                    checker2 = True
                break

        checker = checker and checker2
    
    return (not checker)


def compareVertices(vertex1: tuple, vertex2: tuple):
    return (vertex1[0] == vertex2[0] and vertex1[1] == vertex2[1])


def compareDetectedObjects(obj1, obj2):
    """
    Compare the middle point and size of 2 objects.

    :param obj1: The first object
    :param obj2: The second object
    :return: Bool - Returns True if the 2 given objects are in the same bounding box. Otherwise, returns False.
    """
    # get size and middle point of each object
    size1, midpoint1 = obj1.getSizeAndMidPoint()
    size2, midpoint2 = obj2.getSizeAndMidPoint()

    sizeLimit = 5

    # compare the middle points
    if (compareVertices(midpoint1, midpoint2)):
        if (size1 == size2):
            return True
        else:
            bigger = size1
            smaller = size2
            
            if size2 >= size1:
                bigger = size2
                smaller = size1
            if (bigger - sizeLimit) < smaller:
                return True

    return False


def mergeDetectedObjectLists(list1, list2):
    """
    Merge 2 list of detected objects, and return the result.
    :param list1: The list that contains the detected objects that are detected by the general YOLO
    :param list2: The list that contains the detected objects that are detected by the YOLO model that is retrained with the domain dataset.
    """
    newList = []
    newList.extend(list1)

    for obj1 in list1:
        for obj2 in list2:
            sameBoundingBox = compareDetectedObjects(obj1, obj2)
            if (not sameBoundingBox):
                newList.append(obj2)

    return newList
