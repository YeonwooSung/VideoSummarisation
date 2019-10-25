from __future__ import division


class DetectedObject():
    def __init__(self):
        super().__init__()
        self.vertex1 = []
        self.vertex2 = []
        self.label = ''

    def setVertices(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2

    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label
    
    def getVertex1(self):
        return self.vertex1
    
    def getVertex2(self):
        return self.vertex2
    
    def getInfoString(self):
        return '{0} ({1}, {2}) ({3}, {4})'.format(self.label, self.vertex1[0], self.vertex1[1], self.vertex2[0], self.vertex2[1])


def compareObjectLists(list1: list = [], list2: list = []) -> bool:
    """
    Compare the 2 lists of DetectedObject instances.

    :param list1: the first list of DetectedObject type instances.
    :param list2: the other list of DetectedObject type instances.
    :return: (bool)
    """
    if (list2 is []):
        return False

    checker = True

    for obj in list1:
        label = obj.getLabel()
        vertex1 = obj.getVertex1()
        vertex2 = obj.getVertex2()

        checker2 = False

        for o in list2:
            label2 = o.getLabel()
            v1 = o.getVertex1()
            v2 = o.getVertex2()

            if (compareVertices(v1, vertex1) and compareVertices(v2, vertex2)):
                checker2 = True
                break
        
        checker = checker and checker2
    
    return checker


def compareVertices(vertex1: tuple, vertex2: tuple):
    if (vertex1 == vertex2):
        return True
    
    return False
