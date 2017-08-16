import math

class Point():
    '''2D Point'''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toStr(self):
        return "(" + str(self.x) + " , " + str(self.y) + ")"

def length(pointA, pointB):
    return math.sqrt(math.pow((pointB.y - pointA.y), 2) + math.pow((pointB.x - pointA.x), 2))