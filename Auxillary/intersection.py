'''Given 2 straight line segments (represented as a start point and end point)
compute the point of intersection if any
'''

import math

class Point():
    '''2D Point'''
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line():
    '''Line from pointA to pointB'''
    def __init__(self, pointA, pointB):
        self.pointA = pointA
        self.pointB = pointB
        if(pointB.x - pointA.x != 0):
            self.slope = (pointB.y - pointA.y) / (pointB.x - pointA.x)
        else:
            self.slope = 999999999
        self.intercept = pointB.y - self.slope*pointB.x



    def evaluate(self, x):
        return x*self.slope + self.intercept

def length(pointA, pointB):
    return math.sqrt(math.pow((pointB.y - pointA.y), 2) + math.pow((pointB.x - pointA.x), 2))

line1 = Line(Point(1,1), Point(1, 10))
line2 = Line(Point(-1,4), Point(12, 4))

print(line1.slope)
print(line2.slope)

'''
y1 = m1x + b1
(y1 - b1)/m1 = x


y2 = m2x + b2
(y2-b2)/m2 = x


m1x + b1 = m2x + b2
(m1 - m2)x = b2 - b1
x = (b2 - b1)/(m1 - m2)
Then to get y, evaluate it.
'''

def intersection(segment1, segment2):
    if(segment1.slope != segment2.slope):
        x = (segment2.intercept - segment1.intercept)/(segment1.slope - segment2.slope)
        y = segment1.evaluate(x)
        print(y)
        #Check if its within the line segment
        point = Point(x, y)
        if not((length(point, segment1.pointA) + length(point, segment1.pointB)) == length(segment1.pointA, segment1.pointB) and
            (length(point, segment2.pointA) + length(point, segment2.pointB)) == length(segment2.pointA, segment2.pointB)):
            print("Point is not within the line segments, so NOPE")
            return(None, None)

        return (x, y)
    else:
        print("NO INTERSECTION")
        return (None, None) # <- Return a tuple

(x, y) = intersection(line1, line2)

print("The intersection is " + "(" + str(x) + " , " + str(y) + ")")