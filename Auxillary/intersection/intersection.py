'''Given 2 straight line segments (represented as a start point and end point)
compute the point of intersection (or point of line segment intersection) if any
'''

import math
from point import Point, length
from line import Line

'''
y1 = m1x + b1
(y1 - b1)/m1 = x


y2 = m2x + b2
(y2-b2)/m2 = x


m1x + b1 = m2x + b2
(m1 - m2)x = b2 - b1
x = (b2 - b1)/(m1 - m2)
Then to get y, evaluate it.

They can intersect if they are the same sloped line and they are on each other 
or they are dif lines and intersect

'''

def intersection(segment1, segment2):
    if(segment1.slope != segment2.slope):
        print("They are not parallel case")
        x = (segment2.intercept - segment1.intercept)/(segment1.slope - segment2.slope)
        y = segment1.evaluate(x)
        #print(y)
        #Check if its within the line segment
        point = Point(x, y)
        print("The possible intersection is " + "(" + str(x) + " , " + str(y) + ")")
        if not(segment1.checkPointWithinLineSegmentIfPointIsOnLine(point)
             and segment2.checkPointWithinLineSegmentIfPointIsOnLine(point)):
            print("Point is not within the line segments, so NOPE")
            return(None, None)
        else:
            return (x, y) #<- Tuple
    else:
        print("They are parallel case")
        if segment1.intercept == segment2.intercept:
            if(segment2.checkPointWithinLineSegmentIfPointIsOnLine(segment1.pointB)):
                print("Intersects in a line")
                return (segment2.pointA, segment1.pointB)
            elif(segment2.checkPointWithinLineSegmentIfPointIsOnLine(segment1.pointA)):
                print("Intersects in a line")
                return (segment1.pointA, segment2.pointB)
            else:
                return(None, None)
        else:
            return (None, None) # <- Return a tuple

#Same line case Parallel
line1 = Line(Point(1,1), Point(2, 2))
line2 = Line(Point(-4,-4), Point(1, 1))

print(line1.slope)
print(line2.slope)

(x, y) = intersection(line1, line2)

if(isinstance(x, Point)):
    print("The line segment has points " + x.toStr() + " and " + y.toStr())
else:
    print("The intersection is " + "(" + str(x) + " , " + str(y) + ")")

line3 = Line(Point(1,1), Point(2, 2))
line4 = Line(Point(-4,-4), Point(0, 0))

print(line3.slope)
print(line4.slope)

(a, b) = intersection(line3, line4)

if(isinstance(a, Point)):
    print("The line segment has points " + a.toStr() + " and " + b.toStr())
elif(isinstance(a, int)):
    print("The intersection is " + "(" + str(a) + " , " + str(b) + ")")
else:
    print("fook")

'''Intersection Case'''

line3 = Line(Point(-1,1), Point(1, -1))
line4 = Line(Point(1,1), Point(-1, -1))

print(line3.slope)
print(line4.slope)

(a, b) = intersection(line3, line4)

if(isinstance(a, Point)):
    print("The line segment has points " + a.toStr() + " and " + b.toStr())
elif(isinstance(a, int)):
    print("The intersection is " + "(" + str(a) + " , " + str(b) + ")")
else:
    print("fook")

line3 = Line(Point(-1,1), Point(-1.1, 4))
line4 = Line(Point(-2,3), Point(2, 3))

print(line3.slope)
print(line4.slope)

(a, b) = intersection(line3, line4)

if(isinstance(a, Point)):
    print("The line segment has points " + a.toStr() + " and " + b.toStr())
elif(isinstance(a, int) or isinstance(a, float)):
    print("The intersection is " + "(" + str(a) + " , " + str(b) + ")")
else:
    print("fook")