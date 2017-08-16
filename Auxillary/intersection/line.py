from point import Point

class Line():
    '''Line from pointA to pointB
        leftmost point is pointA, and rightmost point is pointB
    '''
    def __init__(self, pointA, pointB):
        if(pointA.x <= pointB.x):
            self.pointA = pointA
            self.pointB = pointB
        else:
            self.pointA = pointB
            self.pointB = pointA
        if(pointB.x - pointA.x != 0):
            self.slope = (pointB.y - pointA.y) / (pointB.x - pointA.x)
        else:
            self.slope = 999999999 #Do something better than this
        self.intercept = pointB.y - self.slope*pointB.x

    def checkPointWithinLineSegmentIfPointIsOnLine(self, point):
        return (self.pointA.x <= point.x and point.x <= self.pointB.x) and (self.pointA.y <= point.x and point.x <= self.pointB.x)

    def evaluate(self, x):
        return x*self.slope + self.intercept