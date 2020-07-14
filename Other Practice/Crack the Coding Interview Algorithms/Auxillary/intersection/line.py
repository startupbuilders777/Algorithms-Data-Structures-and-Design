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
            self.slope = 9999999 #None is infinite slopes
        self.intercept = pointB.y - self.slope*pointB.x

    def checkPointWithinLineSegmentIfPointIsOnLine(self, point):
        print("IT IS WITHIN THIS LINE")
        if(self.slope >= 0): #Positive Slope
            return (self.pointA.x <= point.x and point.x <= self.pointB.x) and (self.pointA.y <= point.y and point.y <= self.pointB.y)
        else: #Negative Slope
            return (self.pointA.x <= point.x and point.x <= self.pointB.x) and (
            self.pointA.y >= point.y and point.y >= self.pointB.y)
    def evaluate(self, x):
        return x*self.slope + self.intercept