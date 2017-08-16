'''Can be implemented with adjacency list, adjacency matrix, or OOP Style '''
'''OOP Style:'''

class Graph():
    def __init__(self):
        self.nodes = []
        self.numberOfNodes = 0
        self.numberOfEdges = 0
        self.isPlanar = False

    def addNode(self, node):
        self.nodes.append(node)

    def addBidirectionalEdge(self, nodeA, nodeB):

    def addUnidirectionalEdge(self, nodeA, nodeB):

    
    def information(self):
        print("Number of Nodes is " + str(self.numberOfNodes))
        print("Number of Edges is " + str(self.numberOfEdges))


class Node():
    '''Graph Node'''

    def __init__(self, label):
        '''Initialize Node'''
        self.label = label
        self.neighbours = []
        self.degree = 0

    def addNeighbour(self, newNode):
        self.neighbours.append(newNode)
        self.degree += 1

    def removeNeighbour(self, otherNode):
        self.neighbours.remove(otherNode)
        self.degree -= 1


