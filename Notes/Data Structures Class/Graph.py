'''Can be implemented with adjacency list, adjacency matrix, or OOP Style '''
'''OOP Style:'''
'''PLEASE FINISH THE GRAPH CLASS IMPLEMENATION THANKS'''

class Graph():
    def __init__(self, graphName):
        self.graphName = graphName
        self.nodes = []
        self.edges = []
        self.numberOfNodes = 0
        self.numberOfEdges = 0
        self.isPlanar = False

    def addNode(self, node):
        self.nodes.append(node)
        self.numberOfNodes += 1

    def addBidirectionalEdge(self, nodeA, nodeB, length = 1):
        self.edges.append(Edge(nodeStart=nodeA, nodeEnd=nodeB, length=length))
        self.edges.append(Edge(nodeStart=nodeB, nodeEnd=nodeA, length=length))
        self.numberOfEdges += 1

    def addUnidirectionalEdge(self, nodeA, nodeB, length = 1):
        self.edges.append(Edge(nodeStart=nodeA, nodeEnd=nodeB, length=length))

 #   def removeBidirectionalEdge(self, nodeA, nodeB):
 #       self.edges.append(Edge(nodeStart=nodeA, nodeEnd=nodeB, length=length))

 #   def removeUnidirectionalEdge(self, nodeA, nodeB):

    def information(self):
        print("Number of Nodes is " + str(self.numberOfNodes))
        print("Number of Edges is " + str(self.numberOfEdges))

    def displayStructure(self):
        print("The graph " + self.graphName + " has the following structure")

class Edge():
    def __init__(self, nodeStart, nodeEnd, length = 1):
        nodeStart.addNeighbour(nodeEnd)
        self.nodeStart = nodeStart
        self.nodeEnd = nodeEnd
        self.length = length

    def delete(self, nodeStart, nodeEnd):
        nodeStart.removeNeighbour(nodeEnd)
        self.nodeStart = None
        self.nodeEnd = None


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
