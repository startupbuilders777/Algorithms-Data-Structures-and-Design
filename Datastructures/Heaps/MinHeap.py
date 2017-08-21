class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None
        self.size = 1

    def __init__(self, left, right, value):
        self.left = left
        self.right = right
        self.size = self.left.size + self.right.size + 1
        self.value = value



class minHeap():
    def __init__(self):
        self.root = None
        self.availableNode

    def getMinimum(self):
        if(self.root is not None):
            return self.root.value
        else:
            return None

    def insert(self, value):
        def insertRecursive(currNode, ):

        '''Bubble up algo -> Place in the first available location then bubble up'''
        if():
            self.root,l,

