class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.value = None
        self.data = None
        self.size = 1

    def __init__(self, value, data):
        self.value = value
        self.data = data
        self.parent = None
        self.left = None
        self.right = None
        self.size = 1

class minHeap():
    def __init__(self):
        self.root = None

    def getMinimum(self):
        if(self.root is not None):
            return (self.root.value, self.root.data)
        else:
            return None

    def deleteMin(self):
        print("REMOVE")
        def swap(node1, node2):
            tempValue = node1.value
            tempData = node1.data
            node1.value = node2.value
            node1.data = node2.data
            node2.value = tempValue
            node2.data = tempData
        def getNode(node):
            if(node.left is None and node.right is None):
                print("OOO")
                result = (node.value, node.data)
                if(node.parent.right == node):
                    node.parent.right = None
                elif(node.parent.left == node):
                    node.parent.left = None
                return result
            if(node.right is not None and node.left is not Node and node.right.size >= node.left.size):
                node.size -= 1
                return getNode(node.right)
            elif(node.right is not None and node.left is not Node and node.right.size < node.left.size):
                node.size -= 1
                return getNode(node.left)
            elif(node.right is not None):
                print("AHH")
                node.size -= 1
                return getNode(node.right)
            elif(node.left is not None):
                node.size -= 1
                return getNode(node.left)
        def bubbleDown(node):
            if(node.left == None and node.right == None):
                return
            elif(node.left is not None and node.right is not None):
                if(node.left.size <= node.right.size): #Try inserting left, then try right
                    if (node.left.value < node.value):
                        swap(node.left, node)
                    elif (node.right.value < node.value):
                        swap(node.right, node)
                    else:
                        return None
                else:
                    if (node.right.value < node.value):
                        swap(node.right, node)
                    elif(node.left.value < node.value):
                        swap(node.left, node)
                    else:
                        return None
            elif(node.left is not None and node.left.value < node.value):
                swap(node.left, node)
            elif(node.right is not None and node.right.value < node.value):
                swap(node.right, node)
            else:
                return
            bubbleDown(node)
        if(self.root == None):
            return Exception
        elif(self.root.size == 1 ):
            self.root = None
        else:
            val, data = getNode(self.root)
            self.root.value = val
            self.root.data = data
            bubbleDown(self.root)




    def insert(self, value, data):
        '''Bubble up algo -> Place in the first available location then bubble up'''
        def insertRecursive(currNode, val, dat):
            if currNode.left is None:
                currNode.left = Node(val, data=dat)
                currNode.left.parent = currNode
                bubbleUp(currNode.left)
            elif currNode.right is None:
                currNode.right = Node(val, dat)
                currNode.right.parent = currNode
                bubbleUp(currNode.right)
            elif(currNode.left.size <= currNode.right.size):
                insertRecursive(currNode.left, val, dat)
            else:
                insertRecursive(currNode.right, val, dat)
            currNode.size += 1

        def bubbleUp(currNode):
            if(currNode.parent is None):
                return
            elif currNode.parent.value >= currNode.value:
                tempValue = currNode.parent.value
                tempData = currNode.parent.data
                currNode.parent.value = currNode.value
                currNode.parent.data = currNode.data
                currNode.value = tempValue
                currNode.data = tempData
                bubbleUp(currNode.parent)
            else:
                return


        if(self.root is None):
            self.root = Node(value, data)
        else:
            insertRecursive(self.root, val=value, dat = data)


heap = minHeap()
heap.insert(2,3)
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.insert(2,4)
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.insert(1,5)
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.insert(0,3)
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.insert(6,4)
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
print("###############################################")
heap.deleteMin()
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.deleteMin()
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.deleteMin()
print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))
heap.deleteMin()

print(heap.getMinimum())
if(heap.root.left is not None):
    print("left: " + str(heap.root.left.size))
if(heap.root.right is not None):
    print("right: " + str(heap.root.right.size))

heap.deleteMin()
print(heap.getMinimum())
'''
heap.deleteMin()
'''