'''

Implement with heaps: -> A complete binary tree that has the heap property

min-PQ -> p value of all children is larger number than that of the parent

'''

from Heaps import MinHeap

Heap = MinHeap.minHeap

class PriorityQueue():
    def __init__(self):
        self.heap = Heap()

    def insert(self, x, p):
        '''insert x with priority p'''
        self.heap.insert(p, x)

    def deleteMin(self):
        self.heap.deleteMin()

    def top(self):
        self.heap.getMinimum()

