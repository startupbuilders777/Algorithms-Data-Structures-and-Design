from Stack import Stack

''' Implement a queue with 2 stacks '''

class Queue():
    '''
    Invarient :
    Front will always contain elements if the back contains elements and peeking at front is always O(1)
    '''
    def __init__(self):
        self.front = Stack()
        self.back = Stack()

    def size(self):
        return self.front.size() + self.back.size()

    def add(self, item):
        self.back.push(item)
        if (self.front.empty()):
            self.transfer()

    def front(self):
        if(self.empty()):
            return None
        else:
            print("fook")
    #        return self.front.top()

    def empty(self):
        return self.front.empty()

    def remove(self):
        if(self.empty()):
            return Exception
        self.front.pop()
        if(self.front.empty()):
            self.transfer()

    def transfer(self):
        while not self.back.empty():
            self.front.push(self.back.top())
            self.back.pop()

aQueue = Queue()
print(aQueue)
aQueue.add(2)
print(aQueue.front())
print(aQueue.size())
aQueue.add(3)
print(aQueue.front())
print(aQueue.size())
aQueue.add(6)
print(aQueue.front())
print(aQueue.size())
aQueue.remove()
print(aQueue.front())
print(aQueue.size())
aQueue.add(69)
print(aQueue.front())
print(aQueue.size())
aQueue.remove()
print(aQueue.front())
print(aQueue.size())