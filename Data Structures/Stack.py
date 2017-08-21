
''''
Very useful for recursive applications to push temporary data on a stack as you recurse, but then remove them as you backtrack

Stack can also be used to implement a recursive algo iteratively!!!

Stacks can be implemented with a linked list or an array =>
    You may need to resize array but at least you dont have to worry about ptrs
'''

class StackNode():
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class Stack():
    def __init__(self):
        self.topNode = None
        self.count = 0

    def push(self, data):
        self.count += 1
        newNode = StackNode(data)
        if(self.topNode is None):
            self.topNode = newNode
        else:
            self.topNode.next = newNode
            newNode.prev = self.topNode
            self.topNode = newNode

    def top(self):
        if self.topNode is None:
            return None
        else:
            return self.topNode.data

    def size(self):
        return self.count

    def pop(self):
        if self.topNode is None:
            return ReferenceError
        else:
            self.count -= 1
            prev = self.topNode.prev
            self.topNode = prev

    def empty(self):
        return self.topNode is None
'''

aStack = Stack()

aStack.push(2)
print(aStack.top())
print(aStack.size())
aStack.push(3)
print(aStack.top())
print(aStack.size())
aStack.push(6)
print(aStack.top())
print(aStack.size())
aStack.pop()
print(aStack.top())
print(aStack.size())
aStack.push(69)
print(aStack.top())
print(aStack.size())
aStack.pop()
print(aStack.top())
print(aStack.size())


print("")
'''

def fibIterative(n):
    stack = Stack()
    stack.push(n)       #Stack contains the question initially which is iteratively solved
    sum = 0

    while not stack.empty(): #We return when no more stuff in stack
        n = stack.top()
        if(n <= 10):
            print(n)
        stack.pop()     #Dont need the current stack frame anymore, we are replacing it with 2 stack frames pushed below
        if(n == 0):     #Base cases like usual
            sum += 0
            continue    # A sort of way to just return (close a stack frame when it isnt needed anymore)
        if(n == 1):
            sum += 1
            continue

        stack.push(n-2)     #Do recursive calls by pushing onto the stack problems that will approach the base case
        stack.push(n-1)

    return sum          #Solution after iteration

#print(fibIterative(12))