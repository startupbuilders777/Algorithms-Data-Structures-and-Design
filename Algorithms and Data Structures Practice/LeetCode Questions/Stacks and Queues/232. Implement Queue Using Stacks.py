'''
232. Implement Queue using Stacks
Easy

1052

142

Add to List

Share
Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.
Example:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false
Notes:

You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).
Accepted
214,156
Submissions
437,030

'''



class MyQueue:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stackA = []
        self.stackB = []
        
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.stackA.append(x)
        
    def dump(self):
        while self.stackA:
            ele = self.stackA.pop()
            self.stackB.append(ele)
        
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return False
        if len(self.stackB) == 0:
            # dump!
            self.dump()
        
        ele = self.stackB.pop()
        return ele
        
    def peek(self) -> int:
        """
        Get the front element.
        """
        
        if len(self.stackB) == 0:
            self.dump()
        
        return self.stackB[-1]
        
    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return (len(self.stackA) + len(self.stackB)) == 0



'''
AMORTIZED O(1) TIME OPERATIONS!

The loop in peek does the moving from input to output stack. 
Each element only ever gets moved like that once, though, and only after we already spent 
time pushing it, so the overall amortized cost for each operation is O(1).

class Queue {
    stack<int> input, output;
public:

    void push(int x) {
        input.push(x);
    }

    void pop(void) {
        peek();
        output.pop();
    }

    int peek(void) {
        if (output.empty())
            while (input.size())
                output.push(input.top()), input.pop();
        return output.top();
    }

    bool empty(void) {
        return input.empty() && output.empty();
    }
};

'''


