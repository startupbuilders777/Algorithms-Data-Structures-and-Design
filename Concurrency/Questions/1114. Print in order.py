
'''

1114. Print in Order

Share
Suppose we have a class:

public class Foo {
  public void first() { print("first"); }
  public void second() { print("second"); }
  public void third() { print("third"); }
}

The same instance of Foo will be passed to three different threads. 
Thread A will call first(), thread B will call second(), and thread 
C will call third(). Design a mechanism and modify the program to 
ensure that second() is executed after first(), 
and third() is executed after second().

 

Example 1:

Input: [1,2,3]
Output: "firstsecondthird"
Explanation: There are three threads being fired asynchronously. 
The input [1,2,3] means thread A calls first(), thread B calls second(), 
and thread C calls third(). "firstsecondthird" is the correct output.
Example 2:

Input: [1,3,2]
Output: "firstsecondthird"
Explanation: The input [1,3,2] means thread A calls first(), 
thread B calls third(), and thread C calls second(). 
"firstsecondthird" is the correct output.
 

Note:

We do not know how the threads will be scheduled in the operating 
system, even though the numbers in the input seems to imply the 
ordering. The input format you see is mainly to ensure 
our tests' comprehensiveness.

'''
'''
Raise two barriers. Both wait for two threads to reach them.

First thread can print before reaching the first barrier. 
Second thread can print before reaching the second barrier. 
Third thread can print after the second barrier.
'''

from threading import Barrier

class Foo:
    def __init__(self):
        self.first_barrier = Barrier(2)
        self.second_barrier = Barrier(2)
            
    def first(self, printFirst):
        printFirst()
        self.first_barrier.wait()
        
    def second(self, printSecond):
        self.first_barrier.wait()
        printSecond()
        self.second_barrier.wait()
            
    def third(self, printThird):
        self.second_barrier.wait()
        printThird()

'''
Start with two locked locks. 
First thread unlocks the first lock that the second thread 
is waiting on. Second thread unlocks the second 
lock that the third thread is waiting on.
'''
from threading import Lock

class Foo:
    def __init__(self):
        self.locks = (Lock(),Lock())
        self.locks[0].acquire()
        self.locks[1].acquire()
        
    def first(self, printFirst):
        printFirst()
        self.locks[0].release()
        
    def second(self, printSecond):
        with self.locks[0]:
            printSecond()
            self.locks[1].release()
            
            
    def third(self, printThird):
        with self.locks[1]:
            printThird()

'''
Set events from first and second threads when they are done. 
Have the second thread wait for first one to set its event. 
Have the third thread wait on the second thread to raise its event.
'''
from threading import Event

class Foo:
    def __init__(self):
        self.done = (Event(),Event())
        
    def first(self, printFirst):
        printFirst()
        self.done[0].set()
        
    def second(self, printSecond):
        self.done[0].wait()
        printSecond()
        self.done[1].set()
            
    def third(self, printThird):
        self.done[1].wait()
        printThird()
'''

Start with two closed gates represented by 0-value semaphores. 
Second and third thread are waiting behind these gates. 
When the first thread prints, it opens the gate for the 
second thread. When the second thread prints, it opens the gate for the third thread.
'''

from threading import Semaphore

class Foo:
    def __init__(self):
        self.gates = (Semaphore(0),Semaphore(0))
        
    def first(self, printFirst):
        printFirst()
        self.gates[0].release()
        
    def second(self, printSecond):
        with self.gates[0]:
            printSecond()
            self.gates[1].release()
            
    def third(self, printThird):
        with self.gates[1]:
            printThird()


'''
Have all three threads attempt to acquire an 
RLock via Condition. The first thread can always
acquire a lock, while the other two have to wait for the order 
to be set to the right value. First thread sets the order after 
printing which signals for the second thread to run. 
Second thread does the same for the third.
'''


from threading import Condition

class Foo:
    def __init__(self):
        self.exec_condition = Condition()
        self.order = 0
        self.first_finish = lambda: self.order == 1
        self.second_finish = lambda: self.order == 2

    def first(self, printFirst):
        with self.exec_condition:
            printFirst()
            self.order = 1
            self.exec_condition.notify(2)

    def second(self, printSecond):
        with self.exec_condition:
            self.exec_condition.wait_for(self.first_finish)
            printSecond()
            self.order = 2
            self.exec_condition.notify()

    def third(self, printThird):
        with self.exec_condition:
            self.exec_condition.wait_for(self.second_finish)
            printThird()