'''
Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks.Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.

Example 1:
Input: tasks = ['A','A','A','B','B','B'], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
Note:
The number of tasks is in the range [1, 10000].
The integer n is in the range [0, 100].


'''


class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int


        """
        tasksDict = {}
        for i in tasks:
            if tasksDict.get(i) is None:
                tasksDict[i] = 1
            else:
                tasksDict[i] += 1

        def doTask(tasksDict, onHold, n):  # Should be using the set operation Except for this.
            action = None
            # Greedily pick the action that isnt on HOLD and has the highest frequency in the tasksDict
            # Sort keys from greatest to least and use that

            keylist = sorted(tasksDict.iterkeys(), key=lambda k: tasksDict[k], reverse=True)
            # print(tasksDict)
            for i in keylist:
                if (onHold.get(i) is None):
                    action = i
                    tasksDict[i] -= 1
                    if (tasksDict[i] == 0):
                        tasksDict.pop(i)
                    break

            for i in onHold.keys():
                onHold[i] -= 1
                if (onHold[i] == 0):
                    onHold.pop(i)

            if action is not None and n >= 1:
                onHold[action] = n

            return action

        onHold = {}

        count = 0
        while True:  # True when tasksDict is not empty
            action = doTask(tasksDict, onHold, n)
            # print("DO THIS: " + str(action))
            count += 1
            if not tasksDict:
                break
        return count
