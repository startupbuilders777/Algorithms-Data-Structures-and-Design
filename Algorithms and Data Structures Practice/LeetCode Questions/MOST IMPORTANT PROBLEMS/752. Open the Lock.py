'''
752. Open the Lock
Medium

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, 
the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the 
 minimum total number of turns required to open the lock, or -1 if it is impossible.

Example 1:
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".
Example 2:
Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation:
We can turn the last wheel in reverse to move from "0000" -> "0009".
Example 3:
Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
Explanation:
We can't reach the target without getting stuck.
Example 4:
Input: deadends = ["0000"], target = "8888"
Output: -1
Note:
The length of deadends will be in the range [1, 500].
target will not be in the list deadends.
Every string in deadends and the string target will be a string of 4 digits from the 10,000 possibilities '0000' to '9999'.
'''

# SOLVED WITH BIDIRECTIONAL BFS!


class Solution:
    from collections import deque
    def openLock(self, deadends: List[str], target: str) -> int:
        # so we bfs, every possibility? 
        # nah double bfs? 
        # if we see a deadlock, we do not add its kids!
        # DOUBLE BFS!
        
        qA = deque([(0,0,0,0)])
        targetTuple = tuple(map(lambda x: int(x), tuple(target)))
        
        qB = deque([targetTuple])
        
        distA = {}
        distB = {}
        
        distA[(0,0,0,0)] = 0
        distB[targetTuple] = 0
        
        deadendSet = set(list(map(lambda deadend: tuple(map(lambda x: int(x), deadend)), list(deadends))))
        
        if targetTuple in deadendSet or (0,0,0,0) in deadendSet:
            return -1
        # DOUBLE BFS
        
        def move(queue, distA, distB):
            node = queue.pop()
            kid1 = None
            kid2 = None
            
            for idx, i in enumerate(node):
                if i == 9:
                    kid1 = node[:idx] + (0,) + node[idx+1:]
                    kid2 = node[:idx] + (8,) + node[idx+1:]
                elif i == 0:
                    kid1 =  node[:idx] + (1,) + node[idx+1:]
                    kid2 = node[:idx] + (9,) + node[idx+1:]
                else: 
                    kid1 = node[:idx] +  (i - 1,) + node[idx+1:]
                    kid2 = node[:idx] + (i + 1,) + node[idx+1:]
                

                for kid in [kid1, kid2]:
                    if kid in deadendSet:
                        continue
                       
                    elif kid in distA:
                        continue
                    else:
                        distA[kid] = distA[node] + 1

                        if kid in distB:
                            return kid
                        else:
                            queue.appendleft(kid)
            return None

        # STOP WHEN EITHER OF THEM REACHES THE END! 
        # THAT MEANS THE OTHER QUEUE CANNOT REACH EITHER 
        while qA and qB:    
            resultA = move(qA, distA, distB)
            resultB = move(qB, distB, distA)
            
            if resultA:
                return distA[resultA] + distB[resultA]
            
            if resultB:
                return distA[resultB] + distB[resultB]
            
        return -1


# FASTEST SOLUTION WIERD SOLUTION:

class Solution:
    def openLock(self, deadends, target):
        def dist(code):
            return sum(min(int(c), 10-int(c)) for c in code)
		
        def neighbors(code):
            for i in range(4):
                pre = code[:i]
                x = int(code[i])
                sur = code[i:]
                yield code[:i] + str((x - 1) % 10) + code[i + 1:]
                yield code[:i] + str((x + 1) % 10) + code[i + 1:]
		
        dead = set(deadends)
        if '0000' in dead or target in dead: return -1
        last_moves = set(neighbors(target)) - dead
        
        
        if not last_moves: return -1
        ans = dist(target)
        for code in last_moves:
            if dist(code) < ans: return ans
        return ans + 2
