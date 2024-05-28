
For very quick refresher -> Understand the following. 
He goes through 300 leetcode problems: https://nishmathcs.wordpress.com/category/leetcode-questions/

Scrape the following -> short and sweet: https://nishmathcs.wordpress.com/category/data-structures-algorithms/page/1/
Post about all string algorithms and hashing types (KMP, Boyer Moore, etc, Rabin Karp)

COMPETITIVE PROGRAMMING GUIDE TO READ: https://cp-algorithms.com/
TOPCODER COMPETITIVE PROGRAMMING GUIDES -> https://www.topcoder.com/community/competitive-programming/tutorials/

REALLY COOL MEDIUM ARTICLE -> https://medium.com/@karangujar43/best-resources-for-competitive-programming-algorithms-and-data-structures-730cb038e11b


TOPICS TO UNDERSTAND: 
        Segment tree (with lazy propagation)
        Skip lists.
        Interval Tree
        Binary Indexed Tree
        Fast Modulo Multiplication (Exponential Squaring) 90
        Heuristic Algorithms
        KMP string searching
        Manacher’s Algorithm
        Union Find/Disjoint Set
        Trie
        Prime Miller Rabin
        Matrix Recurrence + Fast Modulo Multiplication for counting
        Stable Marriage Problem
        Extended Euclid’s algorithm 25
        Ternary Search
        Fast Fourier Transform for fast polynomial multiplication
        Djikstra’s algorithm, Bellman-ford algorithm, Floyd-Warshall Algorithm
        Prim’s Algorithm, Kruskal’s Algorithm
        RMQ, LCA
        Flow related algorithms, assignment problem, Hungarian algorithm
        Bipartite matching algorithms
        Heavy-light decomposition
        Sweep line algorithm
        Z algorithm
        Convex Hull
        Suffix Arrays 21
        LCP
        Suffix Tree
        Gaussian Elimination
        Numerical Integration/Differentiation
        Line Clipping
        Advanced Maths Ad-Hoc problems
        Aho–Corasick string matching algorithm;
        Calculate nCr % M Lucas’s Theorem 21
        Graph Coloring
        Network Flow
        Sqrt Decomposition
        Heavy Light decomposition in trees
        Inverse Modulo operations
        Pollard Rho Integer Factorization
        Catalan Numbers
        Euclid’s GCD Algorithm
        Extended Euclid’s algorithm
        Binary Search, Ternary Search
        Sieve of Eratosthenes for finding primes
        Fast Fourier Transformation for fast polynomial multiplication
        Graph algorithms - BFS, DFS, finding connected components
        Djikstra’s algorithm, Bellman-ford algorithm, Floyd-Warshall Algorithm
        Prim’s Algorithm, Kruskal’s Algorithm
        RMQ, LCA
        Flow related algorithms, assignment problem, Hungarian algorithm
        Bipartite matching algorithms
        Heavy-light decomposition
        Sweep line algorithm
        Z algorithm


THESE ARE HARMAN'S PERSONAL SET OF PARADIGMS/ INTERVIEW NOTES:
+-117) Learn to simplify your thoughts after finding a soln. You make a bunch of assumptions but you actually need less assumptions to solve the problem.
+      Stop adding so many artificial constraints yourself, and think beyound about your own assumptons to solution!!
+      Find Peak Element 1D 
+        162. Find Peak Element
+        Medium
+        5689
+        3628
+        Add to List
+        Share
+        A peak element is an element that is strictly greater than its neighbors.
+        Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
+        You may imagine that nums[-1] = nums[n] = -∞.
+        You must write an algorithm that runs in O(log n) time.
+        
+        Example 1:
+        Input: nums = [1,2,3,1]
+        Output: 2
+        Explanation: 3 is a peak element and your function should return the index number 2.
+        Example 2:
+        Input: nums = [1,2,1,3,5,6,4]
+        Output: 5
+        Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
+        class Solution:
+            def findPeakElement(self, nums: List[int]) -> int:
+                l = 0 
+                r = len(nums) - 1 
+                # just append -inf to front and end of array!
+                # kills log n behavior! =c
+                # deal with size = 0 array here. 
+                while l < r:
+                    # right bias used here just for fun
+                    mid = l + (r-l + 1)//2
+                    # if mid == 0 && 
+                    leftElement = nums[mid-1] if mid - 1 >= 0 else float("-inf")
+                    rightElement = nums[mid + 1] if mid + 1 < len(nums) else float("-inf")
+                    
+                    if nums[mid] > leftElement and nums[mid] > rightElement:
+                        return mid
+                    
+                    elif leftElement < nums[mid] and nums[mid] < rightElement: 
+                        # ok so its on right side
+                        l = mid 
+                    else:
+                        r = mid - 1
+                # returning either l or r works here. 
+                return r
+        
+
+        Much simpler soln: (you need to take out that BS assumption!) -> otherwise followup problems become much harder
+        
+        public class Solution {
+            public int findPeakElement(int[] nums) {
+                int l = 0, r = nums.length - 1;
+                while (l < r) {
+                    int mid = (l + r) / 2;
+                    if (nums[mid] > nums[mid + 1])
+                        r = mid;
+                    else
+                        l = mid + 1;
+                return l;
+        }
+-116) Identifying when DP is needed and BITMASK DP
+        [* H * * * ]
+        [H * * * * ]
+        [ * * * * H]
+        [ * * H * *]
+        2D matrix, H stands for house, * stands for empty spots and these empty spots can be planted with trees
+        Find the minium number of trees need to be planted and they will be neighbouring to each house, and the neighbouring condition is 8 directions neighbored
+        the answer should be ( t stands for trees)
+        [* H * * * ]
+        [H t * * * ]
+        [ * * * t H]
+        [ * * H * *]
+    Initial idea (Greedy):
+        you can collect the trees all around the houses.
+        Then you can take the tree that covers the max number of houses, 
+        remove that tree, then compute the next tree that covers the max number of houses,
+        and end it.
+    The greedy soln doesnt work for the following case: 
+        * * * H * * *
+        * * * * * * *
+        * * * H * * *
+        H * H * H * H
+        * * * H * * *
+        * * * * * * *
+        * * * H * * *
+        This only takes 4 trees to plant optimally, but with your algorithm it'd take 5. I think that this problem is NP-hard, and that we'll need DP to solve it.
+    -> Ok so DP.
+    
        quick dp soln:
        Tc: O(t2^t) SC O(t2^t) t->total H

        create bipartite graph i->j edge denotes planting tree at i place will satisfy neighbouring to jth cell(house)
        We just need to find minimum set of left side nodes such that all right side nodes are covered.
        It's bitmask DP dp[i][mask] denotes we are at ith left node and set bits in mask denotes set of right nodes available.
        dp[i][mask] = min(dp[i+1][mask], 1 + dp[i+1][mask- mask&curMask]). curMask->set of nodes which can be satisfied by this i.

        Call dp(0, 1<<rightNodes - 1)) and compute top down dp.

        LeftNodes->T, rightNodes->T hence SC: O(T*2^T).


+    At the very start of DP what you want to do is try to write out the recursion and do what you think will happen and 
+    that will pwoer your bottom up! Lets try it.
+    If you solve recursively, you have to take or not take certain tree locations, 
+    and that will update the subset of houses that are covered.
+    Label each house 1 to N, and then you can start taking subsets of 1 to N. 
+    You can also just use a bitmask instead of a set()
+    Like this:
+    def soln(grid):
+        N = len(grid)
+        M = len(grid[0])
+        treeLocations = []
+        M = {}
+        
+        k = 0
+        for i in range(N):
+            for j in range(M):
+                if grid[i][j] == "H":
+                    treeLocations.extend([(i+1, j), (i-1,j)])...ETC!
+                    M[(i,j)] = k
+                    k += 1
+        @lru_cache(None)
+        def recursion(i, setProcessed):
+            (treeX, treeY) = treeLocations[i]
+            if setProcssed == 2^(len(m)) - 1:
+                # alrdy processed all
+                return 0
+            # ok either process take it or not. 
+            # find all houses around me!
+            dirs = [(-1,-1), (-1, 0), (1,1), (1,0), (0,1), (0, -1), (-1, 1), (1, -1)]
+            taken = setProcessed
+            for i,j in dirs:
+                houseLabel = M.get(treeX+i, treeY+j)
+                if houseLabel is not None:
+                    taken |= ( 1 << houseLabel)
+            # ok either take or dont take
+            return min( 1 + recursion(i+1, taken), recursion(i+1, setProcessed) )

+        # Now go over the trees and see which houses covered
+    Recursively, we want to find the min, and solve for every subset -> therefore our DP would be
+    OPT[ith tree processed][set(houses covered)] = Minimum trees required to cover the set of houses. 
+    Choosing or not choosing ith tree will affect all house coverages from 1 to i-1th coverages!
+    To get index of a house use a map that maps (housex, housey) -> idxofhouseinsubset
+    OPT[0][0] = 0
+    
+    OPT[i][x] = 
+        min(OPT[i-1][x], 1 + OPT[i-1][all partial x which when introducing tree i -> results in x]) 
+    You can space optimize out the i parameter given you only need the previous to generate next
+    x goes from 0 to 2^n - 1
+    
+    Total complexity (2^n - 1) * #ofCandidateTrees
+    
+    Maybe forward DP is easier. 
+    # Process tree i, calculate all the x it generates -> set the mins!
+    # Relax it from infinite!
+    -> Exercise: check if this is correct:
+    F = len(treeLocations)
+    prev = [0] # base case
+    curr = [0] # base case
+    prev.extend([float("inf") for i in range(1, 2**N)])
+    curr.extend([float("inf") for i in range(1, 2**N)])
+    
+    dirs = [(-1,-1), (-1, 0), (1,1), (1,0), (0,1), (0, -1), (-1, 1), (1, -1)]
+    for i in range(F):
+        for j in range(2**N - 1):
+            # -> check all the houses you cover and forward update!
+            # all_forward_updates = []
+            # or should we just do 1 forward update for all the houses we covered!
+            new_houses_covered = j
+            for i,j in dirs:
+                houseLabel = M.get(treeX+i, treeY+j)
+                if houseLabel is not None:
+                    new_houses_covered |= ( 1 << houseLabel)
+            curr[new_houses_covered] = min(curr[new_houses_covered], 1 + prev[j])
+        prev = curr
+    return curr[2**N-1]
+    
+

+
+-115) Lazy Deleting VS doubly linked list:
+    
+        Design a wait list for customers at restaurant:
+
+        Add customers to wait list (for example: Bob, party of 4 people)
+        Remove a customer from wait list
+        Given a open table with N seats, remove and return the first customer party of size N
+        Clarifications:
+
+        10 unique table sizes
+        Customer names unique
+        FIFO if two parties have the same number of people
+        Table with N seats must have exactly N people
+        Ideal solution O(1) runtime for all 3 methods
+
+    Soln1 :
+        map {
+
+        tableIdx -> [customer1, customer2]
+        1...10 keys
+        }
+
+         Add to waitlist, -> add him to the end
+        Remove from waitlsit. just mark him removed in some set!
+        Assign table -> check if customer has been removed before assinging, otherwise go to next customer. 
+
+        Guranteed O(1) delete complexity + FIFO
+
+        -> Otherwise, you can use a doubly linked list to remove him quickly!
+
+        My solution is to use 3 maps.
+
+        One map for table(size) and list of customer waiting for it in order. customer_wait_list_map <Table_id, List>
+        One map for table(size) and its availability count. <Table_Id, Count> (Optional if we have multiple same table with same size)
+        One map for customer and it's Node address. <Customer_name, Node>
+        Node is Doubly Linked List.
+        So all 3 could be done in O(1)
+
+
+-114)   Understand how to do graph property type questions, where you have to satisfy a graph shape:
+    
+        Give a undirected graph as a list of edges [(1, 2), (1, 3), (1, 4), (2, 3)], check if the graph forms a grid?
+        For examples,
+
+        True Case: [(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (5, 6)]
+
+        Because we can form grid,
+
+        1 2 3
+        4 5 6
+
+        False Case: [(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5)]
+
+        Because we cannot form grid,
+
+        1 2 3
+        4 5
+
+        there should be 4 nodes with degree 2
+        Check weather all the vertices with degree less than or equal to 4
+        choose one corner and run a bfs until you get 2 more corners, so now you have your row size and column size.
+        from the corner other than the one found in step 3 bfs again and check if m,n are satisfied here as well
+        now you should have 4 two degree vertices, 2*(m+n)-8 three degree vertices, nm-2(m+n)+4 four degree vertices.
+
+
+-113) Insert, Delete, GetRandom O(1)
+
+    Implement the RandomizedSet class:
+
+    RandomizedSet() Initializes the RandomizedSet object.
+    bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
+    bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
+    int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this 
      method is called). Each element must have the same probability of being returned.
+    You must implement the functions of the class such that each function works in average O(1) time complexity.
+
+    class RandomizedSet:
+        def __init__(self):
+            self.arr = []
+            self.m = {}
+        def insert(self, val: int) -> bool:
+            if(val in self.m):
+                return False
+            self.arr.append(val)
+            idx = len(self.arr) - 1
+            self.m[val] = idx
+            return True
+        def remove(self, val: int) -> bool:
+            # -> get idx from val  
+            
+            if(self.m.get(val) is None):
+                return False
+            
+            idx = self.m[val] 
+        
+            lastIdx = len(self.arr) - 1
+            lastIdxVal = self.arr[lastIdx]
+            
+            self.m[lastIdxVal] = idx
+            self.arr[idx], self.arr[lastIdx] = self.arr[lastIdx], self.arr[idx]
+            
+            del self.m[val]     
+            self.arr.pop()
+            
+            
+            # what if removing the last one? edge case
+            return True
+        
+        def getRandom(self) -> int:
+            r = random.randint(0, len(self.arr)-1)
+            return self.arr[r]
+
+


+
+-112) Multisource BFS:
+        For multisource BFS you will have to use visited set twice if yu fck it up.
+        Make sure you process every node once, or if your parent is processing the child, 
+        multiple parents arent processing the same child!!!
+
+        Look at below example for correctness!
+
+        '''
+        663 · Walls and Gates
+
+        You are given a m x n 2D grid initialized with these three possible values.
+
+        -1 - A wall or an obstacle.
+        0 - A gate.
+        INF - Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
+        Fill each empty room with the distance to its nearest gate. If it is impossible to reach a Gate, that room should remain filled with INF
+
+        Explanation:
+        the 2D grid is:
+        INF  -1  0  INF
+        INF INF INF  -1
+        INF  -1 INF  -1
+        0  -1 INF INF
+        the answer is:
+        3  -1   0   1
+        2   2   1  -1
+        1  -1   2  -1
+        0  -1   3   4
+        Example2
+
+        Input:
+        [[0,-1],[2147483647,2147483647]]
+        Output:
+        [[0,-1],[1,2]]
+        '''
+
+        from collections import deque
+
+        class Solution:
+            """
+            @param rooms: m x n 2D grid
+            @return: nothing
+            """
+            def walls_and_gates(self, rooms: List[List[int]]):
+                N = len(rooms)
+                M = len(rooms[0])
+
+                d = deque()
+                visited = set()
+                dist = {}
+                for i in range(N):
+                    for j in range(M):
+                        if rooms[i][j] == 0:
+                            d.append((i,j))
+                            dist[(i,j)] = 0
+                            # rooms[i][j] = 0
+            
+
+                directions = [(0,1), (0,-1), (1,0), (-1,0)]
+
+                # THIS PROBLEM ILLUSTRATES WHEN NODES SHOULD BE ADDED TO VISITED!
+                # to reduce any duplication that can occur in updates.
+                # nodes should be processed once and know that defn!
+
+                while len(d) > 0:
+                    r, c = d.popleft()
+                    # Both visiteds are required in multisource BFS 
+                    # The below visited can be removed if you add the  gate nodes at the very start 
+                    # to the visited set
+                    # BE CAREFUL WHEN YOU ARE DOING PARENT TO CHILD UPDATE aka parent updates child
+                    # that means child was processed! so 2 parents shouldnt update same child!!
+                    visited.add((r,c) )
+
+                    for (i,j) in directions:
+                        if(r + i < N and r + i >=0 and c + j < M and c + j >= 0 and (r+i, c+j) not in visited
+                            and rooms[r+i][c+j] !=-1):
+                            
+                            d.append((r+i, c+j))
+                            dist[(r+i, c+j)] = dist[(r,c)] + 1
+                            
+                            # The child should be updating based on parent!
+                            # The child should be updated once its accessed, not from parent to child update!!!
+                            # because children are accessed multiple times before they get added to visited in top 
+                            # statement. So we have to add child to visited as well!
+                            rooms[r+i][c+j] = dist[(r,c)] + 1
+                            # effectively child was processed in above statement. 
+                            visited.add((r+i, c+j))
+                            # 
+                            print("updated ", (r, c, ))

+-111) Realize in the question when things are sorted for you. Will the datastructure calls come in a sorted manner?
+    Do we need to sort it based on time using a BST or does it come presorted just by how it is called. If it comes presorted
+        put it in a deque or a priority queue instead of a bst. 



+-110)   Doubly Linked List O(1) popping requirement VS Lazy Popping with a deque + map
+        Which soln is better for these datastructures??
+        Build a data structure to perform three operations (Restaurant is full initially):
+        1) waitList (string customer_name, int table_size):
+        Add customer with given name and table size they want to book into the waitlist
+        2) leave (string customer_name):
+        Customer wants to leave the waitlist so remove them.
+        3) serve (int table_size):
+        This means restaurant now has a free table of size equal to table_size. Find the best customer to serve from waitlist
+        Best Customer: Customer whose required size is less than or equal to the table_size. 
+
+        If multiple customers are matching use first come first serve.
+        For e.g. if waitlist has customers with these table requirements => [2, 3, 4, 5, 5, 7] 
+        and restaurant is serving table_size = 6 then best customer is index 3 (0-based indexing).
+
+
+
+        Add to waitlist -> keep it sorted first on size, then on time. 
+
+        to keep sorted on time push into a deque, append!!
+        and also popLeft. (for getting the earlist person!)
+
+
+        map{
+            size -> [customers in deque]
+        }
+
+        Also:  
+        customerName -> locate idx in deque! (Need it implemented as doubly linked list for O(1) removal!)
+
+        map {
+        customerName -> (size, locInDeque/Node in deque)
+        }
+
+        Then when we want to serve someone,
+            binary search the tree map for the queue of names, and give the first name, then remove the first name. 
+
+        Someone elses soln, doesnt use Doubly linked list??
+
+
+            from sortedcontainers import SortedList
+            from collections import deque
+
+            class Restaurant:
+                def __init__(self):
+                    # maps customer to the table they're waiting for
+                    self.customerToTable = {}
+                    # sorted list of unique table sizes that people are waiting for
+                    self.sortedWaitList = SortedList()
+                    # maps table size to a deque of customer names waiting for that table
+                    # in order of arrival. some of these names may have left, so we have 
+                    # to check at the end of each leave call to ensure that we always have
+                    # a valid, present customer waiting for a table at the leftmost element.
+                    self.tableSizeDeques = {}
+
+                def waitList(self, cn, ts):
+                    self.customerToTable[cn] = ts
+                    if ts not in self.sortedWaitList:
+                        self.sortedWaitList.add(ts)
+                    if ts not in self.tableSizeDeques:
+                        self.tableSizeDeques[ts] = deque()
+                    self.tableSizeDeques[ts].append(cn)
+
+                def leave(self, cn):
+                    ts = self.customerToTable[cn]
+                    del self.customerToTable[cn]
+
+                    # this while loop ensures that the leftmost customer has not
+                    # yet left the restaurant for this table size, and if there are 
+                    # no customers left for this table size, we will delete the
+                    # entry for this table size. that means we always have a valid
+                    # customer to use for this table size if it still exists in the waitlist.
+                    while self.tableSizeDeques[ts] and \
+                        self.tableSizeDeques[ts][0] not in self.customerToTable:
+                        self.tableSizeDeques[ts].popleft()
+
+                    if len(self.tableSizeDeques[ts]) == 0:
+                        self.sortedWaitList.remove(ts)
+                        del self.tableSizeDeques[ts]
+
+                def serve(self, ts):
+                    i = min(self.sortedWaitList.bisect_left(ts), \
+                            len(self.sortedWaitList) - 1)
+
+                    customer = self.tableSizeDeques[i][0]
+                    # instead of rewriting our leave code, let's just call it here to make the customer leave
+                    self.leave(self, customer)
+                    return customer
+
+
+-109) Integer Break (Dp soln, not cool math one):
+    In the DP you might have a lot of edge casey stuff around 1 state, or a few states, and
+    then all the other states are easy. In this case, the edge casey stuff was around computing
+    answers to the small input values, dealing with only 2 factors in the product:
+
+    343. Integer Break
+    Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.
+    Return the maximum product you can get.
+    Example 1:
+    Input: n = 2
+    Output: 1
+    Explanation: 2 = 1 + 1, 1 × 1 = 1.
+    Example 2:
+    Input: n = 10
+    Output: 36
+    Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
+    
+    Soln:
+
+    class Solution:
+        def integerBreak(self, n: int) -> int:
+            '''
+            OPT[i] = OPT product we get for i
+            OPT[0] = 0
+            OPT[1] = 0?
+            OPT[2] = 1 -> 1*1? I GUESS
+            OPT[3] = 2
+            OPT[i] = OPT[i - k] * k for every k from n to 1
+            k can go from n to 2 
+            The below solution needs the extra 
+            (i-k)*(k)
+            in the max, 
+            because there are so many edge cases with doing products when small numbers are involved.
+            So calculating anything with 2 factors is edge casey, but 3 or more can build of the 2 
+            product stuff we calculate with the above.
+            
+            '''
+            OPT = [0 for i in range(max(n+1, 4))]
+            
+            for i in range(n+1):
+                for k in range(1, n+1):
+                    if i - k >= 0:
+                        OPT[i] = max(OPT[i], max((i-k)*(k), OPT[i-k]*k))
+            return OPT[n]



        -111.11) Do you think you could use sliding window here?? lolol
            solve this when you have time. 


            727. Minimum Window Subsequence
                Attempted
                Hard
                Topics
                Companies
                Hint
                Given strings s1 and s2, return the minimum contiguous substring part of s1, so that s2 is a subsequence of the part.

                If there is no such window in s1 that covers all characters in s2, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

                

                Example 1:

                Input: s1 = "abcdebdde", s2 = "bde"
                Output: "bcde"
                Explanation: 
                "bcde" is the answer because it occurs before "bdde" which has the same length.
                "deb" is not a smaller window because the elements of s2 in the window must occur in order.
                Example 2:

                Input: s1 = "jmeqksfrsdcmsiwvaovztaqenprpvnbstl", s2 = "u"
                Output: ""


            My attempt:

            class Solution:
                def minWindow(self, s1: str, s2: str) -> str:
                    """
                    SLIDING WINDOW DOESNT WORK ON THIS PROBLEM? DO YOU KNOW WHY LOL? CAUSE
                    I TTRIED AND IT DIDNT! DO DP HERE. BECAUSE IT SAYS SUBSEQUENCE. 


                    s[i][j]
                    at character i how much of j did we match? so is it LCS right?

                    idk is it:
                    DP[i][j] = max( D[i-1][j-1] + 1 if s1[i] == s[j], D[i-1][j], D[i][j-1])

                    how many characters in ht prefix s2[j] do we match when we process each character of s1 right..

                    if we matched all the characters, then DP[i][j] == len(s2)
                    i think!

                    to think about params/tabulation method try to also do it recursively.. and work it backwards.

                    helper(i, j, left_index):
                        
                        if matched: 
                            return left most index?
                            # need to keep track of..
                            # either use the character to match the next character, or dont right. 
                            # just set the answer in the global variable. 
                            # we need to keep track the length of the string we just went through right..?
                            # leftmost index tbh in s1 tbh!

                            # we will need to process s2 over and over hmm 
                            return True, left_index - i 

                        if s2[j] == s1[i]:
                            res1, res1_len = helper(i+1, j+1, left_index)
                        
                        # we process it again!
                        if s1[i] == s2[0]:
                            res2, res2_len = helper(i+1, 1, i)

                        # otherwise, try next character in seq to see if it can match j!
                        third, third_len = helper(i+1, j, left_index)

                        # ok so we solve it and.. 
                        return min()

                    FUCK I CANT DO IT :C

                    DAMN

                    RECURRENT SHOULD LOOK LIKE:
                    For substring S[0, i] and T[0, j], 
                    dp[i][j] is starting index k of the shortest postfix of S[0, i], 
                    such that T[0, j] is a subsequence of S[k, i]. 
                    Here T[0] = S[k], T[j] = S[i]. Otherwise, dp[i][j] = -1.

                    The goal is the substring with length of min(i-dp[i][n-1]) for all i < m,  
                    where m is S.size() and n is T.size() 
                    Initial condition: dp[i][0] = i if S[i] = T[0], else -1
                    Equations: If S[i] = T[j], dp[i][j] = max(dp[k][j-1]) for all k < i; else dp[i][j] = -1;   

                    aka leetcode hint is:
                    Let dp[j][e] = s be the largest index for which S[s:e+1] has T[:j] as a substring.

                    Here is explanation of recurrence:

                    OK leetcode discuss soln:

                    dp[i][j] stores the starting index of the substring where T has length i and S has length j.

                    So dp[i][j would be:
                    if T[i - 1] == S[j - 1], this means we could borrow the start index from dp[i - 1][j - 1] to make the current substring valid;
                    else, we only need to borrow the start index from dp[i][j - 1] which could either exist or not.

                    Finally, go through the last row to find the substring with min length and appears first.


                    dp[i][j] represents the largest occurence of T[0] in S 
                            such that all the characters of T are included in S
                    e.g. S = "abcdebdde" T="bd"
                    dp[2][4] = 2 which corresponds to S being only "abcd"
                    dp[2][7] = 6 which correspods to S being only "abcdebd"            

                        public String minWindow(String S, String T) {
                            int m = T.length(), n = S.length();
                            int[][] dp = new int[m + 1][n + 1];
                            for (int j = 0; j <= n; j++) {
                                dp[0][j] = j + 1;
                            }
                            for (int i = 1; i <= m; i++) {
                                for (int j = 1; j <= n; j++) {
                                    if (T.charAt(i - 1) == S.charAt(j - 1)) {
                                        dp[i][j] = dp[i - 1][j - 1];
                                    } else {
                                        dp[i][j] = dp[i][j - 1];
                                    }
                                }
                            }

                            int start = 0, len = n + 1;
                            for (int j = 1; j <= n; j++) {
                                if (dp[m][j] != 0) {
                                    if (j - dp[m][j] + 1 < len) {
                                        start = dp[m][j] - 1;
                                        len = j - dp[m][j] + 1;
                                    }
                                }
                            }
                            return len == n + 1 ? "" : S.substring(start, start + len);
                        }

                    """






        -111) Trie and Reverse Trie to cover all cases for search...

        1554. Strings Differ by One Character

        Given a list of strings dict where all the strings are of the same length.

        Return true if there are 2 strings that only differ by 1 character in the same index, otherwise return false.

        

        Example 1:

        Input: dict = ["abcd","acbd", "aacd"]
        Output: true
        Explanation: Strings "abcd" and "aacd" differ only by one character in the index 1.
        Example 2:

        Input: dict = ["ab","cd","yz"]
        Output: false
        Example 3:

        Input: dict = ["abcd","cccc","abyd","abab"]
        Output: true
        

        Constraints:

        The number of characters in dict <= 105
        dict[i].length == dict[j].length
        dict[i] should be unique.
        dict[i] contains only lowercase English letters.  


            class TrieNode():
                def __init__(self):

                    self.children = {}  
                    self.is_end = False

            class Trie():
                def __init__(self):
                    self.root = TrieNode()

                def add(self, word):
                    node = self.root
                    
                    for i in word:
                        if node.children.get(i) is None:
                            node.children[i] = TrieNode()
                        node = node.children[i]
                    node.is_end = True
                
                def search_one_off(self, word):

                    node = self.root

                    st = [(node, 0, False)] # second state is differ by one
                    
                    while st:
                        
                        n, i, state = st.pop()
                        if i == len(word) and state and n.is_end:
                            return True 

                        if n.children.get(word[i]) is not None:
                            # ok recur on next letter. 
                            st.append((n.children.get(word[i]), i + 1, state))
                        elif state == False:
                            for k, v in n.children.items():
                                st.append( (v, i+1, True))
                    return False

                def __repr__(self):
                    def recur(node, indent):
                        return "".join(indent + key + ("$" if child.is_end else "") + recur(child, indent + " ") for key, child in node.children.items())

                    return recur(self.root, "\n")

            class Solution:
                def differByOne(self, dict: List[str]) -> bool:
                    """
                    Insert into a trie?
                    then check if trie has it, but one character off?
                    
                    We have to check both forwards and backwards with trie and reverse trie or else you miss cases with just forward 
                    causing incorrect soln.
                    """
                    trie = Trie()
                    reverse_trie = Trie()

                    for word in dict:
                        if trie.search_one_off(word):
                            return True
                        if reverse_trie.search_one_off(word[::-1]):
                            return True 

                        trie.add(word)
                        reverse_trie.add(word[::-1])

                    return False


            The better solution is following:
            Use hashset, to insert all possible combinations adding a character "*". For example: If dict[i] = "abc", insert ("*bc", "a*c" and "ab*").  


-110.5) Strings differ by one character rabinkarp soln:


        class Solution:
            def differByOne(self, dict: List[str]) -> bool:
                # rolling hash
                N, M = len(dict), len(dict[0])
                MOD = 1000000007
                BASE = 27
                memo = set()

                def helper(char: str):
                    if char == "*":
                        return 26
                    return ord(char) - ord('a')

                def collision(word1, word2):
                    cnt = 0
                    for i in range(len(word1)):
                        if word1[i] != word2[i]:
                            cnt += 1
                    return cnt == 1

                # time: O(N * M)
                # space: O(N)
                arr = []
                for word in dict:
                    hash_value = 0
                    for i, char in enumerate(word):
                        hash_value = ((hash_value * BASE) % MOD + helper(char)) % MOD
                    arr.append(hash_value)

                # time: O(N * M)
                # space: O(N * M)
                memo = {}
                for idx, word in enumerate(dict):
                    base = 1
                    hash_value = arr[idx]
                    for i in range(M - 1, -1, -1):
                        new_hash_value = (hash_value + base * (helper("*") - helper(word[i]))) % MOD
                        if new_hash_value in memo:
                            # collision test
                            if collision(word, dict[memo[new_hash_value]]):
                                return True
                        memo[new_hash_value] = idx
                        base = BASE * base % MOD

                return False


-110) DP max dot product of 2 subsequences:
    
    make sure to review LIS, LCS, and other common DP problems!

    class Solution:
        def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
            
            '''
            index i, and j:
            maximum dot product to index i and j for each array. 
            '''
            
            ROWS = len(nums1)
            COLS = len(nums2)
        
            OPT = [[0 for _ in range(COLS) ] for _ in range(ROWS) ]
            
            OPT[0][0] = nums1[0] * nums2[0]
            
            for i in range(1, ROWS):     
                OPT[i][0] = max(OPT[i-1][0], nums1[i]*nums2[0])
            
            for j in range(1, COLS):
                OPT[0][j] = max(OPT[0][j-1], nums1[0]*nums2[j])
            
            for i in range(1, ROWS):
                for j in range(1, COLS):
                    OPT[i][j] = max(nums1[i]*nums2[j], 
                                    OPT[i-1][j-1] + nums1[i]*nums2[j], 
                                    OPT[i][j-1], 
                                    OPT[i-1][j])    
            return OPT[-1][-1]




-109) Kth largest element using counting sort because why not:

    class Solution:
        def findKthLargest(self, nums: List[int], k: int) -> int:
            """
            if heap len > k and heap[0] < num, heapop and then heappush
            """
            
            max_n = max(nums)
            min_n = min(nums)
            counts = [0] * (max_n - min_n  + 1)
            for num in nums:
                counts[num - min_n] += 1
            delta = k
            for idx in range(len(counts) - 1, -1, -1):
                delta -= counts[idx]
                if delta <= 0:
                    return idx + min_n
            raise "SHEEET"

-108) Sliding window example subtracting K from K-1:
        ALSO A REVIEW ON COUNTING -> LEARN 3 WAY INTERSECTION VENN DIAGRAM COUNTING!

        992. Subarrays with K Different Integers
        Given an integer array nums and an integer k, return the number of good subarrays of nums.

        A good array is an array where the number of different integers in that array is exactly k.

        For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.
        A subarray is a contiguous part of an array.
        

        Example 1:

        Input: nums = [1,2,1,2,3], k = 2
        Output: 7
        Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2]
        Example 2:

        Input: nums = [1,2,1,3,4], k = 3
        Output: 3
        Explanation: Subarrays formed with exactly 3 different integers: [1,2,1,3], [2,1,3], [1,3,4].
        

        Constraints:

        1 <= nums.length <= 2 * 104
        1 <= nums[i], k <= nums.length

        class Solution:
            def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
                def helper(nums, k):
                    i = 0 
                    j = 0 
                    if k == 0:
                        return 0 

                    distinct = {}
                    cnt = 0
                    while True: 
                        while j < len(nums):
                            if nums[j] in distinct:
                                distinct[nums[j]] += 1        
                            
                            elif len(distinct.keys()) == k:
                                # cannot insert this right!
                                # count pairs and break..
                                # we cant insert this element yet.. 
                                break
                            else:
                                distinct[nums[j]] = 1
                            j+= 1
                        while i < j:
                            cnt += j - i 
                            toRemove = nums[i]
                            distinct[toRemove] -=  1
                            i += 1
                            if distinct[toRemove] == 0:
                                del distinct[toRemove]
                                break
                                # ok should be fixed.
                            # window should be fixed..
                        
                        if i == j == len(nums):
                            break 
                            
                    return cnt 

                return helper(nums, k) - helper(nums, k-1)


-107.5) LCA traversing from child to parent:

        class Solution:
            def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
                """
                """
                # CREATE a reverse graph ... 
                g = defaultdict(list)

                for region in regions:
                    
                    parent = region[0]
                    for i in range(1, len(region)):
                        g[ region[i] ] = parent 

                # now get a set of all the parents for region 1 and store it.
                # then climb up region 2 and see if you see it..
                marked = set()
                def find_parents(reg):
                    if reg in marked:
                        # found lca
                        return reg

                    marked.add(reg)
                    parent = g.get(reg, None)
                    
                    if parent is None:
                        return None

                    return find_parents(parent)

                res1 = find_parents(region1)
                res2 = find_parents(region2)
                print("res1, res2",res1, res2)
                return res1 or res2



-107) Find median in datastrea:

        class MedianFinder:

            def __init__(self):
                """
                Use 1 MAXHEAP for bottom half of numbers.
                Use 1 MINHEAP for top half of numbers. 
                """
                self.low = []  # maxheap
                self.high = [] # minheap
                

            def addNum(self, num: int) -> None:            
                if len(self.low) == len(self.high):
                    # insert into low!
                    # same size, insert into low.
                    # compare against high, if its smaller, add to low, 
                    # if bigger, pop high, insert into high, put popped element o fhigh into low
                    insert = num
                    if len(self.high) > 0 and self.high[0] < num:
                        insert = heapq.heappop(self.high)
                        heapq.heappush(self.high, num)
                        
                    heapq.heappush(self.low, -insert)             
                else:
                    insert = num
                    if  -self.low[0] > num:
                        insert = -(heapq.heappop(self.low))
                        heapq.heappush(self.low, -num)
                        
                    heapq.heappush(self.high, insert)
                
            def findMedian(self) -> float:
                if len(self.low) == len(self.high):
                    # print("self.low", self.low)
                    # convert maxh val from neg -> pos    
                    med = (-self.low[0] + self.high[0])/2
                    return med
                else:
                    return -self.low[0]


-106) Djiksta Cheapest flight within K stops:

        from collections import defaultdict
        from heapq import *

        class Solution:
            def create_graph(self, flights):
                g = defaultdict(list)
                
                for flight in flights:
                    s, d, cost = flight
                    g[s].append((d, cost))
            
                return g
            
            
            def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
                # Modified Djikstra, aka BFS
                
                g = self.create_graph(flights)
                
                pq = []
                heappush(pq, [0, K+1, src])
                
                while pq:
                    
                    cost, k, node = heappop(pq)
                    
                    print("popped node", (cost, k, node))
                    
                    if node == dst:
                        return cost
                    
                    if k == 0:
                        continue
                    
                    for kid, kid_w in g[node]:
                        heappush(pq, [cost + kid_w, k-1, kid])
                        
                return -1


-105) SUDOKU SOLVER back tracking.
    Write a program to solve a Sudoku puzzle by filling the empty cells.

    A sudoku solution must satisfy all of the following rules:

    Each of the digits 1-9 must occur exactly once in each row.
    Each of the digits 1-9 must occur exactly once in each column.
    Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
    The '.' character indicates empty cells.
    Do you know why heap wouldnt improve the performance here?
    constant time because theres only 9 rows and 9 cols. 



    class Solution:
        def solveSudoku(self, board: List[List[str]]) -> None:
            rows = [[0 for j in range(10)] for i in range(9)]
            cols = [[0 for j in range(10)]for i in range(9)]
            boxes = [[0 for j in range(10)] for i in range(9)]

            def boxidx(r, c):
                return 3*(r//3) + c//3

            pos = []
            for ri, row in enumerate(board): 
                for ci, ele in enumerate(row):
                    if ele != ".":
                        ele = int(ele)
                        rows[ri][ele] = 1
                        cols[ci][ele] = 1
                        boxes[boxidx(ri, ci)][ele] = 1
                    else:
                        # keep track of these elements we set. 
                        # actually if we are allowed to modify the board in place.. lets just do that
                        # iterate the board.
                        pos.append((ri, ci))

            def setVal(i, j, val):
                if rows[i][val] != 1 and cols[j][val] != 1 and boxes[boxidx(i, j)][val] != 1:
                    # set it in all of them. 
                    board[i][j] = str(val)
                    rows[i][val] = 1
                    cols[j][val] = 1
                    boxes[boxidx(i, j)][val] = 1
                    return True 
                return False

            def unsetVal(i, j, val):
                board[i][j] = "."
                rows[i][val] = 0
                cols[j][val] = 0
                boxes[boxidx(i, j)][val] = 0

            def solve(i):
                if i == len(pos):
                    return True 
                (ri, ci) = pos[i]
                for num in range(1, 10):
                    check = setVal(ri, ci, num)
                    if check:
                        res = solve(i+1)
                        if res:
                            return True
                        else:
                            unsetVal(ri,ci, num)
            solve(0)
            return board




-104) FIND RIGHT INTERVAL C++ WITH PRIORITY QUEUE (fast solution):
      (2 ARRAY INTERVAL SOLUTIONS WITH SORTED START END TIMES + POINTERS)
    fck 2 array ? bc sorting does nlogn work anyway.

    Just sort the starts (with attached original indexes), then binary search the ends in them.
    import bisect

    class Solution:
        def findRightInterval(self, intervals):
            starts = sorted([I[0], i] for i, I in enumerate(intervals)) + [[float('inf'), -1]]
            return [starts[bisect.bisect(starts, [I[1]])][1] for I in intervals]




-103) FIND RIGHT INTERVAL C++ WITH PRIORITY QUEUE (slow solution):
    Find Right Interval
    Medium
    You are given an array of intervals, where 
    intervals[i] = [starti, endi] and each starti is unique.

    The right interval for an interval i is an interval j 
    such that startj >= endi and startj is minimized.

    Return an array of right interval indices for each interval i. 
    If no right interval exists for interval i, then put -1 at index i.

    Example 1:

    Input: intervals = [[1,2]]
    Output: [-1]
    Explanation: There is only one interval in the collection, so it outputs -1.
    Example 2:

    Input: intervals = [[3,4],[2,3],[1,2]]
    Output: [-1,0,1]
    Explanation: There is no right interval for [3,4].
    The right interval for [2,3] is [3,4] since start0 = 3 is the smallest start that is >= end1 = 3.
    The right interval for [1,2] is [2,3] since start1 = 2 is the smallest start that is >= end2 = 2.
    Example 3:

    Input: intervals = [[1,4],[2,3],[3,4]]
    Output: [-1,2,-1]
    Explanation: There is no right interval for [1,4] and [3,4].
    The right interval for [2,3] is [3,4] since start2 = 3 is the smallest start that is >= end1 = 3.


    class Solution {
        public:
            vector<int> findRightInterval(vector<vector<int>>& intervals) {
                // sort by start time. 
                // then just choose the right interval as you go through? 
                /*
                sorting + binary search?
                sorting + heap?
                sorting + treemap? 
                sorting + 2 arrays [ayyy the best soln ]
                
                The reason we shoud seek something faster than heap/treemap is because
                we are dealing with static data that doesnt change, and those structures are used 
                for dynamic data, hence 2 array soln. 
                */    
                
                //loop through intervals and save the index as part of tuple!
                for(int i = 0; i != intervals.size(); ++i) {
                    intervals[i].push_back(i);
                }
                
                sort(intervals.begin(), intervals.end(), [](auto x, auto y) { return x[0] < y[0];});
                
                // sort by largest finish time at top!
                // IF YOU LOOK AT CMP, TO DO LEAST TO GREATEST, YOU ACTUALLY HAVE TO INVERSE
                // so its not a[1] < b[1] like in sort function above but a[1] > b[1]
                auto cmp = [](vector<int> a, vector<int>  b) {return a[1] > b[1];};
                priority_queue< vector<int>, vector< vector<int> >, decltype(cmp)> pq(cmp);
                
                vector<int> res;
                res.resize(intervals.size());
                
                for(int i = 0; i!= intervals.size(); ++i) {
                    vector<int> inte = intervals[i];
                    
                    while(pq.size() > 0 && pq.top()[1] <= inte[0]) {
                        res[pq.top()[2]] = inte[2];
                        pq.pop();
                    }
                    pq.push(inte);
                }
                
                while(pq.size() > 0) {
                    res[pq.top()[2]] = -1;
                    pq.pop();
                }
                return res; 
            }
    };





-102) BINARY SEARCH AND THE TEMPLATES: 

        1.   Peak Index in a Mountain Array
        Let's call an array arr a mountain if the following properties hold:
        arr.length >= 3
        There exists some i with 0 < i < arr.length - 1 such that:
        arr[0] < arr[1] < ... arr[i-1] < arr[i]
        arr[i] > arr[i+1] > ... > arr[arr.length - 1]
        Given an integer array arr that is guaranteed to be a mountain, 
        return any i such that arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1].

        Example 1:

        Input: arr = [0,1,0]
        Output: 1
        Example 2:

        Input: arr = [0,2,1,0]
        Output: 1

        class Solution {
            public:
            int peakIndexInMountainArray(vector<int>& arr) {
                int low = 0;
                int high = arr.size();

                while(low < high) {
                    int mid = low + (high - low)/2;
                    
                    /*
                    if(mid - 1 < 0) {
                        low = low+1;
                        continue;
                    }    
                    if(mid + 1 >= arr.size()) {
                        high = high-1; 
                        continue;
                    }
                    */
                    
                    int left = arr[mid-1];
                    int right= arr[mid+1];
                    
                    if(left < arr[mid] && arr[mid] < right) {
                        // search right side 
                        low = mid + 1;
                    } else if(left < arr[mid] && arr[mid] > right) {
                        return mid; 
                    } else if(left > arr[mid] && arr[mid] > right) {
                        high = mid;
                    } 
                }
                return -9999999; // should never reach this. 
            }
        };


-101.7) Binary Search with chocolates:

     Divide Chocolate
     You have one chocolate bar that consists of some chunks. 
     Each chunk has its own sweetness given by the array sweetness.

     You want to share the chocolate with your K friends so you start 
     cutting the chocolate bar into K+1 pieces using K cuts, 
     each piece consists of some consecutive chunks.

     Being generous, you will eat the piece with the minimum total 
     sweetness and give the other pieces to your friends.

     Find the maximum total sweetness of the piece you can 
     get by cutting the chocolate bar optimally.

      

     Example 1:

     Input: sweetness = [1,2,3,4,5,6,7,8,9], K = 5
     Output: 6
     Explanation: You can divide the chocolate to [1,2,3], [4,5], [6], [7], [8], [9]

    class Solution {
        public:
            
            int enough(const vector<int> & sweetness, int minSweet, int K) {
                int res = 0;
                int groups = 0;
                for(auto & i: sweetness) {
                    if(res + i >= minSweet) {
                        groups += 1;
                        res = 0;
                    } else {
                        res += i;
                    }
                }
                if(groups >= K+1) 
                    return true;
                return false; 
            }
            
            int maximizeSweetness(vector<int>& sweetness, int K) {
                int low = *min_element(sweetness.begin(), sweetness.end());
                int high = std::accumulate(sweetness.begin(), sweetness.end(), 0) + 1;
                int mid;
                
                while(low < high) {
                    mid =  low + (high - low)/2;
                    if(enough(sweetness, mid, K)) {
                        low = mid + 1;
                    }  else {
                        high = mid;
                    }
                }
                return low - 1;
            }
    };

    PYTHON SOLN:
    def maximizeSweetness(self, A, K):
        left, right = 1, sum(A) / (K + 1)
        while left < right:
            mid = (left + right + 1) / 2
            cur = cuts = 0
            for a in A:
                cur += a
                if cur >= mid:
                    cuts += 1
                    cur = 0
            if cuts > K:
                left = mid
            else:
                right = mid - 1
        return right




-101.5) Binary Search again: 

     1.    Find the Smallest Divisor Given a Threshold

     Share
     Given an array of integers nums and an integer threshold, we will 
     choose a positive integer divisor and divide all the array by it and 
     sum the result of the division. Find the smallest divisor such that the 
     result mentioned above is less than or equal to threshold.

     Each result of division is rounded to the nearest integer greater 
     than or equal to that element. (For example: 7/3 = 3 and 10/2 = 5).
     It is guaranteed that there will be an answer.



    class Solution {
    public:
        bool enough(vector<int>& nums, const int & threshold, const int & divisor) {
            int res = 0;
            for(auto & i : nums) {
                // cieling
                res += (i/divisor) + (i % divisor != 0);
                if(res  > threshold){
                    return false;
                }
            } 
            return true;
        }
        
        int smallestDivisor(vector<int>& nums, int threshold) {
            int low = 1; 
            int high = *max_element(nums.begin(), nums.end());
            while(low < high) {   
                int mid = low + (high - low)/2;
                if(enough(nums, threshold, mid)) {
                    // divisor worked go smaller. 
                    high = mid;
                } else {
                    //divisior too small, need bigger. 
                    low = mid + 1;
                }
            }
            return low;
        }
    };



-101) Binary Search VARIABLES HOW TO SET:
    STUDY HOW Low, high, and the condiition in if statement for binary search
    // both solutions below work. 

    1.   Split Array Largest Sum
    Share
    Given an array nums which consists of non-negative integers and an integer m, 
    you can split the array into m non-empty continuous subarrays.

    Write an algorithm to minimize the largest sum among these m subarrays.
    Example 1:
    Input: nums = [7,2,5,10,8], m = 2
    Output: 18

    Solution: 

    class Solution {
        public:
            
            int enough(vector<int>& nums, int m, int k) {
                int groups = 0;
                int curr = 0;
                for(auto & i : nums) {
                    if(curr + i > k)  {
                        groups += 1;
                        curr = 0;
                    }
                    curr += i;
                }
                groups += 1; // last group
                if(groups > m) {
                    return false;
                } 
                return true; 
            }
            
            int splitArray(vector<int>& nums, int m) {
                // binary search because we only want the minimized value as answer 
                int high = std::accumulate(nums.begin(), nums.end(), 0);
                // low is actually the largest element in the array? 
                int low = *max_element(nums.begin(), nums.end());
                int ans = high;
                
                while(low < high) {
                    
                    int mid = low + (high - low)/2;
                    if(enough(nums, m, mid)) {
                        high = mid;
                        ans = min(ans, mid);
                    } else {
                        low = mid+1; 
                    }   
                }
                // returning low or high below also works!
                // bc at end of loop low==high==mid -> enough
                return ans;
            }

            int splitArray2(vector<int>& nums, int m) {
                // binary search because we only want the minimized value as answer 
                int high = std::accumulate(nums.begin(), nums.end(), 0);
                // low is actually the largest element in the array? 
                int low = *max_element(nums.begin(), nums.end());
                int ans = high;
                
                while(low <= high) {
                    
                    int mid = low + (high - low)/2;
                    // cout << "testing value " << mid << endl; 
                    if(enough(nums, m, mid)) {
                        // ok that worked. can we go smaller?
                        high = mid-1;
                        ans = min(ans, mid);
                    } else {
                        // we need it bigger. 
                        low = mid+1; 
                    }   
                }
                // only returning low or ans works here. cant return high
                return ans;
            }   
    };





-100) MOD TRICKS -> GET MOD IN BETWEEN [0, K]


    /*
    1.    Check If Array Pairs Are Divisible by k
    Medium
    Share
    Given an array of integers arr of even length n and an integer k.
    We want to divide the array into exactly n / 2 pairs such that the sum of each pair is divisible by k.
    Return True If you can find a way to do that or False otherwise.

        #include <bits/stdc++.h> 

        class Solution {
        public:
            bool canArrange(vector<int>& arr, int k) {
                unordered_multiset<int> s;
                
                for(auto & i : arr) {
                    if(i < 0){ 
                        // HOW TO MAKE NUMBER POSITIVE
                        i += (abs(i)/k + (i%k != 0))*k;
                    }
                    // ANOTHER WAY  TO KEEP MODS BETWEEN [0, K-1 is just do following]
                    // i = (i%k + k)%k
                    if(s.find(k - i%k) != s.end()) {
                        s.erase(s.find(k - i%k));
                    } else {
                        if(i%k == 0) {
                            s.insert(k);
                        }else {                
                            s.insert(i%k);
                        }
                    }
                }
                if(s.size() == 0) {
                    return true;
                }
                return false;
            }
        };


    // VERY CLEAN SOLUTION

    class Solution {
    public:
        bool canArrange(vector<int>& arr, int k) {
            vector<int> freq(k);
            
            for (int x : arr)
                freq[((x % k) + k) % k]++;
            
            if (freq[0] % 2)
                return false;
            
            for (int i=1, j=k-1; i<j; i++, j--)
                if (freq[i] != freq[j])
                    return false;
            
            return true;
        }
    };

    /*
    FASTER CPP SOLUTIONS
    */

    class Solution {
    public:
        bool canArrange(vector<int>& arr, int k) {
            vector<int> freq(k,0);
            int n = arr.size();
            for(int i = 0; i < n; ++i) {
                if(arr[i] >= 0) {
                    freq[arr[i] % k] = ((freq[arr[i] % k] + 1) % k);
                }
                else {
                    int temp = k - abs(arr[i] % k);
                    if(temp == k)
                        temp = 0;
                    freq[temp] = ((freq[temp] + 1) % k);
                }
            }

            if(freq[0] % 2 != 0) 
                return false;
            for(int i = 1; i <= freq.size() / 2; i++){
                if(freq[i] != freq[k - i]) return false;
            }
            return true;
        }
        
    };

    static const auto speedup = []() {
            std::ios::sync_with_stdio(false); std::cin.tie(nullptr); cout.tie(nullptr); return 0;
    }();



-99) Use 2 MULTISET TREEMAPS instead of 2 Heaps for Median Finding Trick!!
    1)   Sliding Window Median
    (However slightly slower because of log(N) 
    cost retrival of best elements vs PQ)
    Window position                Median
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       1
    1 [3  -1  -3] 5  3  6  7       -1
    1  3 [-1  -3  5] 3  6  7       -1
    1  3  -1 [-3  5  3] 6  7       3
    1  3  -1  -3 [5  3  6] 7       5
    1  3  -1  -3  5 [3  6  7]      6

        class Solution {
        public:
            void insertTree(int element, multiset<double> & l, multiset<double> & r) {
                if(l.size() == r.size()) {
                    // insert into left. 
                    
                    if(r.size() > 0 && *(r.begin()) < element) { 
                        double temp = *(r.begin());
                        
                        // ERASING BY VALUE IS BUG FOR MULTISET BECAUSE IT REMOVES ALL COPIES
                        // ONLY ERASE THE ITERATOR!! TO ERASE ONE. 
                        r.erase(r.begin());
                        r.insert(element);
                        element = temp;
                    }
                    l.insert(element);
                } else {
                    // l is bigger, insert into right. 
                    
                    if( *(--l.end()) > element ) {
                        double temp = *(--l.end()) ;
                        l.erase(--l.end()); //COOL TIP, YOU CAN ERASE WITH EITHER VALUE OR ITERATOR
                        l.insert(element);
                        element = temp; 
                    }
                    
                    r.insert(element);
                }
            }
            
            void deleteTree(int element, multiset<double> & l, multiset<double> & r ) {
                // Find tree that contains element, remove, then rebalance. 
                bool leftBigger = l.size() > r.size();
                
                auto leftSearch =l.find(element);  
                if( leftSearch != l.end()) {
                    l.erase(leftSearch);
                    // if left is greater than right by 1 dont do anything    
                    // if left is same size as right, move right element to left.  
                    if(!leftBigger) {
                        // move right to left. 
                        auto rightEle = *(r.begin());
                        r.erase(r.begin());
                        l.insert(rightEle);
                    }            
                } else {
                    // search right, has to contain it.  
                    auto rightSearch = r.find(element);
                    r.erase(rightSearch);
                    
                    // if left is same size as right do nothing
                    // otherwise, move left to right. 
                    
                    if(leftBigger) {
                        auto leftEle = *(--l.end());
                        l.erase(--l.end());
                        r.insert(leftEle);
                    }
                }
            }
            
            
            double calcMedian(const multiset<double> & l, const multiset<double> & r) {
            // always ensure left has 1 more element than right. 
            // then always return *(left.end() - 1)
                
                if(l.size() == r.size()) {
                    
                    return ( *(--l.end()) + *(r.begin()) ) / 2.0;  
                }  else {
                    return *(--l.end());
                }
            } 
            
            vector<double> medianSlidingWindow(vector<int>& nums, int k) {    
                // keep 2 multsets. 
                multiset<double> l;
                multiset<double> r;
                
                int i = 0;
                int j = 0;

                while(j < k) {            
                    insertTree(nums[j], l, r);
                    j += 1;
                }
                
                vector<double> res;
                double med = calcMedian(l, r);
                res.push_back(med);
                
                while(j != nums.size()) {            
                    insertTree(nums[j], l, r);
                    deleteTree(nums[i], l, r);
                
                    med = calcMedian(l, r);
                    res.push_back(med);
                    i += 1;
                    j += 1;
                }
                return res;    
            }
        };


    

-98) C++ A Priority Queue USAGE VS BINARY TREE MAP USAGE: 
    
    1. Find K Pairs with Smallest Sums

    You are given two integer arrays nums1 and nums2 
    sorted in ascending order and an integer k.

    Define a pair (u,v) which consists of one element from the first array and one element from the second array.

    Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.


    class Solution:
        def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
            from heapq import heappush, heappop
            m = len(nums1)
            n = len(nums2)

            ans = []
            visited = set()

            minHeap = [(nums1[0] + nums2[0], (0, 0))]
            visited.add((0, 0))
            count = 0

            while k > 0 and minHeap:
                val, (i, j) = heappop(minHeap)
                ans.append([nums1[i], nums2[j]])

                if i + 1 < m and (i + 1, j) not in visited:
                    heappush(minHeap, (nums1[i + 1] + nums2[j], (i + 1, j)))
                    visited.add((i + 1, j))

                if j + 1 < n and (i, j + 1) not in visited:
                    heappush(minHeap, (nums1[i] + nums2[j + 1], (i, j + 1)))
                    visited.add((i, j + 1))
                k = k - 1
            
            return ans



    Example 1:

    Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
    Output: [[1,2],[1,4],[1,6]] 
    Explanation: The first 3 pairs are returned from the sequence: 
                [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
    Example 2:

    Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
    Output: [1,1],[1,1]
    Explanation: The first 2 pairs are returned from the sequence: 
                [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
    Example 3:

    Input: nums1 = [1,2], nums2 = [3], k = 3
    Output: [1,3],[2,3]
    Explanation: All possible pairs are returned from the sequence: [1,3],[2,3]

    C++ SOLUTION A (fastest):

    class Solution {
        public:
        vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
            vector<pair<int,int>> result;
            if (nums1.empty() || nums2.empty() || k <= 0)
                return result;
            auto comp = [&nums1, &nums2](pair<int, int> a, pair<int, int> b) {
                return nums1[a.first] + nums2[a.second] > nums1[b.first] + nums2[b.second];};
            priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(comp)> min_heap(comp);
            min_heap.emplace(0, 0);
            while(k-- > 0 && min_heap.size())
            {
                auto idx_pair = min_heap.top(); min_heap.pop();
                result.emplace_back(nums1[idx_pair.first], nums2[idx_pair.second]);
                if (idx_pair.first + 1 < nums1.size())
                    min_heap.emplace(idx_pair.first + 1, idx_pair.second);
                if (idx_pair.first == 0 && idx_pair.second + 1 < nums2.size())
                    min_heap.emplace(idx_pair.first, idx_pair.second + 1);
            }
            return result;
        }
    };



    C++ SOLUTION B (slower):

    struct compare
    {
        bool operator() (const pair<int, int> & a, const pair<int, int> & b)
        {
            return a.first + a.second >= b.first + b.second;
        }
    };

    class Solution {
    public:
        vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
            
            if (nums1.empty() || nums2.empty() || k == 0)
                return {};
                
            priority_queue< pair<int, int>, vector<pair<int, int>>, compare > que;
            
            int N = nums1.size();
            int M = nums2.size();
            
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    que.push({nums1[i], nums2[j]});       
                }
            }
            
            vector<vector<int>> ans;
            
            int count = min(k, (int)que.size());
            
            for (int s = 0; s < count; s++)
            {
                auto item = que.top();
                que.pop();
                
                ans.push_back({});
                ans.back().push_back(item.first);
                ans.back().push_back(item.second);
            }
            
            return ans;
        }
    };


    class Solution {
    public:
        vector<vector<int>> kSmallestPairs(vector<int>& v1, vector<int>&v2, int k) {
            map<int,vector<pair<int,int>>>mp;
            int sz1=v1.size(),sz2=v2.size();
            for(int i=0;i<sz1;++i){
                for(int j=0;j<sz2;++j)
                    mp[v1[i]+v2[j]].push_back({v1[i],v2[j]});
            }

            vector<vector<int>>res;
            for(auto it=mp.begin();it!=mp.end();++it){
                for(pair<int,int>p:it->second){
                    if(res.size()==k)
                    break;
                    res.push_back({p.first,p.second});
                }
            }
            return res;
        }
    };





-97) INTERVAL QUESTIONS THAT CAN BE 
     SOLVED BY EITHER SORTING BY START TIME OR END TIME

    Hanging Banners
    Question 212 of 858
    You are given a list of list of integers 
    intervals of the form [start, end] representing 
    the starts and end points of banners you want to hang. 
    Each banner needs at least one pin to stay up, and one 
    pin can hang multiple banners. Return the smallest number 
    of pins required to hang all the banners.

    Note: The endpoints are inclusive, so if two banners are 
    touching, e.g. [1, 3] and [3, 5], you can put a pin at 
    3 to hang both of them.

    intervals = [
        [1, 4],
        [4, 5],
        [7, 9],
        [9, 12]
    ]
    Output

    2
    Explanation

    You can put two pins at 4 and 9 to hang all the banners..

    Example 2
    Input

    intervals = [
        [1, 10],
        [5, 10],
        [6, 10],
        [9, 10]
    ]
    Output

    1
    Explanation

    You can put one pin at 10.

    // TWO WAYS TO SOLVE WOWOWO
    // YOU CAN EITHER SORT BY START TIME LIKE BELOW
    int solve1(vector<vector<int>>& intervals) {
        /*
        sort by start time, 

        keep set of end times. 
        update with smallest end time so far seen. 

        if next interval is past the current smallest end time, pop all intervals and add a pin,
        then restart algo. 
        */
        sort(intervals.begin(), intervals.end(), 
             [](vector<int> a, vector<int> b)-> bool {return a[0] < b[0];} );
        
        int pins = 0;
        int nearestEnd = -1;
        
        for(int i = 0; i != intervals.size(); ++i) {
            
            auto intv = intervals[i];
            if(intv[0] > nearestEnd) {
                pins += 1;
                nearestEnd = intv[1];
            } else {
                // keep in set of intervals!
                nearestEnd = min(nearestEnd, intv[1]);
            }
        }
        return pins;
    }

    // YOU CAN SORT BY END TIME TOO LIKE BELOW: 

    class Solution:
        def solve(self, intervals):
            intervals.sort(key=lambda i: i[1])
            last = float("-inf")
            ans = 0
            for s, e in intervals:
                if s <= last:
                    continue
                last = e
                ans += 1
            return ans





-96) Largest Rectangle in Histogram with Pointer Segment Tree:

    // Largest Rectangle in Histogram
    // Stack solution, O(NlogN) solution

    class SegTreeNode {
    public:
    int start;
    int end;
    int min;
    SegTreeNode *left;
    SegTreeNode *right;
    SegTreeNode(int start, int end) {
        this->start = start;
        this->end = end;
        left = right = NULL;
    }
    };

    class Solution {
    public:
    int largestRectangleArea(vector<int>& heights) {
        if (heights.size() == 0) return 0;
        // first build a segment tree
        SegTreeNode *root = buildSegmentTree(heights, 0, heights.size() - 1);
        // next calculate the maximum area recursively
        return calculateMax(heights, root, 0, heights.size() - 1);
    }
    
    int calculateMax(vector<int>& heights, SegTreeNode* root, int start, int end) {
        if (start > end) {
        return -1;
        }
        if (start == end) {
        return heights[start];
        }
        int minIndex = query(root, heights, start, end);
        int leftMax = calculateMax(heights, root, start, minIndex - 1);
        int rightMax = calculateMax(heights, root, minIndex + 1, end);
        int minMax = heights[minIndex] * (end - start + 1);
        return max( max(leftMax, rightMax), minMax );
    }
    
    SegTreeNode *buildSegmentTree(vector<int>& heights, int start, int end) {
        if (start > end) return NULL;
        SegTreeNode *root = new SegTreeNode(start, end);
        if (start == end) {
            root->min = start;
        return root;
        } else {
        int middle = (start + end) / 2;
        root->left = buildSegmentTree(heights, start, middle);
        root->right = buildSegmentTree(heights, middle + 1, end);
        root->min = heights[root->left->min] < heights[root->right->min] ? root->left->min : root->right->min;
        return root;
        }
    }
    
    int query(SegTreeNode *root, vector<int>& heights, int start, int end) {
        if (root == NULL || end < root->start || start > root->end) return -1;
        if (start <= root->start && end >= root->end) {
        return root->min;
        }
        int leftMin = query(root->left, heights, start, end);
        int rightMin = query(root->right, heights, start, end);
        if (leftMin == -1) return rightMin;
        if (rightMin == -1) return leftMin;
        return heights[leftMin] < heights[rightMin] ? leftMin : rightMin;
    }
    };

-95) NLOGN solution Largest Rectangle in Histogram with 
    DIVID & CONQUER + SEGMENT TREES (Segment tree built with array): 

    We can use Divide and Conquer to solve this in O(nLogn) time. 
    The idea is to find the minimum value in the given array. 
    Once we have index of the minimum value, the max area is maximum of following three values.
    a) Maximum area in left side of minimum value (Not including the min value)
    b) Maximum area in right side of minimum value (Not including the min value)
    c) Number of bars multiplied by minimum value.
    The areas in left and right of minimum value bar can be calculated recursively. 
    If we use linear search to find the minimum value, then the worst case time 
    complexity of this algorithm becomes O(n^2). In worst case, we always have (n-1) 
    elements in one side and 0 elements in other side and if the finding minimum 
    takes O(n) time, we get the recurrence similar to worst case of Quick Sort.

    How to find the minimum efficiently? Range Minimum Query 
    using Segment Tree can be used for this. We build segment tree of 
    the given histogram heights. Once the segment tree is built, all 
    range minimum queries take O(Logn) time. So over all
    complexity of the algorithm becomes.

    Overall Time = Time to build Segment Tree + Time to recursively find maximum area
    Time to build segment tree is O(n). Let the time to recursively find max area be T(n). 
    It can be written as following.
    T(n) = O(Logn) + T(n-1)

    The solution of above recurrence is O(nLogn). 
    So overall time is O(n) + O(nLogn) which is O(nLogn).

    C++ SOLN ###################################

        // A Divide and Conquer Program to find maximum rectangular area in a histogram 
        #include <bits/stdc++.h> 
        using namespace std; 
        
        // A utility function to find minimum of three integers 
        int max(int x, int y, int z) 
        {  return max(max(x, y), z); } 
        
        // A utility function to get minimum of two numbers in hist[] 
        int minVal(int *hist, int i, int j) 
        { 
            if (i == -1) return j; 
            if (j == -1) return i; 
            return (hist[i] < hist[j])? i : j; 
        } 
        
        // A utility function to get the middle index from corner indexes. 
        int getMid(int s, int e) 
        {   return s + (e -s)/2; } 
        
        /*  A recursive function to get the index of minimum value in a given range of 
            indexes. The following are parameters for this function. 
        
            hist   --> Input array for which segment tree is built 
            st    --> Pointer to segment tree 
            index --> Index of current node in the segment tree. Initially 0 is 
                    passed as root is always at index 0 
            ss & se  --> Starting and ending indexes of the segment represented by 
                        current node, i.e., st[index] 
            qs & qe  --> Starting and ending indexes of query range */
        int RMQUtil(int *hist, int *st, int ss, int se, int qs, int qe, int index) 
        { 
            // If segment of this node is a part of given range, then return the 
            // min of the segment 
            if (qs <= ss && qe >= se) 
                return st[index]; 
        
            // If segment of this node is outside the given range 
            if (se < qs || ss > qe) 
                return -1; 
        
            // If a part of this segment overlaps with the given range 
            int mid = getMid(ss, se); 
            return minVal(hist, RMQUtil(hist, st, ss, mid, qs, qe, 2*index+1), 
                        RMQUtil(hist, st, mid+1, se, qs, qe, 2*index+2)); 
        } 
        
        // Return index of minimum element in range from index qs (quey start) to 
        // qe (query end).  It mainly uses RMQUtil() 
        int RMQ(int *hist, int *st, int n, int qs, int qe) 
        { 
            // Check for erroneous input values 
            if (qs < 0 || qe > n-1 || qs > qe) 
            { 
                cout << "Invalid Input"; 
                return -1; 
            } 
        
            return RMQUtil(hist, st, 0, n-1, qs, qe, 0); 
        } 
        
        // A recursive function that constructs Segment Tree for hist[ss..se]. 
        // si is index of current node in segment tree st 
        int constructSTUtil(int hist[], int ss, int se, int *st, int si) 
        { 
            // If there is one element in array, store it in current node of 
            // segment tree and return 
            if (ss == se) 
            return (st[si] = ss); 
        
            // If there are more than one elements, then recur for left and 
            // right subtrees and store the minimum of two values in this node 
            int mid = getMid(ss, se); 
            st[si] =  minVal(hist, constructSTUtil(hist, ss, mid, st, si*2+1), 
                            constructSTUtil(hist, mid+1, se, st, si*2+2)); 
            return st[si]; 
        } 
        
        /* Function to construct segment tree from given array. This function 
        allocates memory for segment tree and calls constructSTUtil() to 
        fill the allocated memory */
        int *constructST(int hist[], int n) 
        { 
            // Allocate memory for segment tree 
            int x = (int)(ceil(log2(n))); //Height of segment tree 
            int max_size = 2*(int)pow(2, x) - 1; //Maximum size of segment tree 
            int *st = new int[max_size]; 
        
            // Fill the allocated memory st 
            constructSTUtil(hist, 0, n-1, st, 0); 
        
            // Return the constructed segment tree 
            return st; 
        } 
        
        // A recursive function to find the maximum rectangular area. 
        // It uses segment tree 'st' to find the minimum value in hist[l..r] 
        int getMaxAreaRec(int *hist, int *st, int n, int l, int r) 
        { 
            // Base cases 
            if (l > r)  return INT_MIN; 
            if (l == r)  return hist[l]; 
        
            // Find index of the minimum value in given range 
            // This takes O(Logn)time 
            int m = RMQ(hist, st, n, l, r); 
        
            /* Return maximum of following three possible cases 
            a) Maximum area in Left of min value (not including the min) 
            a) Maximum area in right of min value (not including the min) 
            c) Maximum area including min */
            return max(getMaxAreaRec(hist, st, n, l, m-1), 
                    getMaxAreaRec(hist, st, n, m+1, r), 
                    (r-l+1)*(hist[m]) ); 
        } 
        
        // The main function to find max area 
        int getMaxArea(int hist[], int n) 
        { 
            // Build segment tree from given array. This takes 
            // O(n) time 
            int *st = constructST(hist, n); 
        
            // Use recursive utility function to find the 
            // maximum area 
            return getMaxAreaRec(hist, st, n, 0, n-1); 
        } 
        
        // Driver program to test above functions 
        int main() 
        { 
            int hist[] =  {6, 1, 5, 4, 5, 2, 6}; 
            int n = sizeof(hist)/sizeof(hist[0]); 
            cout << "Maximum area is " << getMaxArea(hist, n); 
            return 0; 
        } 

-94.7) Largest Rectangle in Histogram with Divide and Conquer
    NO SEGMENT TREE: NLOGN worst case:


    The idea is simple: for a given range of bars, the maximum 
    area can either from left or right half of the bars, or from 
    the area containing the middle two bars. For the last condition, 
    expanding from the middle two bars to find a maximum area is O(n), 
    which makes a typical Divide and Conquer solution with 
    T(n) = 2T(n/2) + O(n). Thus the overall complexity is O(nlgn) 
    for time and O(1) for space (or O(lgn) considering stack usage).

    Following is the code accepted with 44ms. I posted this because 
    I didn't find a similar solution, but only 
    the RMQ idea which seemed less straightforward to me.

    class Solution {
        int maxCombineArea(const vector<int> &height, int s, int m, int e) {
            // Expand from the middle to find the max area containing height[m] and height[m+1]
            int i = m, j = m+1;
            int area = 0, h = min(height[i], height[j]);
            while(i >= s && j <= e) {
                h = min(h, min(height[i], height[j]));
                area = max(area, (j-i+1) * h);
                if (i == s) {
                    ++j;
                }
                else if (j == e) {
                    --i;
                }
                else {
                    // if both sides have not reached the boundary,
                    // compare the outer bars and expand towards the bigger side
                    if (height[i-1] > height[j+1]) {
                        --i;
                    }
                    else {
                        ++j;
                    }
                }
            }
            return area;
        }
        int maxArea(const vector<int> &height, int s, int e) {
            // if the range only contains one bar, return its height as area
            if (s == e) {
                return height[s];
            }
            // otherwise, divide & conquer, the max area must be among the following 3 values
            int m = s + (e-s)/2;
            // 1 - max area from left half
            int area = maxArea(height, s, m);
            // 2 - max area from right half
            area = max(area, maxArea(height, m+1, e));
            // 3 - max area across the middle
            area = max(area, maxCombineArea(height, s, m, e));
            return area;
        }
    public:
        int largestRectangleArea(vector<int> &height) {
            if (height.empty()) {
                return 0;
            }
            return maxArea(height, 0, height.size()-1);
        }
    };


-94.5) Largest Rectangle in Histogram with DP:

    To obtain the max area, we need to compare all the possible 
    areas based on each height value. To get the max area 
    for each height, we need to know the max widths, or the 
    min index from the left and the max index from the right 
    for each of the height. With that in mind, we can 
    come down to the following code.
    ALSO uses open close interval. 

    (below soln not really O(N))
    int largestRectangleArea(vector<int>& height) {
        int n = height.size(), ans = 0, p;
        vector<int> left(n,0), right(n,n);
        for (int i = 1;i < n;++i) {
            p = i-1;
            while (p >= 0 && height[i] <= height[p])
                p = left[p] - 1;
            left[i] = p + 1;
        }
        for (int i = n-2;i >= 0;--i) {
            p = i+1;
            while (p < n && height[i] <= height[p])
                p = right[p];
            right[i] = p;
        }
        for (int i = 0;i < n;++i)
            ans = max(ans,height[i]*(right[i]-left[i]));
        return ans;
    }

    Using stack ensures O(N) worst case runtime

    # O(n): using dp without stack 
    class Solution:
        def largestRectangleArea(self, heights: List[int]) -> int:
            if not heights:
                return 0

            size = len(heights)
            dp_l = [0] * size
            dp_r = [0] * size

            # find stop index from left
            stack = []
            for i in range(size):
                dp_l[i] = i
                while stack and heights[stack[-1]] >= heights[i]:
                    dp_l[i] = stack.pop()
                if dp_l[i] > 0 and heights[i] <= heights[dp_l[i] - 1]:
                    dp_l[i] = dp_l[dp_l[i]]
                stack.append(i)

            # find stop index from right
            stack = []
            for i in range(size - 1, -1, -1):
                dp_r[i] = i
                while stack and heights[stack[-1]] >= heights[i]:
                    dp_r[i] = stack.pop()
                if dp_r[i] < size - 1 and heights[i] <= heights[dp_r[i] + 1]:
                    dp_r[i] = dp_r[dp_r[i]]
                stack.append(i)

            # find max area
            area = 0
            for i in range(size):
                area = max(area, heights[i] * (dp_r[i] - dp_l[i] + 1))

            return area


-94) MONOTONIC STACK + Largest rectangle in histogram

    Given n non-negative integers representing the histogram's bar 
    height where the width of each bar is 1, 
    find the area of largest rectangle in the histogram.

    Above is a histogram where width of each bar is 1, 
    given height = [2,1,5,6,2,3].

    The largest rectangle is shown in the 
    shaded area, which has area = 10 unit.
    Intuition (LOOK AT PROBLEM CONSTRAINTS):

    Why could there be a better solution than O(n^2) ? 
    How would we know that ?

    Because if the length of the array is n, the largest possible 
    rectangle has to have a height one of the elements of the array, 
    that is to say, there are only n possible largest rectangles. 
    So we don't really need to go through every pair of bars, 
    but should rather search by the height of the bar.

    Why Stack?
    At each step we need the information of previously seen 
    "candidate" bars - bars which give us hope. These are the bars 
    of increasing heights. And since they'll need to be put in the 
    order of their occurence, stack should come to your mind.

    MONOTONIC STACK CONCEPTS:

    ## Similar to Leetcode: 907. Sum Of Subarray Minimums ##
    ## Similar to Leetcode: 85. maximum Rectangle ##
    ## Similar to Leetcode: 402. Remove K Digits ##
    ## Similar to Leetcode: 456. 132 Pattern ##
    ## Similar to Leetcode: 1063. Number Of Valid Subarrays ##
    ## Similar to Leetcode: 739. Daily Temperatures ##
    ## Similar to Leetcode: 1019. Next Greater Node In LinkedList ##

    LOGIC:
    ## 1. Before Solving this problem, go through Monotone stack.
    ## 2. Using Monotone Stack we can solve 
    1) Next Greater Element 2) Next Smaller Element 
    2) Prev Greater Element 4)Prev Smaller Element
    ## 3. Using 'NSE' Monotone Stack concept, we can find width of rectangles, 
    height obviously will be the minimum ofthose. Thus we can calculate the area
    ## 4. As we are using NSE concept, adding 0 to the end, will make 
    sure that stack is EMPTY at the end. ( so all theareas can be calculated while popping )
        


    META STRATEGY -> when algo becomes complicated/
    has disjoint parts and you have to coordinate how disjointparts work
    you are PROBABLY DOING IT WRONG. ITS A SMELL. wierd algoconstructions 
    are usually wrong.         
    */
            
    struct Pair {
            int h;
            int len;
            Pair(int h, int len): h(h), len(len) {}
        
    };
        
    int largestRectangleArea(vector<int>& heights) {
            
            stack<Pair> st; 
            int widthFromRight;
            int currArea = 0;
            
            for(auto it = heights.begin(); it != heights.end(); it++) {    
                int h = *it;            
                int widthFromRight = 0;
                while(st.size() > 0 && st.top().h >= h) {
                    // start popping. 
                    // the elements we pop actually form an increasing sequence
                    // and you can get max area of increasing rectangles 
                    // easily by accumulating!
                    auto [h2, w] = st.top();
                    widthFromRight += w;
                    currArea = max(widthFromRight*h2, currArea);
                    st.pop();
                } 
                
                // multiply width and 
                // height and compare with curr max. 
                // the below line can be erased, and algo would still work
                // because this shit gets computed anyway in the while loop below
                currArea = max(h*(1+widthFromRight), currArea);
                st.push(Pair(h, (1+widthFromRight)));
            }
            // elements left in stack are what?
            // -> elements that form an increasing sequence 1, 2, 3
            widthFromRight = 0;
            while(st.size() > 0) { 
                auto [h, w] = st.top();
                widthFromRight += w;
                currArea = max(widthFromRight*h, currArea);
                st.pop();
            }
            return currArea;  
        }
    };


    // More improved solutions:

    /*
    Explanation: As we check each height, we see if it is less than 
    any height we've seen so far. If we've seen a larger height in 
    our stack, we check and see what area that rectangle could have 
    given us and pop it from the stack. We can continue this approach 
    for each for each rectangle: finding the max area of any larger 
    rectangle previously seen before adding this rectangle to the stack 
    and thus limiting the height of any possible rectangle after.
    */

    int largestRectangleArea(vector<int>& heights) {
            if(heights.size() == 0) return 0;
            
            stack<int> s;
            int area = 0;
            
            for(int i = 0; i <= heights.size(); i++){
                while(!s.empty() && (i == heights.size() || heights[s.top()] > heights[i])){
                    int height = heights[s.top()];
                    s.pop();
                    int width = (!s.empty()) ? i - s.top() -1 : i;
                    area = max(area, height * width);
                }
                s.push(i);
            }
            return area;
    }

    Another way: This solution works because a dummy 0 was inserted into vector.
    JUST AS IN LINKED LISTS, USE DUMMY NODES TO SIMPLIFY, AND REDUCE 2 FOR LOOPS TO 1:

        int largestRectangleArea(vector<int>& height) {
            height.push_back(0);
            const int size_h = height.size();
            stack<int> stk;
            int i = 0, max_a = 0;
            while (i < size_h) {
                if (stk.empty() || height[i] >= height[stk.top()]) stk.push(i++);
                else {
                    int h = stk.top();
                    stk.pop();
                    max_a = max(max_a, height[h] * (stk.empty() ? i : i - stk.top() - 1));
                }
            }
            return max_a;
        }





-93) Bloomberg BINARY SEARCHING ON AN OBJECTIVE QUESTION. 
     Calculate amt you can take out monthly that leads to balance 0
    when balance also gets interest rate of 6%.

    Just make guesses from 0 to totalamt     through binary searching,
    and then check through math formula if 
    taking out that specific monthly amount X will 
    lead to balance of 0
    when person dies. 

    When do math approximations -> TRY BINARY SEARCH. 


-92) HOW TO SORT PARTIALLY UNSORTED ARRAYS: ( SPLIT + JOIN)
    Round 1: Given a sorted n-size array, there are k elements 
    have been changed i.e. [1, 3, 5, 6, 4, 2, 12] (it might be changed from [1, 3, 5, 6, 7, 8, 12] with k = 2). Important to know is that k is unknown and k is much smaller than n. 
    The task is to re-sort the entire array.
    The interviewer wants O(n) solution. I bombed this one. In the end, the 
    interviewer kind of fed the solution to me. What he suggested: 
    a. break the array into two: one sorted array and one unsorted array e.g. [1, 3, 5, 12] 
    and [6, 4, 2]. This takes O(n) 
    b. Sort the unsorted array. This takes O(klogk) 
    c. Merge two sorted arrays. This takes O(n). Because k is very small, so in the end O(n) + O(klogk) ~= O(n).
        


-91) 
    Round 2
    You have two arrays of odd length (same length). 
    You should check if you can pair elements from both arrays such that 
    xor of each pair is the same.
    
    Ex : [a, b, c] and [d, e, f] check we can find a pairing 
         say (arrays actually have integers)
    a xor e = v
    b xor d = v
    c xor f = v

    SOLUTION:
    O(N). XOR all => v. we can take advantage of the property that a^b=v => a^v=>b 
        and use a hashset.


+-88) DFA AND DP PROBLEM!!!!
 
      You are a traveling salesperson who travels 
+      back and forth between two cities (A and B). 
+      You are given a pair of arrays (revA and revB) of length n.
+
+    You can only sell goods in one city per day.
+    At the end of each day you can choose to travel to another 
+    city but it will cost a constant amount of money (travelCost).
+
+    Ex::
+    revA[] = {3, 7,2,100};
+
+    revB[] = {1,1,1,10};
+
+    travelCost = 2;
+    Find maximum profit.
+        int MaxProfitBySalesMan ( int arr1 [] , int arr2 [] , int n )
+        {
+            int dp [ 2 ] [ n ] ; 
+            dp [ 0 ] [ 0 ]  = arr1 [ 0 ] ; 
+            dp [ 1 ] [ 0 ]  = arr2 [ 0 ] ;
+            for ( int i = 1 ; i < n ; i ++ )
+            {
+                dp [ 0 ] [ i ] = max ( dp [ 0 ] [ i -  1 ] , dp [ 1 ][ i  - 1 ] - 2  ) + arr1 [ i ]  ; 
+                dp [ 1 ] [ i ] = max ( dp [ 1 ] [ i -  1 ] , dp [ 0 ][ i  - 1 ] - 2  ) + arr2 [ i ]  ;
+            }
+            return max ( dp [ 0] [ n - 1 ] , dp [ 1 ] [ n - 1 ] ) ;
+        }
+
+
+
+-87)Optimizing binary tree questions with bottom up DP: 
+    One way to optimize these questions is to use post-order traversal.
+    Compute the value for the children then compute for parent sorta like DP:
+
+    1.   Count Univalue Subtrees
+    中文English
+    Given a binary tree, count the number of uni-value subtrees.
+    
+    A Uni-value subtree means all nodes of the subtree have the same value.
+    
+    Example
+    Example1
+    
+    Input:  root = {5,1,5,5,5,#,5}
+    Output: 4
+    Explanation:
+                  5
+                 / \
+                1   5
+               / \   \
+              5   5   5
+    Example2
+    
+    Input:  root = {1,3,2,4,5,#,6}
+    Output: 3
+    Explanation:
+                  1
+                 / \
+                3   2
+               / \   \
+              4   5   6
+
+    Solution:
+    def countUnivalSubtrees(self, root):
+        count = 0
+        def helper(node):
+            nonlocal count 
+            if node is None:
+                return None
+            left_result = helper(node.left)
+            right_result = helper(node.right)
+            if left_result == False:
+                return False
+            if right_result == False:
+                return False
+            if left_result and left_result != node.val:
+                return False
+            if right_result and right_result != node.val:
+                return False
+            count += 1
+            return node.val
+        helper(root)
+        return count
+


-90) USING TREESETS AND TREE MAPS C++

    Round 1
    Design a data structure with two operations 1. addRange(int start, int end) 
    and 2. contains(int point)
    Here range means an interval, so the data structure contains information 
    of all ranges added uptil that point and you can have interleaved queries 
    of the form contains(int point) which returns true if the point 
    is contained in any of the ranges or false otherwise.

    The solution I thought of was to store the ranges/intervals as a 
    list of sorted disjoint intervals (i.e merge overlapping intervals). 
    Now when we get a contains query, we can perform binary 
    search(interval with start point equal to or less than point) 
    to find a potential interval that may contain it. addRange would 
    be O(n) and contains would be O(logn). I believe there is a better 
    solution with interval trees but interviewer said this solution 
    was okay which I ended up coding.

    USE sortedcontainers in python or
    Q1:
    Use a treemap => amortized O(logn) merge and O(logn) contains.
    
    STL map is inherently a binary search tree - just use map::find. 
    Using container member functions, where they are present, 
    is preferable to algorithms.

    How to find all elements in a range in STL map(set)

    If you are using std::map, it's already sorted, your 
    can use lower_bound/upper_bound an example from cplusplus.com:

    // map::lower_bound/upper_bound
    #include <iostream>
    #include <map>

    int main ()
    {
        std::map<char,int> mymap;
        std::map<char,int>::iterator itlow,itup;

        mymap['a']=20;
        mymap['b']=40;
        mymap['c']=60;
        mymap['d']=80;
        mymap['e']=100;

        itlow=mymap.lower_bound ('b');  // itlow points to b
        itup=mymap.upper_bound ('d');   // itup points to e (not d!)

        mymap.erase(itlow,itup);        // erases [itlow,itup)

        // print content:
        for (std::map<char,int>::iterator it=mymap.begin(); it!=mymap.end(); ++it)
            std::cout << it->first << " => " << it->second << '\n';

        return 0;
    }


    
-89) Round 1:

    In this round i was asked a constructive problem. It goes like this:
    Let's say we have a permutation P of length n(n = 5 here) = [3, 5, 1, 4, 2]
    Now we delete elements from this permutation P from 1 to n in order and write their index to
    another array Q. When an element is deleted, remaining elements are shifted to left by 1.
    Initial: P = [3, 5, 1, 4, 2], Q = []
    delete 1, P = [3, 5, 4, 2], Q = [3] (index of 1 was 3 so write 3(bcz it's index of 1) in Q)
    delete 2, P = [3, 5, 4], Q = [3, 4]
    delete 1, P = [5, 4], Q = [3,4,1]
    delete 1, P = [5], Q = [3, 4, 1, 2]
    delete 1, P = [], Q = [3, 4, 1, 2, 1]

    Now given Q, we have to restore P.

    I gave a Nlog^N solution using fenwick tree and binary search.
    He asked me a follow up in which i have to optimize space.

    How to use fenwick tree?
    



-87)Optimizing binary tree questions with bottom up DP: 
    One way to optimize these questions is to use post-order traversal.
    Compute the value for the children then compute for parent sorta like DP:

    1.   Count Univalue Subtrees
    中文English
    Given a binary tree, count the number of uni-value subtrees.
    
    A Uni-value subtree means all nodes of the subtree have the same value.
    
    Example
    Example1
    
    Input:  root = {5,1,5,5,5,#,5}
    Output: 4
    Explanation:
                  5
                 / \
                1   5
               / \   \
              5   5   5
    Example2
    
    Input:  root = {1,3,2,4,5,#,6}
    Output: 3
    Explanation:
                  1
                 / \
                3   2
               / \   \
              4   5   6

    Solution:
    def countUnivalSubtrees(self, root):
        count = 0
        def helper(node):
            nonlocal count 
            if node is None:
                return None
            left_result = helper(node.left)
            right_result = helper(node.right)
            if left_result == False:
                return False
            if right_result == False:
                return False
            if left_result and left_result != node.val:
                return False
            if right_result and right_result != node.val:
                return False
            count += 1
            return node.val
        helper(root)
        return count




-86) monotonic stack vs monotonic queue and how to build a monotonic structure
        LOOK AT HRT PROBLEM.
        


-85) think of the algo to do citadel problem -> round robin ALGORITHM!!!


-84) Using cumulative array for sums in 1D and 2D case tricks:
    1D) sum between i and j inclsuive:
        sum(j) - sum(i-1)
        REMEMBER TO DO I-1 to make it INCLUSIVE!

    2D)
    Have a 2D cumulative array,
    of size N+1, M+1, for NxM array
    top row is all 0s.
    left column is all 0s.
    similar to cumualtive array. 
    
    2 coordinates is top left and bottom right. 
    
    (from snap interview)
    SUM OF LARGE RECTANGE - SUM OF TOP RIGHT - SUM OF BOTTOM LEFT + SUM OF SMALL RECTANGLE. 
    


    topleft -> tlx, tly
    bottomright -> brx, bry
    
    # because inclusive, not sure though, do lc to check below part.
    tlx -= 1
    tly -= 1

    arr[brx][bry] - arr[brx][tly] - arr[tlx][bry]  + arr[tlx][tly]




-83) Fenwick Trees youtube video explanation


    Fenwick Tree youtube video ideas: 

    1, -7, 15, 9, 4, 2, 0, 10
    We keep grouping elements, first by 2, then by 4, then by 8
 
    1, -7, 15, 9, 4, 2, 0, 10
       -6, 24, 6, 10
          18,   16
              34
    
    Make fenwick tree indexed by 1 -> whatever lenght array we get, make fenwick array 1 size bigger.
    8 blocks -> size 9 array. 
    
    INITIALIZATION: 
    4 bits to represent 9 blocks. 
    First node is always dummy node in fenwick tree, and represents 0.  
    
    Explained:
    No number between 1 and 2 -> is there a number between 2 and 4 -> need 2 bits to represent 3,
    so it goes into level 2
    Between 4 and 8 are there numbers -> with 2 bits we can represent 
                                        5 and 6 so they go to level 2.
    We still have 7 and we need 3 bits to represent 7 -> so it goes to level 3.


    Level 0 (use no bits)                                  0000
    Level 1 (use 1 bit)          (1) 0001, (2)0010,       (4)0100,                    (8)1000
    Leve  2 (use 2 bits)                           (3)0011         5(0101) 6(0110)
    Level 3 (use 3 bits)                                                        7(0111)

    Parent relationships of tree are following: 
    Parent of 3 is 2,  
    parent of 8 is 0, 
    parent of 7 is 6, 
    parent of 6 is 4, parent of 5 is 4
    parent of 3 is 2
    parent of 1 is 0. 
    parent of 4 is 0
    parent of 2 is 0
    To go from number to parent -> remove the rightmost 1 or the rightmost 1 bit. 

    so parent = i - (i & -i)
        i is 7 -> parent is 6
        7                -> 0111
        -7 2s complement -> 1001
                        and it -> 1
        7 - 1 = 6
        So that trick works -> parent = i - (i & -i)

    We need to cumulatively sum everything in the array and put in fenwick tree. 
    So for:
    1, -7, 15, 9, 4, 2, 0, 10
    
    Fenwick array: 0, 1, -6, 9, 18, 22, 24, 24, 34
    OK finally, 
    you have to subtract the parent from the child in above fenwick array. 
    From 34 you subtract 0. 
    From 24 you subtract 24
    from 24 you subtract 18
    from 22 subtract 18
    from 9 subtract -6
    from 1 subtract 0. 

    Use the trick of parent. 
    Final fenwick array: 
    0, 1, -6, 15, 18, 4, 6, 0, 34

    To get sum from idx 0 to 6: -> its 24. 
    fenwick tree indexed by +1, so to get indx 0-6, need to do 0 to 7 
    from 7 go up the tree -> 0 + 6 + 18 + 0 -> 24

    INITIALIZE FENWICK TREE IN O(N) time:

    int n = length of array
    int[] fw = new int[n+1];
    fw[1] = arr[0];
    for(int i = 1; i < n; i++) {
        fw[i+1] = fw[i] + arr[i];
    }

    // now remove value of parent node from given node. 
    for(int i = n; i >0 ; i--) {
        parent = i - (i& -i);
        if (parent >= 0) {
            fw[i] -= fw[parent];
        }
    }


    SUM OPERATION (LOGN operation):
    Fenwick incremented by 1, so always add 1 before you start. 
    int sum(int x) {
        x ++;
        int res = 0; 
        while(x > 0) {
            res += fw[x];
            x = x - (x & -x)
        } 
    }

    //Increment (LOG N)
    // i is index, v is val.
    // go top down tree 
    // find parent -> do opposite to findnext node. 
    void increment(int i, int val) {
        i ++; //fenwick tree do + 1
        while(i <= n ) {
            fw[i] += val;
            // find next node, not parent this time. 
            i = i + (i&-i)
        } 
    }

    LEETCODE 307 Range Sum Query: 

    sumRange(int i, int j) {
        // want sum inclusive.
        // its i-1 because we want INCLUSIVE SUM, 
        // THIS IS TRUE FOR 1D GRIDS AND 2D GRID SUMS.  
        sum(j) - sum(i-1)

    }
    
    void update(int i, int val) {
        //change ith location to val
        int diff = val - arr[i];
        arr[i] = val;
        increment(i, diff);
    }



-82)Bit Tries and solving XOR problems:
    BIT TRIES -> stores a number in binary form. 
    #######################################################################33
    1 - Given an array of integers, we have to find 2 elements whose XOR is maximum

    BIT TRIE OPERATIONS: 
        2 types of queries for datastructure:
        1. insert a number in data structure.
        2. Given y, find maximum xor of y with all elements that have been inserted till now. 

    insert(1) int variable consists of 32 bits
    but lets think of number as 4 bits

    Insert EXAMPLES:
        1 is 0001 -> we have to insert in trie. 

        Right now trie is empty. 
        if 0 occurs move left, 1, move right. 

          Insert 1 
          since 0, go left from root node 
          0 - left
          1 - right
  
               /
              /
             /
             \
      
          Insert 2:
          0010
               /
              /
             / \
             \ /
             
          Now insert 3: 0011
                /
               /
              / \ 
              \ /\
    
    FIND QUERY:
        find max xor with all elements inserted: 
        Y: b1, b2, b3, b4, b5, where b1, b2, b3... are bits
        if b1 is 0, so we are going to find number in trie whose MSB is 1.
        if not, then just go other side.
        
        if b1 is 1, we are going to find number in trie whose most sig bit is 0
        if not just go other side. 

        Do this for every query. 


    #include <bits/stdc++.h>
    using namespace std;


    class Node {
        public:
        child[2];
        Node() {
            childe[0] = NULL;
            child[1] = NULL;
        }

    }

    // Time Complexity -> N * log2(max(A[i]) )
    // Worst case make NlogN nodes -> so space complexity NlogN
    void insert(Node * root, int x) {
        // contains 32 bits.    
        // start from MSB and insert. 

        for(int i = 32; i >= 0; i --) {
            int bit = ((x >> i) & 1);
            if(root->child[bit] == NULL) {
                root->child[bit] = new Node();
            }
            root = root->child[bit];
        }
    }

    // Time Complexity -> N * log2(max(A[i]) )
    int maXxor(Node * root, int y) {

        int ret = 0;
        for(int i = 32; i >= 0; --i){

            int bit = ((y>>i) & 1);
            if(root->child[bit^1] == NULL) {
                // if opposite bit not present,
                // simply move where you can.
                root = root->child[bit];
            } else {
                //if opposite bit present. 
                // calculate its contribution. 
                //The LL makes the integer literal of type long long.
                //So 2LL, is a 2 of type long long.
                //Without the LL, the literal would only be of type int.
                //This matters when you're doing stuff like this:
                //1   << 40
                //1LL << 40
                // 0^1 = 1; 2**i contribution
                ret |= (1LL << i);
                root = root->child[bit^1];
            }
            return ret;
        }
    }


    int32_t main() {
        int n;
        cin >> n;
        vector<int> arr(n);
        Node * root = Node();
        int ans = 0;

        // Time Complexity -> N * log2(max(A[i]) )
        // Insertion N times, search N times. 

        // below is cool trick because we insert, and then search 
        // at the same time all in one for loop.
        // and we search on an bit trie that is building up each iteration. 
        // very cool!

        for(int i = 0; i < n; i++) {
            cin >> arr[i]
            int y = arr[i];
            insert(Node, arr[i]);
            int temp = maXxor(root, y);
            ans = max(ans, temp);
        }
        cout << ans << endl;
    }

    #######################################################################33



-80) Union Find VS DFS for finding connected components pros/cons:

    '''
    UNION FIND VS DFS FOR CONNECTED COMPONENTS: 

    The union-find algorithm is best suited for situations 
    where the equivalence relationship is changing, i.e., there are 
    "Union" operations which need to be performed on your set of partitions. 
    Given a fixed undirected graph, you don't have the equivalence relationships 
    changing at all - the edges are all fixed. OTOH, if you have a graph with new 
    edges being added, DFS won't cut it. While DFS is asymptotically faster than 
    union-find, in practice, the likely deciding factor would be the 
    actual problem that you are trying to solve.

    tl;dr - Static graph? DFS! Dynamic graph? Union-find!

    when new edges being added or remoed -> O(1) union find

    If the graph is already in memory in adjacency list format, 
    then DFS is slightly simpler and faster (O(n) versus O(n alpha(n)), 
    where alpha(n) is inverse Ackermann), but union-find can handle the 
    edges arriving online in any order, which is sometimes useful 
    (e.g., there are too many to fit in main memory).

    RUNTIME ANALYSIS:

    According to CLRS,

    When the edges of the graph are static—not changing over
    Time—we can compute the connected components faster by using depth-first search.

    I tried to do some runtime analysis, and in a graph G(V,E) 
    on which we have to answer Q connectivity queries.

    Your running time analysis of Union-Find is incorrect. 
    To use Union-Find for this situation, we need to perform 
    V MakeSet operations, E Union operations, and 2Q Find operations. 
    Each operation takes O(α(V)) amortized time. Therefore, the total 
    running time for the Union-Find-based algorithm will be O(α(V)(V+E+Q)) ... not O(α(V)(E+Q)) 
    as you claimed. I suspect you forgot the cost of the MakeSet operations.

    It seems clear that O(α(V)(V+E+Q)) is asymptotically 
    slower than O(V+E+Q), or at least no faster.

    If you care, you can reduce the running time of DFS to O(E+Q) 
    and the running time of the Union-Find algorithm to O(α(V)(E+Q)). 
    Basically, you first scan the graph to remove all isolated vertices 
    (vertices with no edges incident on them). You know that the 
    isolated vertices aren't connected to anything, so they can be ignored. 
    Then, run your algorithm on the resulting graph, after removal 
    of isolated vertices. For the resulting graph, we have E/2≤V≤E, 
    so V=Θ(Q) and O(V+E+Q)=O(E+Q) and O(α(V)(V+E+Q))=O(α(V)(E+Q)).

    '''

-79) UNION FIND VS DFS Question -> Swap lexiographical order

    Given a string str and array of pairs that indicates 
    which indices in the string can be swapped, return the 
    lexicographically largest string that results from doing the 
    allowed swaps. You can swap indices any number of times.

    Example

    For str = "abdc" and pairs = [[1, 4], [3, 4]], the output should be
    swapLexOrder(str, pairs) = "dbca".

    By swapping the given indices, you get the strings: "cbda", "cbad", 
    "dbac", "dbca". The lexicographically largest string in this list is "dbca".


    from collections import defaultdict

    def swapLexOrder(s, pairs):
        '''
        Taking advantage of transitivity relationship using union find:
        find all connected components aka indexes that can be swaped with other indexes. 
        if a swaps with b and b swaps with c then a swaps with c
        as you process each pair, do union find.
        do simpler version store parent of each node 
        with parents map.
        '''
            
        '''
        UNION BY RANK + PATH COMPRESSION
        '''
        parents = {}
        rank = {}
        
        # make set
        for i in range(len(s)):
            parents[i] = i
            rank[i] = 0
            
        def find(item):
            # do path compression 
            if item == parents[item]:
                return item
                
            parents[item] = find(parents[item])
            return parents[item]
                
        def union(a, b):
            # unions. 
            parent_a = find(a)
            parent_b = find(b)
            if parent_a == parent_b:
                return 
                
            if rank[parent_a] > rank[parent_b]:
                parent_a, parent_b = parent_b, parent_a

            parents[parent_a] = parent_b        
            if rank[parent_a] == rank[parent_b]:
                rank[parent_b] += 1
        
        for p in pairs: 
            idx1, idx2 = p
            union(idx1-1, idx2-1)
                
        comp = defaultdict(list)
        
        for i in range(len(s)):
            parent = find(i)
            comp[parent].append(i)
        
        # ok go through each comp, sort the chars in reverse!
        res = ["" for i in range(len(s))]
        
        for k in comp.keys():
            indices = comp[k]
            chars = sorted([s[i] for i in indices], reverse=True)
            # then sort indices?
            
            locs = zip(chars, sorted(indices))
            
            for c, i in locs:
                res[i] = c
        
        return "".join(res)




-75) ORDERED SETS vs PRIORTIY QUEUES (Python does not have ordered set aka bst)

    Since both std::priority_queue and std::set (and std::multiset) are data 
    containers that store elements and allow you to access them in an ordered 
    fashion, and have same insertion complexity O(log n), what are the advantages 
    of using one over the other (or, what kind of situations call for the one or the other?)?

    While I know that the underlying structures are different, I am not as much 
    interested in the difference in their implementation as I am in the comparison 
    their performance and suitability for various uses.

    Note: I know about the no-duplicates in a set. That's why I also mentioned std::multiset 
    since it has the exactly same behavior as the std::set but can be used 
    where the data stored is allowed to compare as equal elements. 
    So please, don't comment on single/multiple keys issue.

    The priority queue only offers access to the largest element, while the set 
    gives you a complete ordering of all elements. This weaker interface means that 
    implementations may be more efficient (e.g. you can store the actual queue 
    data in a vector, which may have better performance on account of its memory locality).


    A priority queue only gives you access to one element in sorted order -- i.e., 
    you can get the highest priority item, and when you remove that, you can get 
    the next highest priority, and so on. A priority queue also allows duplicate 
    elements, so it's more like a multiset than a set. [Edit: As @Tadeusz Kopec pointed out, 
    building a heap is also linear on the number of items in the heap, where building a 
    set is O(N log N) unless it's being built from a sequence that's already 
    ordered (in which case it is also linear).]

    A set allows you full access in sorted order, so you can, for example, find 
    two elements somewhere in the middle of the set, then traverse in order from one to the other.


-74)MULTI TIMESTAMP DP FINITE STATE MACHINE problems. Bottom up with decode ways:
    DECODE WAYS (REVIEW in most important)
    class Solution:
        def numDecodings(self, s):        
            # BOTTOM UP ONLY!
            '''        
            ADD UP ALL THE WAYS WE ARRIVED TO A STATE FROM OTHER STATES!!
            USE IF STATEMENTS TO DETECT IF WE CAN ARRIVE TO THE STATE. 
            
            OPT[i] = OPT[i-1]   (TAKE ONE ALWAYS POSSIBLE!)
                    + OPT[i-2]   (TAKE 2 MAY NOT BE POSSIBLE)
            
            s = "12"
            "0" does not map to anything -> can only be used with 10 or 20
            
            226
            2 -> 1  b
            22 -> 1 bb 
            2 26
            3 ways:
            2 2 6
            22 6
            2 26
            
            Base case empty string = 1?
            take 1 
            2 
            take 2:
            22 
            next timestamp?
            we take 1 
            
            OPT[i] -> all the ways to decode up to index i. 
            process index 0 -> only 1 way to decode unless its 0. 
            can take 2 charcters if OPT[i-1] exists. 
            
            In other words solution relies on 3 timesteps to build finite automata
            '''
            
            OPT = [0 for i in range(len(s) +1)]
            
            # BASE CASE
            OPT[0] = 1 
            prevCh = None
            seeNonezero = False
            
            # BTW its easy to optimize this to O(N) space, using 2 variables for 
            # previous 2 timestamps. 
            
            for idx in range(1, len(s)+1):
                # 0 cannot become anything!
                # take current character as single. 
                ch = int(s[idx-1])
                if ch != 0:
                    OPT[idx] += OPT[idx-1]    
                # only way we can use 0 is if we see prev.                         
                # if you see 2 zeros in a row you cant decode it -> answer is 0. 
                if prevCh != None: 
                    # take current character + prev char!
                    if (prevCh == 1 and ch < 10) or (prevCh == 2 and ch < 7):
                        OPT[idx] += OPT[idx-2]
                # loop end set prevCharacter
                prevCh = ch            
                
            return OPT[len(s)]    



-73) To solve weird optimization problems, write out the problem constraints in 
     mathematical notation!  
     + english words
     + greedy detection/hill finding/maximization/montonic queue/segment tree optimize/binary search/quick select   
     + As you find constraints -> think of edge cases -> can they be solved by setting the base case in a recurrence?
     + NEGATIVE SPACE. 
     + or through other means? 
    
    1.    Remove Covered Intervals: 

    Given a list of intervals, remove all intervals that are covered by another interval in the list.

    Interval [a,b) is covered by interval [c,d) if and only if c <= a and b <= d.

    After doing so, return the number of remaining intervals.
    
    Example 1:

    Input: intervals = [[1,4],[3,6],[2,8]]
    Output: 2
    Explanation: Interval [3,6] is covered by [2,8], therefore it is removed.
    Example 2:

    Input: intervals = [[1,2],[1,4],[3,4]]
    Output: 1
    class Solution:
        def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
            '''
            Sort by start time. 
            
            Then check if interval after me I cover. If i do remove, 
            otherwise keep. 
            
            sort intervals by start time.          
            
            when checking next interval, make sure its start time is after 
            the max finish time, otherwise covered. 
            
            Compare end time of first one with endtime of second one, if its within its covered.
            OTHERWISE!!,
            
            if you get an interval with further endtime -> update the FOCUS interval to that. 
            because it will be able to cover more on the right. 
            So keep the one with largest end time asthe main focus. 
            
            Requires sorting on start time  -> NLOGN.
            
            need to do 2 sorts, 
            earliest start time,
            then for those with same start time -> latest finish time comes first. 
            So then the latest finish times can consume the earlier finish times and be used to consume intervals
            without same start time. 
            '''
            
            # DO A DOUBLE SORT -> MAJOR SORT ON START -> MINOR SORT ON FINISH TIME. 
            intervals.sort(key=lambda x: (x[0], -x[1]))
            
            curr_fin = intervals[0][1]
            
            covered_count = 0
            for i in range(1, len(intervals)):
                nxt_fin = intervals[i][1]

                if nxt_fin <= curr_fin:
                    covered_count += 1
                else:
                    curr_fin = nxt_fin
                    
            return len(intervals) - covered_count







-72) Getting tripped up by doordash challanege:
     WHEN TO USE DP VS SORTING+GREEDY+MAXIMALMUNCH+HEAPQ SOLUTION/HILLFINDING. 

    Question: 
    So given drivers, they have speed, and professionalism such as:
    5 drivers and [100, 12, 4, 23, 5], [20, 30, 50, 12, 12]

    We can only select a maximum of 3 drivers and a minium of 1. 
    We want to get the max quality of a set of drivers 
    where quality = (Sum of drivers speed ) * min(of all drivers professionalism)

    What we can do is sort by professionals, so we can always take the best drivers first
    and then the worse drivers next. And we should take maximal because its sum for the speed.
    HOWEVER WE ARE LIMITED IN THE NUMBER OF MAX DRIVERS WE CAN TAKE.
    TO DEAL WITH THAT USE A HEAPQ TO STORE THE CURRENT SUM OF DRIVERS, AND POP THE SMALLEST
    VALUE DRIVER FROM SET OF DRIVERS.
    and keep track of max sum as you do this. 
    

    Initially i tried to solve with DP because thinking through the problem was hard.
    You should know when to use dp and when not too. YOU MUST START BY EXPLOITING PROBLEM 
    STRUCTURE AND THINKING GREEDY IN ALL DIRECTONS. Then when you figure that out, 
    TRY HILL FINDING WITH A DEQUE/HEAPQ/SEGMENT TREE. 

    THAT IS THE WAY!!!!
    SOLUTION BELOW: 


    from heapq import heappush, heappop

    def maximumTeamQuality(speed, professionalism, maxDashers):
        '''
        sort by professionalism.
        
        keep taking ppl, and update max?
        as we reduce professionalism -> speed increases. 
        and we should always take everyone thats more professional
        '''  
        
        zipped = sorted(zip(professionalism, speed), key=lambda x: x[0], reverse=True)
                
        # pop the lowest sum element each time when we go over maxDashers!
        # since the professionalism doesnt matter for the chosen ones if we choosing lower
        # professionalism one. 
        # need minheap.
        
        maxQ = 0
        curr_sum = 0
        h = []
        
        for p, s in zipped: 
            curr_sum += s
            # check
            if len(h) == maxDashers:
                smallest = heappop(h)
                curr_sum -= smallest[0]
                
            heappush(h, [s])    

            maxQ = max(maxQ, curr_sum*p)
        return maxQ


        REVIEW LEETCODE PROBLEM 1383 MAXIMUM PERFORMANCE OF TEAM ITS SIMILAR TO THIS FOR BETTER CONTEXT.




-71) Interval Problem Type Intuition, and Line Sweeping
     Try 2 pointer! or heap!
     
    1.   Interval List Intersections
    Given two lists of closed intervals, each list of 
    intervals is pairwise disjoint and in sorted order.

    Return the intersection of these two interval lists.

    (Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  The intersection of two closed intervals is a set of real numbers that is either empty, or can be represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

    Example 1:
    Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
    Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

    2 POINTER SOLUTION OPTIMAL: 
        def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            '''
            FOR INTERVAL QUESTIONS RMBR:
            SORTS FOR INTERSECTIONS USUALLY BEST TO SORT BY START TIME.
            
            BUT WHENEVER WE ARE CHECKING TO REMOVE INTERVALS FROM THE ACTIVE REGION!
            REMOVE BASED ON EARLIEST FINISH TIME, RATHER THAN OTHER METRICS:
            
            SOLUTION IS 2 POINTERS:
            
            ANOTHER INTERSECTION HINT: 
            INTERSECTIONS HAVE THE FORM OF THE FOLLOWING:
            [max(Astart, Bstart), min(Aend, Bend)]
            -> intersection exists if above computation is a real interval!
                (aka has positive length)        
            '''
            
            i = 0
            j = 0
            res = []
            if len(A) == 0 or len(B) == 0:
                return []
            
            '''
            You dont move pointers based on what the next earlier one was
            but the one that finished earlier, 
            because that one can no longer intersect with anything!
            '''
            while i < len(A) and j < len(B):
                
                a_start, a_end = A[i]
                b_start, b_end = B[j]
                
                pos_int_s = max(a_start, b_start)
                pos_int_e = min(a_end, b_end)
                if pos_int_s <= pos_int_e:
                    res.append([pos_int_s, pos_int_e])
                
                if a_end < b_end:
                    i += 1
                else:
                    j += 1 
            return res


    LINE SWEEPING SOLUTION:
        Remember when things are presorted, then line sweeep is not 
        optimal because it requires sorting.
        O(NLOGN) so it is slower than sorted variant!


        Like every interval problem, this can be solved by line sweep. 
        Note that, if the input is already pre-sorted, this 
        isn't an optimal solution. But it is cool and interesting.

        The general idea here is that we have a window, keyed by 
        person A or person B. When we add/remove intervals, we just need 
        to be careful that we're extending any existing window values, 
        if that person already exists in our window.

        Time: O(n log n)
        Space: O(n)

        class Solution:
            def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
                events = []
                OPEN, CLOSE = 0, 1
                
                for start, end in A:
                    events.append((start, OPEN, end, "A"))
                    events.append((end, CLOSE, end, "A"))
                    
                for start, end in B:
                    events.append((start, OPEN, end, "B"))
                    events.append((end, CLOSE, end, "B"))
                
                events.sort()
                window = {}
                ans = []
                
                for time, event_type, end, key in events:
                    if event_type == OPEN:
                        window[key] = (time, end)
                    else:
                        del window[key]

                    if len(window) > 1:
                        start_a, end_a = window["A"]
                        start_b, end_b = window["B"]
                        
                        best_start = max(start_a, start_b)
                        best_end = min(end_a, end_b)
                        ans.append((best_start, best_end))
                        
                return ans


    LINE SWEEPING second method.

        class Solution {
            public int[][] intervalIntersection(int[][] A, int[][] B) {
                List<int[]> res = new ArrayList<>();
                int n = A.length + B.length;
                int[] start = new int[n], end = new int[n];
                int c = 0;
                for (int i = 0;  i < A.length; ++i) {
                    start[c] = A[i][0];
                    end[c++] = A[i][1];
                }
                
                for (int i = 0;  i < B.length; ++i) {
                    start[c] = B[i][0];
                    end[c++] = B[i][1];
                }
                
                // O(n log (n))
                Arrays.sort(start);
                Arrays.sort(end);
                
                /**
                line sweep : go from left to right, stopping at start or end intervals.
                if its a start the increment busy.
                if its an end, decrement busy. before that check if 2 intervals are busy at the moment, if they are, it means A and B are indulged in the period end[e] and start[s - 1].
                note that end[e - 1] < start[s - 1] if busy = 2 else busy < 2, so the interval of interest is start[s - 1] and end[e]. Also s cannot be 0.
                */
                int s = 0, e = 0, busy = 0;
                while( e < n ) {
                if (s < n && start[s] <= end[e]) {
                    busy++;
                    ++s;
                } else {
                    if (busy == 2) {
                        res.add(new int[] {start[s - 1], end[e]});
                    }
                    busy--;
                    ++e;
                }
                }
                
                return res.toArray(new int[0][0]); 
            }
        }




-70) Example of extracting dynamic programming traversal paths 
     after doing DP problem.
        '''
        CombinationSum:

        Given an array of integers a and an integer sum, find all of the 
        unique combinations in a that add up to sum.
        The same number from a can be used an unlimited number of times in 
        a combination.
        Elements in a combination (a1 a2 … ak) must be sorted in non-descending order, 
        while the combinations themselves must be sorted in ascending order.

        If there are no possible combinations that add up to sum, 
        the output should be the string "Empty".

        Example

        For a = [2, 3, 5, 9] and sum = 9, the output should be
        combinationSum(a, sum) = "(2 2 2 3)(2 2 5)(3 3 3)(9)".


        The DP problem is simple, done previously before HARMAN!!

        Here we try to return the paths themselves, that were traversed in the DP
        2 ways to do so:
        A parents map OR as we save our results in the DP array, we also save our paths in a DP paths array.
        Look at both methods and learn!!

        '''
        from collections import defaultdict, deque
        def combinationSum(a, sum):
            # parents map? 
            g = defaultdict(list)
            
            # sort a and deduplicate. 
            
            a = sorted(list(set(a)))
            
            # we could also space optimize and just use One D array, because new 
            # index doesnt use just previous index, but all previous indexes.
            # so include all of em. 
            OPT = [[0 for i in range(sum+1)]]
            OPT[0][0] = 1
            
            
            dp_paths = [[] for i in range(sum+1)]
            dp_paths[0].append([])
            
            for idx, coinVal in enumerate(a):
                # to compute for current index, 
                # first copy previous, then operate on current. 
                curr = OPT[-1][:]
                '''
                idx, coin?
                '''
                for i in range(sum+1):
                    if i >= coinVal:
                        # do we specify the coin type we used??
                        # depends if we built from previous index, or 
                        # coins from this index.  -> cant you use difference in amts
                        # to determine coins -> YESS.
                        # you dont need to save coinVal
                        curr[i] += curr[i-coinVal]
                        # can we save it, as we build the dp?
                        
                        parent_paths = dp_paths[i-coinVal]
                        for p in parent_paths:
                            cp = p[::]
                            cp.append(coinVal)
                            dp_paths[i].append(cp)

                        if(curr[i-coinVal] > 0):
                            g[i].append(i-coinVal)
                                
                OPT.append(curr)
            
            # DP PATHS WORKS HOW YOU EXPECT. IF OPT[sum] = 6, then in DP paths there is 6 paths.
            print("DP_PATHS", dp_paths)
            print("OPT", OPT)
            
            '''
            Problem with getting all paths: we end up with all permutations instead of 
            combinations: 
            
            Output: "(2 2 2 2)(2 2 4)(2 4 2)(2 6)(4 2 2)(4 4)(6 2)(8)"
            Expected Output: "(2 2 2 2)(2 2 4)(2 6)(4 4)(8)"
            SO WE NEED LIMIT ARGUMENT.
            '''
            
            results = []
            
            def get_all_paths(node, path, limit):
                kids = g[node]
                if len(kids) == 0:
                    # nonlocal results
                    results.append(path)
                
                # USING A LIMIT ALLOWS YOU TO TURN 
                # PERMUTATONS INTO COMBINATIONS IF ITS SORTED.
                # BY TRAVERSING COINS FROM LARGEST TO SMALLEST ONLY. 
                
                for k in kids:
                    coinVal = node-k
                    if coinVal <= limit:
                        cp = path.copy()
                        cp.appendleft(coinVal)
                        get_all_paths(k, cp, min(limit, coinVal))
                        
            get_all_paths(sum, deque([]), float("inf"))
            final=[]
            
            # Uncomment this line and code still creates correct output!
            # results = dp_paths[sum]

            for r in results:
                if len(r) == 0:
                    continue
                s = str(r[0])
                for idx in range(1, len(r)):
                    s += " " + str(r[idx])
                final.append(s)
            
            final.sort()
            
            if len(final) == 0:
                return "Empty"
                
            last = ")(".join(final)
            return "(" + last + ")" 



-69) You can return a deduplicated sorted list of combinations from a 
     sorted list that has repeated values without using the sort function, or sets.
     When you dfs, include a limit argument, and dont include the current argument in the take/donttake 
     if its the same as the previous argument, and the previous argument was not taken. 
     Look at sumSubsets for more info. 



-68) Counting clouds by removing and growing as an alternative DFS:

    Given a 2D grid skyMap composed of '1's (clouds) and '0's (clear sky), 
    count the number of clouds. A cloud is surrounded by clear sky, and is 
    formed by connecting adjacent clouds horizontally or vertically. 
    You can assume that all four edges of the skyMap are surrounded by clear sky.

    Example

    For

    skyMap = [['0', '1', '1', '0', '1'],
              ['0', '1', '1', '1', '1'],
              ['0', '0', '0', '0', '1'],
              ['1', '0', '0', '1', '1']]
    the output should be
    countClouds(skyMap) = 2;
    
    
    def countClouds(skyMap):
        if not skyMap or not skyMap[0]:
            return 0
        m, n = len(skyMap), len(skyMap[0])
        ones = {(i, j) for i in range(m) for j in range(n) if skyMap[i][j] == '1'}
        cc = 0
        while ones:
            active = {ones.pop()}
            while active:
                ones -= active
                nxt_active = set()
                for x, y in active:
                    for dx, dy in ((-1,0), (1,0), (0,-1), (0,1)):
                        if 0 <= x+dx < m and 0 <= y + dy < n and \
                            (x+dx, y+dy) in ones:
                            nxt_active.add((x+dx, y+dy))
                active = nxt_active
            cc += 1
        return cc


-67) Lazy updates to build faster data structures (aka min stack extended ):

        Similar hill finding question from IMC oa: 
        Techniques: for stack 2.0, where we create a stack
        but we can also increment alll elements below index i 
        by a value
        
        -> implement push, pop, increment(index i, value v)
        you use 2 stacks, and we do LAZY updates. Similar to min stack.
        When we access an element that should have been increment. 
        add stack value + increment stack value. 
        When we increment we only save it at index i. not [0...i] with for loop
        to do O(1) lookup, push, pop, increment. And when we pop that index,
        assign to index i-1.
        
        THE IDEA IS: -> look at the very specific constraints of problem and solve 
        for only what it is asking. nothing more (which allows you to simplify and 
        improve solutions).
        
        Try to solve by being as LAZY as possible, and keeping track of critical indexes. 
        Do it similar to how you as a lazy human would solve it IRL. 
        
        By waiting to do operations until it is necessary -> and being GREEDY and smart 
        about how to update the state of the problem for only the next state[and just the next state], 
        and not all states, we optimized stack 2.0. 

        IMPLEMENTATION OF SUPER STACK:
    
        def superStack(operations):
            stack = []
            inc = []
            result = []
            '''
            Save and propogate lazy updates using inc[]
            based on how we access stack 
            '''
            for op in operations:
                
                items = op.split()
                cmd = items[0]  
                if cmd == "push":
                    stack.append(int(items[1]) )
                    inc.append(0)
                elif cmd == "pop":
                    if len(stack) > 0:
                        stack.pop()
                        poppedInc = inc.pop()
                        if len(inc) > 0:
                            inc[-1] += poppedInc
                elif cmd == "inc":
                    # inc 2 2
                    pos, val = int(items[1]), int(items[2])
                    inc[pos-1] += val
                
                if len(stack) > 0:
                    print(stack[-1] + inc[-1])
                else:
                    print("EMPTY")








-66)  Hill finding w/ stacks and queues and lazy updates in data structures: 

        '''
        Given an array a composed of distinct elements, find 
        the next larger element for each element of the array, i.e. 
        the first element to the right that is greater than this element, 
        in the order in which they appear in the array, and return the 
        results as a new array of the same length. If an element does 
        not have a larger element to its right, put -1 in the 
        appropriate cell of the result array.

        Example

        For a = [6, 7, 3, 8], the output should be
        nextLarger(a) = [7, 8, 8, -1]

        '''
        # use queue. 
        '''
        HILL FINDING WITH CRITICAL INDEXES + LAZINESS LECTURE.  
        KEEP TRACK OF KEY POINTS ONLY IN QUEUE/STACK. 
        NO WASTE IN QUEUE, JUST WHAT WE NEED. 
        AKA hill finding.         
        '''

        def nextLarger(a):        
            st = []
            res = []

            for i in range(len(a)-1, -1, -1):
                val = a[i]
                while len(st) > 0:
                    if a[i] > st[-1]:
                        st.pop()
                    else:
                        break     
                if len(st) == 0:
                    res.append(-1)
                else:
                    res.append(st[-1])
                st.append(val)
            return res[::-1]



-65) REGEX REVIEW USAGE:

    You categorize strings into three types: good, bad, or mixed. If a string has 
    3 consecutive vowels or 5 consecutive consonants, or both, then it is categorized 
    as bad. Otherwise it is categorized as good. Vowels in the English alphabet are 
    ["a", "e", "i", "o", "u"] and all other letters are consonants.

    The string can also contain the character ?, which can be replaced by either a 
    vowel or a consonant. This means that the string "?aa" can be bad if ? is a 
    vowel or good if it is a consonant. This kind of string is categorized as mixed.

    Implement a function that takes a string s and returns its category: good, bad, or mixed.

    def classifyStrings(s):
        if re.search(r"[aeiou]{3}|[^aeiou?]{5}", s):
            return "bad"
        if "?" not in s:
            return "good"
        a = classifyStrings(s.replace("?", "a", 1))
        b = classifyStrings(s.replace("?", "b", 1))
        return "mixed" if a != b else a


+-64) Bomber DP or is the DP just precomputation below? you should check:
+    (CAN DO WITH PRECOMPUTATION BUT LETS DO WITH DP!!!)
+    
+    Each cell in a 2D grid contains either a wall ('W') or an 
+    enemy ('E'), or is empty ('0'). Bombs can destroy enemies, 
+    but walls are too strong to be destroyed. A bomb placed in 
+    an empty cell destroys all enemies in the same row and column, 
+    but the destruction stops once it hits a wall.
+
+    Return the maximum number of enemies you can destroy using one bomb.
+
+    Note that your solution should have O(field.length · field[0].length) 
+    complexity because this is what you will be asked during an interview.
+
+    Example
+    For
+    field = [["0", "0", "E", "0"],
+            ["W", "0", "W", "E"],
+            ["0", "E", "0", "W"],
+            ["0", "W", "0", "E"]]
+    the output should be
+    bomber(field) = 2.
+
+    Sol'n A Easy (Cool Top Down):
+        from functools import lru_cache
+        def bomber(q):
+            if not q or not q[0]:
+                return 0
+            a , b = len(q),len(q[0])
+            @lru_cache(maxsize=None)
+            def g(m,n,x,y):
+                return 0 if m<0 or n<0 or m>=a or n>=b or q[m][n]=="W" \
+                    else g(m + x,n + y,x,y)+(q[m][n]=="E")
+            ans = 0
+            for i in range(a):
+                for j in range(b):
+                    if q[i][j] == "0":
+                        ans = max(ans,g(i-1,j,-1,0)+g(i,j-1,0,-1)+g(i+1,j,1,0)+g(i,j+1,0,1))
+            return ans
+    Soln B:
+        def bomber(F):
+            if not F or not F[0]         :   return 0
+            row ,col = len(F) ,len(F[0]) ;   F = numpy.array(F)
+            dp = numpy.zeros((row,col))  ;   t = zip(*numpy.where(F == 'E'))
+            for x,y in t:
+                for i in range(y-1,-1,-1):   
+                    if F[x,i] == 'W'  :   break
+                    if F[x,i] == '0' :   dp[x,i]+=1 
+                for i in range(y+1,col):
+                    if F[x,i] == 'W'  :   break
+                    if F[x,i] == '0'  :   dp[x,i]+=1 
+                for i in range(x-1,-1,-1):
+                    if F[i,y] == 'W'  :   break
+                    if F[i,y] == '0'  :   dp[i,y]+=1 
+                for i in range(x+1,row):
+                    if F[i,y] == 'W'  :   break
+                    if F[i,y] == '0'  :   dp[i,y]+=1 
+            return dp.max()
+
+    Soln C:
+        def bomber(A):
+            from itertools import groupby
+            if not A or not A[0]: return 0
+            R, C = len(A), len(A[0])
+            dp = [ [0] * C for _ in xrange(R) ]
+            for r, row in enumerate(A):
+                c = 0
+                for k, v in groupby(row, key = lambda x: x != 'W'):
+                    w = list(v)
+                    if k:
+                        enemies = w.count('E')
+                        for c2 in xrange(c, c + len(w)):
+                            dp[r][c2] += enemies
+                    c += len(w)
+
+            for c, col in enumerate(zip(*A)):
+                r = 0
+                for k, v in groupby(col, key = lambda x: x != 'W'):
+                    w = list(v)
+                    if k:
+                        enemies = w.count('E')
+                        for r2 in xrange(r, r + len(w)):
+                            dp[r2][c] += enemies
+                    r += len(w)
+            
+            ans = 0
+            for r, row in enumerate(A):
+                for c, val in enumerate(row):
+                    if val == '0':
+                        ans = max(ans, dp[r][c])
+            return ans

-63) IMPORTANT TRICK: 
     PRECOMPUTING LEFT AND RIGHT INDEX SUMS WITH KADANES

    Strategy:
        Try other precomputes besides cumulative sums. 
        Use the precomputed solution for one problem to 
        make several other precomputes that can be used for 
        a more difficult but similarly looking problem!!! 

        For instance pre-computes of the Kadane algorithm.

    WITH KADANES, YOU CAN PRECOMPUTE maximum array sum that starts at i, 
    and max array sum that ends at j. and use these to solve a question.

    Problem: max_double_slice_sum

    A non-empty array A consisting of N integers is given.

    A triplet (X, Y, Z), such that 0 ≤ X < Y < Z < N, is called a double slice.

    The sum of double slice (X, Y, Z) is the total of 
        A[X + 1] + A[X + 2] + ... + A[Y − 1] + A[Y + 1] + A[Y + 2] + ... + A[Z − 1].

        A[0] = 3
        A[1] = 2
        A[2] = 6
        A[3] = -1
        A[4] = 4
        A[5] = 5
        A[6] = -1
        A[7] = 2
        
    contains the following example double slices:

    double slice (0, 3, 6), sum is 2 + 6 + 4 + 5 = 17,
    double slice (0, 3, 7), sum is 2 + 6 + 4 + 5 − 1 = 16,
    double slice (3, 4, 5), sum is 0.
    The goal is to find the maximal sum of any double slice.

    Write a function:

    def solution(A)
    that, given a non-empty array A consisting of N integers, 
    returns the maximal sum of any double slice.

    For example for above array,
    the function should return 17, because no double 
    slice of array A has a sum of greater than 17.

    ALGORITHM:
    You can use a modified form of Kadane's algorithm that 
    calculates the MAX Sum subarray ending at each index.
    
    For each index, calculate the max_sum_ending_at[i] value 
    by using Kadane's algorithm in forward direction.
    
    For each index, calculate the max_sum_starting_from[i] 
    value by using Kadane's algorithm in reverse direction.
    
    Iterate these arrays simultaneously and choose the 'Y' that has the maximum value of

    max_sum_ending_at[Y-1] + max_sum_starting_from[Y+1]


    def solution(A):
        l_max_slice_sum = [0] * len(A)
        r_max_slice_sum = [0] * len(A)

        for i in range(1, len(A)-2): # A[X + 1] + A[X + 2] + ... + A[Y − 1]
            # Let's assume that Y is equal to i+1.
            # If l_max_slice_sum[i-1] + A[i] is negative, we assign X to i.
            # It means that the slice sum is 0 because X and Y are consecutive indices.
            l_max_slice_sum[i] = max(l_max_slice_sum[i-1] + A[i], 0)

        for i in range(len(A)-2, 1, -1): # A[Y + 1] + A[Y + 2] + ... + A[Z − 1]
            # We suppose that Y is equal to i-1.
            # As aforementioned, Z will be assigned to i if r_max_slice_sum[i+1] + A[i]
            # is negative, and it returns 0 because Y and Z becomes consecutive indices.
            r_max_slice_sum[i] = max(r_max_slice_sum[i+1] + A[i], 0)

        max_slice_sum = l_max_slice_sum[0] + r_max_slice_sum[2]
        for i in range(1, len(A)-1):
            # Let's say that i is the index of Y.
            # l_max_slice_sum[i-1] is the max sum of the left slice, and
            # r_max_slice_sum[i+1] is the max sum of the right slice.
            max_slice_sum = max(max_slice_sum, l_max_slice_sum[i-1]+r_max_slice_sum[i+1])
            
        return max_slice_sum



-62) RECURSION AND ENUMERATION WITH FROZENSET AND DP:

    You have a collection of coins, and you know the values of the 
    coins and the quantity of each type of coin in it. You want to 
    know how many distinct sums you can make from non-empty 
    groupings of these coins.

    Example

    For coins = [10, 50, 100] and quantity = [1, 2, 1], the output should be
    possibleSums(coins, quantity) = 9.

    # RECURSIVE SOLUTION WITH DP. 
    def possibleSums(coins, quantity):
        @lru_cache(None)
        def recursive(i):
            if i == len(coins):
                return frozenset([0])
                
            coinType = coins[i]
            tot = quantity[i]
            amts = recursive(i+1)
            res = set()
            for amt in amts:
                for k in range(tot+1):
                    res.add(amt + k*coinType)
            return frozenset(res)
            
        s = recursive(0)
        return len(s) - 1


-61) Cycle detection in directed graph using BFS, and topological sort 
     with indegree count. 

    from collections import deque
    def hasDeadlock(connections):
        '''
        If j is in the list connections[i], then there is a directed edge from process i to process j.
        For connections = [[1], [2], [3, 4], [4], [0]], the output should be
        hasDeadlock(connections) = true.
        '''
        N = len(connections)
        
        reverse_g = [[] for i in range(N)]
        for parent, kids in enumerate(connections):
            for k in kids:
                reverse_g[k].append(parent)
                
        indegree = {}
        q = deque([])
        
        for node, kids in enumerate(reverse_g):
            indegree[node] = len(kids)
            if indegree[node] == 0:
                q.append(node)
        
        # no root nodes then cycle.
        if len(q) == 0:
            return True
        
        visited = set()
        
        while q:
            print("process", q)
            
            node = q.popleft()
            kids = connections[node]
            
            visited.add(node)
            
            for k in kids:
                indegree[k] -= 1
                
                print("SAW KID with indg", k, indegree[k])
                if(indegree[k] == 0):
                    q.append(k)
                #elif(indegree[k] <  0):
                    # this elif was wrong because indegree will never fall below 0.
                    # it just wont ever be added to queue if its part of cycle. 
                    # Cycle because the indegrees dropped below 0!
                    # return True

        if len(visited) == N: # No cycle
            return False
        else:
            return True # yes cycle





-60) You cannot use union find to detect cycles in a directed graph. 
     only undirected. 
     You cannot use bfs and bipartition coloring to detect even cycles in
     directed cycle. Only odd cycles can be found. 
     You can use BFS to do topo sort with indegree 0, then indegree 1 etc. 

     Use bellman ford for negative cycle finding. 

-59) BUCKET SORT K most frequent elements:
    Bucket Sort Algorithm: Steps on how it works:
    Create an empty array.
    Loop through the original array and put each object in a “bucket”.
    Sort each of the non-empty buckets
    Check the buckets in order and then put all objects back into the original array.

    function bucketSort(array, k) is
        buckets ← new array of k empty lists
        M ← the maximum key value in the array
        for i = 1 to length(array) do
            insert array[i] into buckets[floor(k × array[i] / M)]
        for i = 1 to k do
            nextSort(buckets[i])
        return the concatenation of buckets[1], ...., buckets[k]

    Here array is the array to be sorted and k is the number of buckets to use. 
    The maximum key value can be computed in linear time by looking up all the keys 
    once. The floor function must be used to convert a floating number to an integer. 
    The function nextSort is a sorting function used to sort each bucket. 
    Conventionally, insertion sort would be used, but other algorithms 
    could be used as well. Using bucketSort itself as nextSort 
    produces a relative of radix sort; in particular, the case 
    n = 2 corresponds to quicksort (although potentially with poor pivot choices).


    Given a non-empty array of integers, return the k most frequent elements.

    Example 1:

    Input: nums = [1,1,1,2,2,3], k = 2
    Output: [1,2]

    Bucket sort is O(N):

    def topKFrequent(self, nums, k):
        bucket = [[] for _ in range(len(nums) + 1)]
        Count = Counter(nums).items()  
        for num, freq in Count: bucket[freq].append(num) 
        flat_list = [item for sublist in bucket for item in sublist]
        return flat_list[::-1][:k]

    


-58) HOARES PARTITION, QUICK SELECT K most frequent elements: 
     Approach 2: Quickselect
     Hoare's selection algorithm
 
     Quickselect is a textbook algorthm typically used to solve the problems 
     "find kth something": kth smallest, kth largest, kth most 
     frequent, kth less frequent, etc. Like quicksort,  
     quickselect was developed by Tony Hoare, and also known as Hoare's selection algorithm.
 
     It has O(N) average time complexity and widely used in practice. It worth to note that its worth 
     case time complexity is O(N^2), although the probability 
     of this worst-case is negligible.
 
     The approach is the same as for quicksort.
 
     One chooses a pivot and defines its position in a sorted array in a 
     linear time using so-called partition algorithm.
 
     As an output, we have an array where the pivot is on its perfect position 
     in the ascending sorted array, sorted by the frequency. All elements 
     on the left of the pivot are less frequent  than the pivot, and 
     all elements on the right are more frequent or have the same frequency.
 
     Hence the array is now split into two parts. If by chance our pivot element 
     took N - kth final position, then k elements on the right are 
     these top k frequent we're looking for. If  not, we can choose 
     one more pivot and place it in its perfect position.
 
     If that were a quicksort algorithm, one would have to process both parts of the array. 
     Quickselect just deal with one side -> O(N)
 
     Algorithm
 
     The algorithm is quite straightforward :
 
     Build a hash map element -> its frequency and convert its keys into the 
     array unique of unique elements. Note that elements are unique, but 
     their frequencies are not. That means we need  
     a partition algorithm that works fine with duplicates.
 
     Work with unique array. Use a partition scheme (please check the next section) 
     to place the pivot into its perfect position pivot_index in the sorted array, 
     move less frequent elements  to the left of pivot, 
     and more frequent or of the same frequency - to the right.
 
     Compare pivot_index and N - k.
 
     If pivot_index == N - k, the pivot is N - kth most frequent element, 
     and all elements on the right are more frequent or of the 
     same frequency. Return these top kk frequent elements.
 
     Otherwise, choose the side of the array to proceed recursively.
 
     Hoare's Partition vs Lomuto's Partition
 
     There is a zoo of partition algorithms. 
     The most simple one is Lomuto's Partition Scheme.
     The drawback of Lomuto's partition is 
     it fails with duplicates.
 
     Here we work with an array of unique elements, but they are 
     compared by frequencies, which are not unique. 
     That's why we choose Hoare's Partition here.
 
     Hoare's partition is more efficient than Lomuto's partition
     because it does three times fewer swaps on average, and 
     creates efficient partitions even when all values are equal.
 
     Here is how it works:
     Move pivot at the end of the array using swap.
 
     Set the pointer at the beginning of the array store_index = left.
 
     Iterate over the array and move all less frequent elements to the 
     left swap(store_index, i). Move store_index 
     one step to the right after each swap.
 
     Move the pivot to its final place, and return this index.

    from collections import Counter
    class Solution:
        def topKFrequent(self, nums: List[int], k: int) -> List[int]:
            count = Counter(nums)
            unique = list(count.keys())
            
            def partition(left, right, pivot_index) -> int:
                pivot_frequency = count[unique[pivot_index]]
                # 1. move pivot to end
                unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
                
                # 2. move all less frequent elements to the left
                store_index = left
                for i in range(left, right):
                    if count[unique[i]] < pivot_frequency:
                        unique[store_index], unique[i] = unique[i], unique[store_index]
                        store_index += 1

                # 3. move pivot to its final place
                unique[right], unique[store_index] = unique[store_index], unique[right]  
                
                return store_index
            
            def quickselect(left, right, k_smallest) -> None:
                """
                Sort a list within left..right till kth less frequent element
                takes its place. 
                """
                # base case: the list contains only one element
                if left == right: 
                    return
                
                # select a random pivot_index
                pivot_index = random.randint(left, right)     
                                
                # find the pivot position in a sorted list   
                pivot_index = partition(left, right, pivot_index)
                
                # if the pivot is in its final sorted position
                if k_smallest == pivot_index:
                    return 
                # go left
                elif k_smallest < pivot_index:
                    quickselect(left, pivot_index - 1, k_smallest)
                # go right
                else:
                    quickselect(pivot_index + 1, right, k_smallest)
            
            n = len(unique) 
            # kth top frequent element is (n - k)th less frequent.
            # Do a partial sort: from less frequent to the most frequent, till
            # (n - k)th less frequent element takes its place (n - k) in a sorted array. 
            # All element on the left are less frequent.
            # All the elements on the right are more frequent.  
            quickselect(0, n - 1, n - k)
            # Return top k frequent elements
            return unique[n - k:]








-57) Counting all subarrays trick with prefixes:
     
     1.   Subarray Sums Divisible by K
    
    Given an array A of integers, return the number of 
    (contiguous, non-empty) subarrays that have a sum divisible by K.

    Example 1:

    Input: A = [4,5,0,-2,-3,1], K = 5
    Output: 7
    Explanation: There are 7 subarrays with a sum divisible by K = 5:
    [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]

    
    arri + .. arrj % k == 0 
    arri.. arrj1 % k=  - arrx + arry % k
    
    0 4 9 9 7 4 5
    0 4 1 1 2 1 0     


    class Solution(object):
        def subarraysDivByK(self, A, K):
            P = [0]
            for x in A:
                P.append((P[-1] + x) % K)

            count = collections.Counter(P)
            return sum(v*(v-1)/2 for v in count.values())



-56) Do N CHOOSE K. Generate all K combinations:
     Recursively:

     // C++ program to print all combinations of size 
    // k of elements in set 1..n 
    #include <bits/stdc++.h> 
    using namespace std; 
    
    void makeCombiUtil(vector<vector<int> >& ans, 
        vector<int>& tmp, int n, int left, int k) 
    { 
        // Pushing this vector to a vector of vector 
        if (k == 0) { 
            ans.push_back(tmp); 
            return; 
        } 
    
        // i iterates from left to n. First time 
        // left will be 1 
        for (int i = left; i <= n; ++i) 
        { 
            tmp.push_back(i); 
            makeCombiUtil(ans, tmp, n, i + 1, k - 1); 
    
            // Popping out last inserted element 
            // from the vector 
            tmp.pop_back(); 
        } 
    } 
    
    // Prints all combinations of size k of numbers 
    // from 1 to n. 
    vector<vector<int> > makeCombi(int n, int k) 
    { 
        vector<vector<int> > ans; 
        vector<int> tmp; 
        makeCombiUtil(ans, tmp, n, 1, k); 
        return ans; 
    } 
    
    // Driver code 
    int main() 
    { 
        // given number 
        int n = 5; 
        int k = 3; 
        vector<vector<int> > ans = makeCombi(n, k); 
        for (int i = 0; i < ans.size(); i++) { 
            for (int j = 0; j < ans[i].size(); j++) { 
                cout << ans.at(i).at(j) << " "; 
            } 
            cout << endl; 
        } 
        return 0; 
    } 
        
-55)




-53) Recursive Multiply:
    Write a recursie function to multiply 2 positive integers without using the 
    * operator. Only addition, subtraction and bit shifting but minimize ops. 

    Answer:

    int minProduct(int a, int b) {
        int bigger = a< b ? b : a;
        int smaller = a < b ? a: b;
        return minProductHelper(a, b);
    }

    int minProdHelper(smaller, bigger) {
        if smaller == 0 return 0
        elif smaller == 1 return 1

        int s = smaller >> 1; //divide by 2
        int halfPrd = minProductHelper(s, bigger);
        if smaller % 2 == 0:
            return halfProd + halfProd
        else:
            return halfProd + halfProd + bigger
    }
    Runtime O(log s)
    





-52) Random Node: Create a binary tree class, which in addition to 
    insert, find, and delete, has a method getRandomNode() which 
    returns a random node from teh tree. All nodes should be equally likely to be chosen

    Just traverse left, return current, or right node with 1/3 probability
    But this isnt O(N) for all nodes, top nodes more likely.

    Return root with 1/N prb
    odds of picking left node is LEFT_SIZE * 1/N, odds of going right is RIGHT_SIZE * 1/n

    But we dont want to make that many RNG calls. Use inorder traversal to help:

    Generate one random number -> this is the node (i) to return, and then 
    we locate ith node using inorder traversal. Suubtracting leftsz + 1 from i 
    reflects that, when we go right, we skip over left+1 nodes in the inorder traversal. 




-51) Inorder Succcesor. Find next node of a given node in a BST. You
    may assume each node has link to parent:

    Node inorderSucc(Node n): 
        if n has a right subtree:
            return leftmost child of right subtree
        else: 
            while n is a right child of n.parent:
                n = n.parent
            
            return n.parent
    


-50) Cool counting trick to count pairs: (Done in CodeSignal HARD)
    ALSO RECALL THEORY THAT SORTEDNESS OF NUMBERS YIELDS COMBINATIONS NOT PERMUTATIONS 
    In case you have a problem where you need to get all combinations! just enforce 
    a sort on the list before picking elements. 

    Problem: A reverse is a number reversed. 
    So 20 reversed is 2, 420 reversed is 24, 56060 reversed is 6065

    Given 2 arrays, A and ReverseA, count all pairs i <= j
    where A[i] + ReverseA[j] == Reverse[i] + A[j].
    
    So for instance given input: [1, 20, 3,  19 ] ->   A[i] + ReverseA[j] 
                ReverseA is then [1,  2, 3,  91 ]
        A[i] + ReverseA[j] == Reverse[i] + A[j] indexes are:
        [(0,0),(1,1), (2,2), (3,3), (0,2)]

    SINCE WE ARE COUNTING AND DONT HAVE TO LIST OUT ALL THE PAIRS IN THE
    SOLUTION THIS TIPS US OFF THAT THERE IS AN O(N) soln. 
    Brute force soln -> Enumerating pairs is O(N^2). 

    OK BE SMART BY REARRANGING CONSTRAINT SO THAT WE CAN DO EASILY PRECOMPUTATIONS. 
    A[i] + ReverseA[j] == Reverse[i] + A[j]
    Rearrange to have variables of same type on same side:
    
    A[i] - Reverse[i] == A[j] - ReverseA[j]
    Ok now precompute the top array. 

    differences = []
    for i,j in zip(A, RevA):
        differences.append(i - j )
    
    AND RMBR YOU CAN COUNT PAIRS USING A MAP!! Pairs where i <= j
    Use formula n*(n-1)/2

    m = defaultdict(int)
    
    for diff in differences:
        m[diff] += 1

    So if m[3] -> means 3 different indexs have the same difference and 3 diff 
                  indexes satisfy A[i] - Reverse[i] == A[j] - ReverseA[j]
                  For instance index: (2, 4, 7)
                  -> So 2<->4, 2<->7, and 4<->7 should be counted. Also rmbr 2<->2, 4<->4, 7<->7
                     need to be counted too.
                    Its a permutation of 2 elements (i,j), but with order, so to remove order, 
                    divide by the ways it can be ordered which is 2!. Finanl solution is  
                    aka _ _   (n) x (n-1) / 2!
                    And then to sum the 2<->2, 4<->4, 7<->7, just add lenght of list. 

                -> For example: count pairs from these indicies where i<=j : [1,3,4,5]
                this is 4*3/2 -> 6
                1<->3, 1<->4, 1<->5, 3<->4, 3<->5, 4<->5s
                lst sz 4: 3+2+1 -> 6

    count = 0
    for k,v in m.items():
        count += v*(v-1)/2
    count += len(differences) # WHY DO WE NEED TO DO THIS 2024??
    return count

    



-49) To do math cieling operation on num:
    (num - 1) // divisor + 1

-48) DP + BINARY Search (DELETE IF YOU DONT UNDERSTAND):

    1180 - Software Company

    This company has n employees to do the jobs. To manage the two 
    projects more easily, each is divided into m independent subprojects. 
    Only one employee can work on a single subproject at one
    time, but it is possible for two employees to work 
    on different subprojects of the same project
    simultaneously. Our goal is to finish the projects as soon as possible.

    Each case starts with two integers n (1 ≤ n ≤ 100), and m (1 ≤ m ≤ 100). Each of the next n lines
    contains two integers which specify how much time in seconds it will take for the specified
    employee to complete one subproject of each project. So if the line contains x and y, it means that it
    takes the employee x seconds to complete a subproject from the first project, and y seconds to
    complete a subproject from the second project.    
    Input        -> Output : 
    3 20           Case 1: 18
    1 1           The input will be such that answer will be within 50000.
    2 4
    1 6

    Run a binary search fixing the time needed to complete both the projects. 
    Now you know for each employee, the doable max amount of sub-projects of A fixing 
    the amount of sub-projects to do of type B. Keep a dp[i][j] which means the maximum 
    number of sub-projects of B that can be done while j sub-projects of A are still 
    to be done and we’re currently at employee i. If dp[0][m] >= m 
    then the time fixed is ok. Answer is the minimum such time.

    const int N = 107;
    int dp[N][N];
    int n, m;
    int x[N], y[N];

    bool ok(int tym) {
        for(int i=1; i<=m; ++i) dp[n][i] = -INF;
        dp[n][0] = 0;
        for(int i=n-1; i>=0; --i) {
            for(int j=0; j<=m; ++j) {
                dp[i][j] = -INF;
                int max_a = tym / x[i];
                max_a = min(j, max_a);
                for(int k=0; k<=max_a; ++k) {
                    int max_b = (tym-k*x[i]) / y[i];
                    dp[i][j] = max(dp[i][j], max_b + dp[i+1][j-k]);
                }
            }
        }
        return (dp[0][m] >= m);
    }

    int main() {
        int t, tc=0;
        scanf("%d", &t);

        while(t--) {
            scanf("%d %d", &n, &m);
            for(int i=0; i<n; ++i) scanf("%d %d", x+i, y+i);

            int lo = 1, hi = 50000;
            while(lo < hi) {
                int mid = (lo + hi) / 2;
                if(ok(mid)) hi = mid;
                else lo = mid + 1;
            }
            printf("Case %d: %d\n", ++tc, hi);
        }

        return 0;
    }


-47) GREEDY ALGORITHM INTERVAL DEADLINES C++
    
    It is required to create such a schedule to accomplish the biggest number of jobs.

    struct Job {
        int deadline, duration, idx;

        bool operator<(Job o) const {
            return deadline < o.deadline;
        }
    };

    vector<int> compute_schedule(vector<Job> jobs) {
        sort(jobs.begin(), jobs.end());

        set<pair<int,int>> s;
        vector<int> schedule;
        for (int i = jobs.size()-1; i >= 0; i--) {
            int t = jobs[i].deadline - (i ? jobs[i-1].deadline : 0);
            s.insert(make_pair(jobs[i].duration, jobs[i].idx));
            while (t && !s.empty()) {
                auto it = s.begin();
                if (it->first <= t) {
                    t -= it->first;
                    schedule.push_back(it->second);
                } else {
                    s.insert(make_pair(it->first - t, it->second));
                    t = 0;
                }
                s.erase(it);
            }
        }
        return schedule;
    }


-43.5) USE BINARY SEARCH EVEN WHEN YOU DONT THINK YOU NEED IT.
       USE IT WHEN YOU CAN MAP HALF THE VALUES TO TRUE AND THE OTHER HALF TO FALSE SOMEHOW
       COME UP WITH MAPPING THEN UTILIZE BINSEARH!

        
        LC:  644-maximum-average-subarray-ii
        Given an array consisting of n integers, find the contiguous 
        subarray whose length is greater than or equal to k 
        that has the maximum average value. 
        And you need to output the maximum average value.
        Input: [1,12,-5,-6,50,3], k = 4
        Output: 12.75
        Explanation:
        when length is 5, maximum average value is 10.8,
        when length is 6, maximum average value is 9.16667.
        Thus return 12.75.

        
        Method 1: Use Binary Search

            first we do a binary search on the average and let it be x
            we decrease x from all of the array elements and if there exists a 
            sub array with lengh more than k whose sum is more than zero then we can 
            say that we have such a sub array whose average is more than x other wise 
            we can say that there doesnt exist any such sub array

            how to find out if there is a sub array whose sum is more than zero and its 
            length is more than k? we can say that a sub array [l, r) equals sum[1, r) — sum[1, l) 
            so if we get the partial sums and fix the r of the sub array we just need an l 
            which sum[1, r) >= sum[1, l) and l <= r — k this can 
            be done with partial minimum of the partial sums

        Method 2: Use a diff bin search
            Goal: Find the maximum average of continuous subarray of 
            length greater than or equal to k.

            Assumption: The answer is between the maximum value 
            and the minimum value in the array.

            As a result, we can do a binary search for the answer, 
            while using the minimum value and the maximum value as left and right boundary (inclusive).

            Given the initial value of left and right boundary, we can 
            compute the mid value. The problem is how we decide where the
            answer lies, in [left, mid) or [mid, right].

            If and only if there exists a subarray whose length is at least k, 
            and its average value is greater than or equal mid, 
            then the answer lies in [mid, right].

            The problem becomes: Decide if there exists a subarray 
            whose length is at least k, and its average value is greater than mid.

            Consider such scenario: The average of A[1..n] >= mid is the same 
            as the average of B[1..n] >= 0 where B[i] = A[i] - mid.

            If we construct the new array B[] based on the given array A[], 
            the problem becomes: Decides if there's a subarray whose length 
            is at least k, and its sum is greater than 0,

            which is the same as finds the maximum subarray sum where 
            the length of the subarray needs to be at least k.

            When it comes to the maximum subarray sum, it is natural to 
            think of the classic solution: "keep adding each integer to the 
            sequence until the sum drops below 0. If sum is negative, reset the sequence."

            However, it cannot work for our case, as we have requirement for the subarray length.

            Another simple way to find the maximum subarray is dynamic programming. 
            If we can use DP to calculate the presum of the array (every subarray 
            with start from the first element), we only need to find the maximum 
            increase in the presum array.

            We can slightly modify this approach to solve our problem. 
            When finding the maximum increase, we no longer record the smallest 
            element appeared before, but record the smallest element appeared k positions before.

            You should easily see that this will cover the case where the 
            length of the subarray is greater than k.
            
            Time complexity O(nlog(max-min))
            Space complexity O(1)

            class Solution {
            public :
                double findMaxAverage(vector< int >& nums, int k) {
                    int n = nums.size();
                    vector < double > sums(n + 1 , 0 );
                    double left = * min_element(nums.begin(), nums.end());
                    double right = * max_element(nums.begin(), nums.end()) ;
                    while (right-left> 1e- 5 ) {
                        double minSum = 0 , mid = left + (right-left) / 2 ;
                        bool check = false ;
                        for ( int i = 1 ; i <= n; ++ i) {
                            sums[i] = sums[i- 1 ] + nums[i- 1 ] -mid;
                            if (i >= k) {
                                minSum = min(minSum, sums[i- k]);
                            }
                            if (i >= k && sums[i]> minSum) {check = true ; break ;}
                        }
                        if (check) left = mid;
                        else right = mid;
                    }
                    return left;
                }
            };




-43) Maximum Slice problem, Kadane's algorithm and extensions

    Now we give a full version of the solution, 
    which additionally also finds the boundaries of the desired segment:

    int ans = a[0], ans_l = 0, ans_r = 0;
    int sum = 0, min_sum = 0, min_pos = -1;

    for (int r = 0; r < n; ++r) {
        sum += a[r];
        int cur = sum - min_sum;
        if (cur > ans) {
            ans = cur;
            ans_l = min_pos + 1;
            ans_r = r;
        }
        if (sum < min_sum) {
            min_sum = sum;
            min_pos = r;
        }
    }



-42.5) 2D Kadanes Algorithm: 

    Two-dimensional case of the problem: search for maximum/minimum submatrix

    Kadane’s algorithm for 1D array can be used to reduce the time complexity to O(n^3). 
    The idea is to fix the left and right columns one by one and find the maximum 
    sum contiguous rows for every left and right column pair. We basically find top and bottom 
    row numbers (which have maximum sum) for every fixed left and right column pair. 
    To find the top and bottom row numbers, calculate sum of elements in every row from left 
    to right and store these sums in an array say temp[]. So temp[i] indicates sum of elements 
    from left to right in row i. If we apply Kadane’s 1D algorithm on temp[], 
    and get the maximum sum subarray of temp, this maximum sum would be the maximum possible 
    sum with left and right as boundary columns. To get the overall maximum sum, we compare 
    this sum with the maximum sum so far.

    def kadane(arr, start, finish, n): 
        
        # initialize sum, maxSum and  
        Sum = 0
        maxSum = -999999999999
        i = None
    
        # Just some initial value to check 
        # for all negative values case  
        finish[0] = -1
    
        # local variable  
        local_start = 0
        
        for i in range(n): 
            Sum += arr[i]  
            if Sum < 0: 
                Sum = 0
                local_start = i + 1
            elif Sum > maxSum: 
                maxSum = Sum
                start[0] = local_start  
                finish[0] = i 
    
        # There is at-least one 
        # non-negative number  
        if finish[0] != -1:  
            return maxSum  
    
        # Special Case: When all numbers  
        # in arr[] are negative  
        maxSum = arr[0]  
        start[0] = finish[0] = 0
    
        # Find the maximum element in array 
        for i in range(1, n): 
            if arr[i] > maxSum: 
                maxSum = arr[i]  
                start[0] = finish[0] = i 
        return maxSum 
    
    # The main function that finds maximum 
    # sum rectangle in M[][]  
    def findMaxSum(M): 
        global ROW, COL 
        
        # Variables to store the final output  
        maxSum, finalLeft = -999999999999, None
        finalRight, finalTop, finalBottom = None, None, None
        left, right, i = None, None, None
        
        temp = [None] * ROW 
        Sum = 0
        start = [0] 
        finish = [0]  
    
        # Set the left column  
        for left in range(COL): 
            
            # Initialize all elements of temp as 0  
            temp = [0] * ROW  
    
            # Set the right column for the left  
            # column set by outer loop  
            for right in range(left, COL): 
                
                # Calculate sum between current left  
                # and right for every row 'i' 
                for i in range(ROW): 
                    temp[i] += M[i][right]  
    
                # Find the maximum sum subarray in  
                # temp[]. The kadane() function also  
                # sets values of start and finish.  
                # So 'sum' is sum of rectangle between   
                # (start, left) and (finish, right) which  
                # is the maximum sum with boundary columns  
                # strictly as left and right.  
                Sum = kadane(temp, start, finish, ROW)  
    
                # Compare sum with maximum sum so far.  
                # If sum is more, then update maxSum  
                # and other output values  
                if Sum > maxSum: 
                    maxSum = Sum
                    finalLeft = left  
                    finalRight = right  
                    finalTop = start[0]  
                    finalBottom = finish[0] 
    
        # Prfinal values  
        print("(Top, Left)", "(", finalTop,  
                                finalLeft, ")")  
        print("(Bottom, Right)", "(", finalBottom,  
                                    finalRight, ")")  
        print("Max sum is:", maxSum) 
    
    # Driver Code 
    ROW = 4
    COL = 5
    M = [[1, 2, -1, -4, -20], 
        [-8, -3, 4, 2, 1],  
        [3, 8, 10, 1, 3],  
        [-4, -1, 1, 7, -6]]  
    
    findMaxSum(M) 


-42) RMQs and processing array index and value at same time using 
     a monotonic queue philosophy + Divide and conquer trick with dealing with all ranges. 
     Sometimes you have to deal with 2 constraints at the same time. 

    Problem: 
    You have an array and a set 
         1. 1. 2. 3. 4
    a = [4, 2, 1, 6, 3]
    S = {0, 1,  3, 4}

    for each pair in the set, find the minimum value in that range,
    then return sum of all minimum values as answer:
    
    0-1 => a[0:2] => [4, 2] => min([4, 2]) = 2
    0-3 => [4, 2, 1, 6] => 1
    1-3 => [2, 1, 6] => 1
    Therefore the answer is ->  2 + 1 + 1 = 4 <-

    To do this question, you have to satisfy both the range condition 
    and the find minimum value condition. 

    My proposed garbage solution was precomputing all running minimum values 
    for every (i, j) range like so: (This soln is in LunchclubIORMQ in most important problems)
    [
        [4, 2, 1,1,1]
        [2, 1,1, 1]
        [1,1, 1]
        [6, 3]
        [3]
    ]
    Then enumerate every set in O(n^2)

    CAN ALSO SOLVE WITH DP LIKE BELOW -----------------------------
    Another way was solving the problem for all set ranges, was to compute the mins 
    between you and the next element to you. THen to compute the min for ranges that 
    are longer than that, just do the min of those precomputed solutions DP style.

    So compute [0,1], [1,3], [3,4] Then to compute [0, 3], and [1, 4] don't look at a, 
    just take the min of the ranges that make up the bigger ranges!!

    THE BEST WAY TO SOLVE BELOW -----------------------------
    Better way was to think of ways to split the sets using array a. 
    We can create a binary tree where it splits on minimum values with a, 
    and query each subset range rite! The lca, is the min. 

    This ideation leads to another solution. 
    Lets sort A: and keep track of the index as well. 
    Since we sorted on the A[i], WE have to do the second order processing 
    on the index, i. We sorted on A[i] so that we can linearly process it
    and in a convenient way that respects the first constraint. 


    A = [(1, 2), (2,1 ), (3, 4), (4, 0), (6, 3)]
    The trick in this problem is to binary search the index to satisfy the second
    constraint, and do a divide and conquer technique on the problem. 

    So process (1,2). its the smallest and it splits the set S in 2:
    S = {0, 1, 3, 4, }
              ^
    All ranges that cross the binary insertion point have 1 as a minimum value!
    So ranges 0-3, 0-4, 1-3, and 1-4 all have index 2 in the middle, and all 
    will have the min value 1. 

    Then we process next element in A, but we do not want to count the same ranges twice in 
    our final resulting answer. You can fix this by using a visited set.
    The smarter way is to divide and conquer S:
    Split S in two: 
    S1 = {0, 1}, and S2 = {3, 4}
    Then binary search in the smaller subsets so that you do not see repeat ranges from 
    having split it already on (A[1], 1)

    Search with index 1 on S1 because S2 does not contain index 1.
    You will process [0, 1] -> give min value 2. 
    Then finally compute answer like so. 




-41) Insert interval into a sorted set of intervals using 2 binary searches:
    Binary search the start time on the end times (bisect_right), 
    and the end time on the start times (bisect_left)
    if same value you can insert, otherwise cant
    Works on open close start end intervals. 
    Review BookingTimes in Most important problems. 



-40) CIRCULAR BUFFERS: Implementation Stack and Queue 
     Declare an array with enough space!
     
    7.1: Push / pop function — O(1).
    1   stack = [0] * N
    2   size = 0
    3   def push(x):
    4       global size
    5       stack[size] = x
    6       size += 1
    7   def pop():
    8       global size
    9       size -= 1
    10      return stack[size]

    7.2: Push / pop / size / empty function — O(1).
    1   queue = [0] * N
    2   head, tail = 0, 0
    3   def push(x):
    4       global tail
    5       tail = (tail + 1) % N
    6       queue[tail] = x
    7   def pop():
    8       global head
    9       head = (head + 1) % N
    10      return queue[head]
    11  def size():
    12      return (tail - head + N) % N
    13  def empty():
    14      return head == tail



-39) LEADER ALGORITHM: 
    Let us consider a sequence a0, a1, . . . , an−1. The leader of this sequence is the element whose
    value occurs more than n/2 times.

    Notice that if the sequence a0, a1, . . . , an−1 
    contains a leader, then after removing a pair of
    elements of different values, the remaining sequence 
    still has the same leader. Indeed, if we
    remove two different elements then only one of them 
    could be the leader. The leader in the
    new sequence occurs more than (n/2) − 1 = (n−2)/2 times. 
    Consequently, it is still the leader of the new sequence of n − 2 elements.

    1 def goldenLeader(A):
    2   n = len(A)
    3   size = 0
    4   for k in xrange(n):
    5       if (size == 0):
    6           size += 1
    7           value = A[k]
    8       else:
    9           if (value != A[k]):
    10              size -= 1
    11          else:
    12              size += 1
    13  candidate = -1
    14  if (size > 0):
    15      candidate = value
    16  leader = -1
    17  count = 0
    18  for k in xrange(n):
    19      if (A[k] == candidate):
    20          count += 1
    21  if (count > n // 2):
    22      leader = candidate
    23  return leader


    After removing all pairs of different
    elements, we end up with a sequence containing all the same values. 
    This value is not necessarily the leader; it is only a candidate for the leader. 
    Finally, we should iterate through all
    the elements and count the occurrences of the candidate; 
    if it is greater than n/2 then we have
    found the leader; otherwise the sequence does not contain a leader.



-38) A comparison of Forward and Backward DP illustrated:


    A game for one player is played on a board consisting of N consecutive squares, 
    numbered from 0 to N − 1. There is a number written on each square. 
    A non-empty array A of N integers contains the numbers written on the squares. 
    Moreover, some squares can be marked during the game.

    At the beginning of the game, there is a pebble on square number 0 and this 
    is the only square on the board which is marked. The goal of the game is to
    move the pebble to square number N − 1.

    During each turn we throw a six-sided die, with numbers from 1 to 6 on 
    its faces, and consider the number K, which shows on the upper face after 
    the die comes to rest. Then we move the pebble standing on square number 
    I to square number I + K, providing that square number I + K exists. If 
    square number I + K does not exist, we throw the die again until we obtain 
    a valid move. Finally, we mark square number I + K.

    After the game finishes (when the pebble is standing on square number N − 1), 
    we calculate the result. The result of the game is the sum of the numbers 
    written on all marked squares.

    For example, given the following array:

        A[0] = 1
        A[1] = -2
        A[2] = 0
        A[3] = 9
        A[4] = -1
        A[5] = -2

    The marked squares are 0, 3 and 5, so the result of the 
    game is 1 + 9 + (−2) = 8. This is the maximal possible result 
    that can be achieved on this board.

    Write a function:

    def solution(A)

    that, given a non-empty array A of N integers, returns the 
    maximal result that can be achieved on the board represented by array A.
    
    SOLUTION:
    from collections import deque

    def backwardDP(A):
        '''
        Backward dp.
            
        Start with idx 0
        update with max value, by using 6 behind you!
            
        OPT[i] = A[i] + max(OPT[i-1], OPT[i-2],..., OPT[i-5])
            
        do we ever double add...
        nah
        Just need array of size 6; to space optimize
        '''
        # the states represent the last 6 updated states. 
        states = deque([float("-inf"),
                        float("-inf"),
                        float("-inf"),
                        float("-inf"),
                        float("-inf"),
                        0])
        
        for i in range(1, len(A)):
            
            # We are always guranteed a link to something before us
            # because we always sum in max states, unlike forward dp, 
            # where you have to force it to use previous states, by 
            # setting your own state to something really shitty, so
            # relaxation forces you to pick up a state
            maxVal = A[i] + max(states)
            states.popleft()
            states.append(maxVal)
            # latestVal = maxVal
        
        # WE HAVE TO ALWAYS INCLUDE A[0] because its marked. 
        return maxVal + A[0]
    
    '''
    Forward DP 
    Do 6 RELAXATIONS FORWARD. 
        
    OPT[i+1] -> max(OPT[i+1], A[i+1] + OPT[i])?
    OPT[i+2] -> A[i+2] + A[i]?
    '''
    def forwardDP(A):
        N = len(A)
        OPT = [float("-inf") for i in range(N)]
        
        # YOU ALSO HAVE TO ENSURE THAT THE PIECES THAT ARE FURTHER THAN 
        # 6 AWAY HAVE ACCESS TO SOMETHING that was used to reach it. 
        # so FIRST 6 ARE ACCESSIBLE!
        
        # In other words, set the first 6 to the value of A[i]
        OPT[0] = 0
        
        for i in range(N):
            for k in range(1, 7):
                if i + k < len(A):
                    OPT[i+k] = max(OPT[i] + A[i+k], OPT[i+k]) 
        
        # WE HAVE TO ALWAYS INCLUDE A[0] because its marked. 
        return OPT[-1] + A[0]
        
    def solution(A):
        # FORWARD DP WORKS 100%
        # return forwardDP(A)
        return backwardDP(A)
        




-37) MAKING SURE YOUR DFS IS CORRECT! And the DP is being resolved 
     in the DFS tree properly. 

    For a given array A of N integers and a sequence S of N integers 
    from the set {−1, 1}, we define val(A, S) as follows:

    val(A, S) = |sum{ A[i]*S[i] for i = 0..N−1 }|

    (Assume that the sum of zero elements equals zero.)
    For a given array A, we are looking for such a sequence S that minimizes val(A,S).

    Write a function:
    def solution(A)

    that, given an array A of N integers, computes the minimum value of val(A,S) 
    from all possible values of val(A,S) for all 
    possible sequences S of N integers from the set {−1, 1}.

    For example, given array:

    A[0] =  1
    A[1] =  5
    A[2] =  2
    A[3] = -2
    
    your function should return 0, since for S = [−1, 1, −1, 1], 
    val(A, S) = 0, which is the minimum possible value.

    def solution(A):
        # THIS FAILS DUE TO MAX RECURSION DEPTH REACHED!
        # BUT IT IS 100% CORRECT
        @lru_cache(None)
        def recurseB(i,s):
            
            if len(A) == i:
                return s
                
            add = recurseB(i+1, s + A[i])
            sub = recurseB(i+1, s - A[i])
            print("CORRECT ADD AND SUB FOR I IS", i, add, sub)

            # print("ADD and sub are", add, sub)
            if abs(add) < abs(sub):
                return add
            else:
                return sub
        
        correct_val = abs(recurseB(0, 0))
        print("CORRECT VALU IS", correct_val)
        
        # BELOW WAY IS WRONG!
        # DO YOU KNOW WHY?
        # IT GENERATES DIFF ANSWERS FROM ABOVE. 
        # BECAUSE IN THE RECURSIVE CALLS CLOSE TO THE 
        # BASE CASE, WE ARENT ABLE TO FINE TUNE THE SOLUTION
        # TO THE INCOMING SUM, BECAUSE YOU NEVER SEE THE INCOMING
        # SUM LIKE ABOVE. 
        # SO INSTEAD, YOU GREEDILY CHOOSE 
        # IN THE ABOVE RECURSION, HELPER SEES INCOMING SUM, 
        # AND THEN RETURNS AN OPTIMIZED SUM BASED ON THE INCOMING SUM!
        # THERE IS COMMUNICATION!
        def recurseA(i):
            if len(A) == i:
                return 0
                
            add = A[i] + recurseA(i+1)
            sub = -A[i] + recurseA(i+1)
            print("INC ADD AND SUB FOR I IS", i, add, sub)
            # print("ADD and sub are", add, sub)
            if abs(add) < abs(sub):
                return add
            else:
                return sub

        incorrect_val = abs(recurseA(0))
        return correct_val

-36.5) Reasoning about hard states in DP: MinAbsSum
       Coming up with bottom up solution with MinAbsSum by rephrasing problem.
       Question above. 

    Since we can arbitrarily choose to take the element or its negative, we can simplify the
    problem and replace each number with its absolute value. Then the problem becomes dividing
    the numbers into two groups and making the difference between the sums of the two groups
    as small as possible. It is a classic dynamic programming problem.
    Assume the sum of absolute values of all the numbers is S. We want to choose some of
    the numbers (absolute values) to make their 
    sum as large as possible without exceeding S/2.

    Let M be the maximal element in the given array A. We create an array dp of size S.
    
    Slow DP:
    Let dpi equal 1 if it is possible to achieve the 
    sum of i using elements of A, and 0 otherwise.
    Initially dpi = 0 for all of i (except dp0 = 1). 
    For every successive element in A we update the
    array taking this element into account. We simply go through all the 
    cells, starting from the
    top, and if dpi = 1 then we also set dpi+Aj
    to 1. The direction in which array dp is processed
    is important, since each element of A can be used only once. 
    After computing the array dp, P is the largest index such that P <= S/2
    and dpP = 1.

    The time complexity of the above solution is O(N^2· M), since S = O(N · M).

    1 def slow_min_abs_sum(A):
    2   N = len(A)
    3   M = 0
    4   for i in xrange(N):
    5       A[i] = abs(A[i])
    6       M = max(A[i], M)
    7   S = sum(A)
    8   dp = [0] * (S + 1)
    9   dp[0] = 1
    10  for j in xrange(N):
    11      for i in xrange(S, -1, -1):
    12          if (dp[i] == 1) and (i + A[j] <= S):
    13              dp[i + A[j]] = 1
    14  result = S    
    15  for i in xrange(S // 2 + 1):
    16      if dp[i] == 1:
    17          result = min(result, S - 2 * i)
    18  return result


    Notice that the range of numbers is quite small (maximum 100). 
    Hence, there must be a lot of duplicated numbers. 
    Let count[i] denote the number of occurrences of the value i. 
    We can process all occurrences of the same value at once. 
    First we calculate values count[i] Then we create array dp such that:

    dp[j] = −1 if we cannot get the sum j,
    dp[j] >= ­ 0 if we can get sum j.
    Initially, dp[j] = -1 for all of j (except dp[0] = 0). Then we scan 
    through all the values a appearing in A; we consider all a such that 
    count[a]>0. For every such a we update dp that dp[j] denotes 
    how many values a remain (maximally) after achieving sum j. 
    Note that if the previous value at dp[j] >= 0 then we can 
    set dp[j] = count[a] as no value a is needed to obtain the sum j. 
    
    Otherwise we must obtain sum j-a first and then use a 
    number a to get sum j. In such a situation 
    dp[j] = dp[j-a]-1. Using this algorithm, we can mark all the 
    sum values and choose the best one (closest to half of S, the sum of abs of A).

    def MinAbsSum(A):
        N = len(A)
        M = 0
        for i in range(N):
            A[i] = abs(A[i])
            M = max(A[i], M)
        S = sum(A)
        count = [0] * (M + 1)
        for i in range(N):
            count[A[i]] += 1
        dp = [-1] * (S + 1)
        dp[0] = 0
        for a in range(1, M + 1):
            if count[a] > 0:
                for j in range(S):
                    if dp[j] >= 0:
                        dp[j] = count[a]
                    elif (j >= a and dp[j - a] > 0):
                        dp[j] = dp[j - a] - 1
        result = S
        for i in range(S // 2 + 1):
            if dp[i] >= 0:
                result = min(result, S - 2 * i)
        return result
    
    The time complexity of the above solution is O(N · M^2), 
    where M is the maximal element,
    since S = O(N · M) and there are at most M different values in A.




-36) DO COMPLEX SET MEMOIZATION ON GRID VS DJIKSTRA.  

    You have a grid, and you can go up down, left or right. 
     Find min cost to go from top left to bottom right: 
        board = [[42, 51, 22, 10,  0 ],
                [2,  50, 7,  6,   15],
                [4,  36, 8,  30,  20],
                [0,  40, 10, 100, 1 ]]

    # Below uses DP but its not fast enough. 
    # Need to use Djikstra to pass the TLE test cases
    def orienteeringGame(board):
        '''
        COMPLEX SET MEMOIZATION GRID PROBLEM: 
        THIS IS A DP PROBLEM
        SINCE WE CAN GO UP AND LEFT AS WELL AS DOWN AND RIGHT, 
        IN OUR MEM TABLE, WE HAVE TO SAVE THE SET OF NODES
        WE TRAVERSED SO FAR. 
        SO ITS LIKE TRAVELLING SALESMAN PROBLEM DP.
        RMBR WHEN YOU HAVE TO MEM IT!
        '''
        
        R = len(board)
        C = len(board[0])
        
        visited = set()
        m = {}
        
        # Using a cumulative sum is a way to hash the nodes traversed so far in the 
        # path which is important for the cache table.
        @lru_cache(None)
        def dfs(i, j, cum_sum):
            if(i == R-1 and j == C-1):
                return cum_sum # board[R-1][C-1]    
            
            visited.add((i,j))
            
            val = board[i][j]
            cost = float("inf")

            if i + 1 < R and (i+1, j) not in visited:
                cost = min(cost, dfs(i + 1, j, cum_sum + val))            
        
            if j+1 < C and (i, j+1) not in visited:
                cost = min(cost, dfs(i, j+1, cum_sum + val))  

            if i - 1 >=0 and (i-1, j) not in visited:
                cost = min(cost, dfs(i - 1, j, cum_sum + val))      

            if j-1 >= 0 and (i, j-1) not in visited:
                cost = min(cost, dfs(i, j-1, cum_sum + val))     
                    
            visited.remove((i,j))    
            return cost
        
        return dfs(0,0, 0) 



-35) Find duplicate subtrees, Caching trees with UNIQUE IDS
    Given a binary tree, return all duplicate subtrees. 
    For each kind of duplicate subtrees, you only need to 
    return the root node of any one of them.
    
    Normal soln -> create merkel hash with strings and postorder/preorder trav O(N^2)
    This soln -> dont create those long strings. 
    O(N) time and space. 

    def findDuplicateSubtrees(self, root):
        self.type_id_gen = 0
        duplicated_subtrees = []
        type_to_freq = defaultdict(int)
        type_to_id = {}
        
        def dfs(node):
            if not node:
                return -1
            type_id_left, type_id_right = (dfs(ch) for ch in (node.left, node.right))
            tree_type = (node.val, type_id_left, type_id_right)
            freq = type_to_freq[tree_type]
            if freq == 0:
                type_id = self.type_id_gen
                self.type_id_gen += 1
                type_to_id[tree_type] = type_id
            elif freq == 1:
                type_id = type_to_id[tree_type]
                duplicated_subtrees.append(node)
            else:
                type_id = type_to_id[tree_type] 
            type_to_freq[tree_type] += 1
            return type_id
            
        dfs(root)
        return duplicated_subtrees 
    
    Stefans version:


    def findDuplicateSubtrees(self, root, heights=[]):
        def getid(root):
            if root:
                # get the id of tree and if there isnt one, assign it with a default value 
                id = treeid[root.val, getid(root.left), getid(root.right)] 
                trees[id].append(root) 
                return id

        trees = collections.defaultdict(list)
        treeid = collections.defaultdict()
        treeid.default_factory = treeid.__len__
        getid(root)
        return [roots[0] for roots in trees.values() if roots[1:]]

    The idea is the same as Danile's: Identify trees by numbering them. 
    The first unique subtree gets id 0, the next unique subtree gets 
    id 1, the next gets 2, etc. Now the dictionary keys aren't deep 
    nested structures anymore but just ints and triples of ints.



-34) How to do transition functions and states for DP: 
     Think of all the states, and transition functions you need
     if its hard to precalculate a state, add another state!
     
     Also if you cant do backward DP, try Forward DP and relaxing states!

     METHOD 1: Think in terms of Finite state machines!!
     METHOD 2: The states may seem wierd BECAUSE YOU NEED TO CONSIDER MORE PREVIOUS LAYERS 
               of i to come up with the solution, not just the last layer like usual DP.

     1.   House Robber
     def rob(self, nums: List[int]) -> int:      
         '''
         Transition function: 
         FREE state + ROB action -> FROZEN 
         Free state + DONT ROB -> Free State
         FROZEN state + Dont Rob -> Free State.  
         '''
         
         COUNT = len(nums)
         
         FROZEN = 0
         FREE = 0 
         
         NXT_FROZEN = 0
         NXT_FREE = 0
         
         for val in nums:
             NXT_FROZEN = FREE  + val
             NXT_FREE = max(FREE, FROZEN)
             
             FROZEN = NXT_FROZEN
             FREE = NXT_FREE
         
         return max(FREE, FROZEN)

    The other way you can think of this, is we are dealing 
    with 3 layers at a time given the following recurrent relation:
    rob(i) = Math.max( rob(i - 2) + currentHouseValue, rob(i - 1) )
    States are: i, i-1, i-2
    

    Understanding the number of layers you are dealing with tell you
    how many states you will need to compute the current state!




-33) For some grid problems that require mapping numbers to new numbers while 
    keeping order with right neighbor, down neighbor, up neighbor and left neighbor 
    (Abiscus interview).And you have to use as few numbers for the mapping as possible
    Remember that to maintain multiple orderings in different directions
    you should be using a GRAPH!! and do BFS to generate those new numbers (and not 
    try to do grid DP like what i did in that interview). 

-32) An example of both forward and backward DP is 
    -> 931. Minimum Falling Path Sum. Check it out! 



-31) FINITE STATE MACHINES PROCESSING, AND BOTTOM UP DP TECHNIQUE. 
     1.   Best Time to Buy and Sell Stock with Cool Down
     Thinking about the problem as a finite state machine can be helpful
     to figure out:
        -> STATES 
        -> HOW THE TRANSITION FUNCTION WORKS. MAKE SURE YOU GET ALL THE TRANSITIONS IN!
        -> THE RECURENCE FOR THE PROBLEM OR MULTIPLE RECURRENCES FOR EACH STATE. 

     You need to figure out how many possible states there are for the DP, 
     and create a grid for each state. 
     
    TO BUY OR SELL STOCK WITH COOLDOWN DP THOUGHT PROCESS O(1) SPACE:

    Design an algorithm to find the maximum profit. You may
    complete as many transactions as you like (ie, buy one and 
    sell one share of the stockmultiple times) 
    After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
    Example:
    Input: [1,2,3,0,2]
    Output: 3 
    Explanation: transactions = [buy, sell, cooldown, buy, sell]
     
     In STOCK COOLDOWN problem (LEET 309), 
     you tried to solve the DP with 2 states -> IS Cooldown/Not Cooldown.
     
     There is a solution where you create 3 Grids -> BUY/SELL/REST GRID. 
     The grid comes from the fact that there are 3 states if you look 
     at the finite state machine. 

    3 GRID SOLUTION -> O(1) SPACE:

    def maxProfit(self, prices):
        free = 0
        have = cool = float('-inf')
        for p in prices:
            free, have, cool = max(free, cool), max(have, free - p), have + p
        return max(free, cool)
    
    '''
    free is the maximum profit I can have while being free to buy.
    have is the maximum profit I can have while having stock.
    cool is the maximum profit I can have while cooling down.

    free = max(free, cool)
    have = max(have, free - p)  # if we were free last round and just bought, 
                                # then our profit(in balance) need to 
                                # adjust because buying cost money
                        
    cool = have + p # to be in cool-down, 
                    # we just sold in last round (realizing profit), 
                    # then profit would increase by the current price
        
    '''
    HARMAN TOP DOWN:

    class Solution:
    # TOP DOWN ACCEPTED SOLUTION
    def maxProfit(self, prices: List[int]) -> int:
        
        @lru_cache(maxsize=None)
        def helper(i, bought, cooldown):
            
            if i == len(prices):
                return 0
            
            if bought == -1 and not cooldown:
                return helper(i+1, prices[i], False)
            
            if bought == -1 and cooldown:
                return helper(i+1, -1, False)
            
            if prices[i] < bought:
                return helper(i+1, prices[i], False)
            
            if prices[i] > bought:
                return max(helper(i+1, -1, True) + (prices[i] - bought), helper(i+1, bought, False) )
            
            if prices[i] == bought:
                return helper(i+1, bought, False)
            
        return helper(0, -1, False)



-30) To create bottom up -> think of recursive solution. The parameters it needs!
     Start bottom up with these parameter dimensions. Now we have to build 
     forward from base case. So figure out base case and direction.  
     
     When bottom up DP isnt working, you are usually missing a case in your recurrence!

     Then create recurrence/thinking of the grid 
     Does greater space optimization mean more performance or same performance because
     its still the same amount of cache hits either way?  
     
     -> Greater space optimization may lead to higher performance, if there are fewer steps in 
        the algorithm. 
     -> An easy space optimization is using only the previous/next rather than saving all the states 
        because the recurrence formula may only require the previous versions of all the states. 

-29.9)  

        16. 3Sum Closest
        Attempted
        Medium
        Topics
        Companies
        Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

        Return the sum of the three integers.

        You may assume that each input would have exactly one solution.

        

        Example 1:

        Input: nums = [-1,2,1,-4], target = 1
        Output: 2
        Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
        Example 2:

        Input: nums = [0,0,0], target = 1
        Output: 0
        Explanation: The sum that is closest to the target is 0. (0 + 0 + 0 = 0).
        

        Constraints:

        3 <= nums.length <= 500
        -1000 <= nums[i] <= 1000
        -104 <= target <= 104



        The two pointers pattern requires the array to be sorted, so we do that first. As our BCR is O(n2)\mathcal{O}(n^2)O(n 
        2
        ), the sort operation would not change the overall time complexity.

        In the sorted array, we process each value from left to right. For value v, we need to find a pair which sum, ideally, is equal to target - v. We will follow the same two pointers approach as for 3Sum, however, since this 'ideal' pair may not exist, we will track the smallest absolute difference between the sum and the target. The two pointers approach naturally enumerates pairs so that the sum moves toward the target.


        class Solution:
            def threeSumClosest(self, nums: List[int], target: int) -> int:
                diff = float("inf")
                nums.sort()
                for i in range(len(nums)):
                    lo, hi = i + 1, len(nums) - 1
                    while lo < hi:
                        sum = nums[i] + nums[lo] + nums[hi]
                        if abs(target - sum) < abs(diff):
                            diff = target - sum
                        if sum < target:
                            lo += 1
                        else:
                            hi -= 1
                    if diff == 0:
                        break
                return target - diff



-29.5)  259. 3Sum Smaller
        Medium
        Topics
        Companies
        Given an array of n integers nums and an integer target, find the number of 
        index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.

        Example 1:

        Input: nums = [-2,0,1,3], target = 2
        Output: 2
        Explanation: Because there are two triplets which sums are less than 2:
        [-2,0,1]
        [-2,0,3]
        Example 2:

        Input: nums = [], target = 0
        Output: 0
        Example 3:

        Input: nums = [0], target = 0
        Output: 0


        Intuition

        Let us try sorting the array first. For example, 
        nums=[3,5,2,8,1]nums = [3,5,2,8,1]nums=[3,5,2,8,1] becomes [1,2,3,5,8][1,2,3,5,8][1,2,3,5,8].

        Let us look at an example nums=[1,2,3,5,8]nums = [1,2,3,5,8]nums=[1,2,3,5,8], and target=7target = 7target=7.

        [1, 2, 3, 5, 8]
        ↑           ↑
        left       right
        Let us initialize two indices, leftleftleft and rightrightright pointing to the first and last element respectively.

        When we look at the sum of first and last element, it is 1+8=91 + 8 = 91+8=9, which is ≥target\geq target≥target. 
        That tells us no index pair will ever contain the index rightrightright. 
        So the next logical step is to move the right pointer one step to its left.

        [1, 2, 3, 5, 8]
        ↑        ↑
        left    right
        Now the pair sum is 1+5=61 + 5 = 61+5=6, which is less than targettargettarget.
        How many pairs with one of the index=leftindex = leftindex=left that satisfy the condition? 
        You can tell by the difference between rightrightright and leftleftleft which is 333, 
        namely (1,2),(1,3),(1,2), (1,3),(1,2),(1,3), and (1,5)(1,5)(1,5). Therefore, we move leftleftleft one step to its right.


        class Solution {
            public int threeSumSmaller(int[] nums, int target) {
                Arrays.sort(nums);
                int sum = 0;
                for (int i = 0; i < nums.length - 2; i++) {
                    sum += twoSumSmaller(nums, i + 1, target - nums[i]);
                }
                return sum;
            }

            private int twoSumSmaller(int[] nums, int startIndex, int target) {
                int sum = 0;
                int left = startIndex;
                int right = nums.length - 1;
                while (left < right) {
                    if (nums[left] + nums[right] < target) {

                    // IMPORTANT
                    // TRICKY PART HERE
                    // IF THREESUM < TARGET, THEN BECAUSE THEE ARRAY IS SORTED
                    // ALL NUMBERS IN BETWEEN WILL ALSO BE LESS OR EQUAL TO K
                    // AND THEREFORE BE VALID ANSWERS

                        sum += right - left;
                        left++;
                    } else {
                        right--;
                    }
                }
                return sum;
            }
        }


-29.8) three sum counting (Valid Triangles.)

        611. Valid Triangle Number
        Attempted
        Medium
        Topics
        Companies
        Given an integer array nums, return the number of triplets chosen from 
        the array that can make triangles if we take them as side lengths of a triangle.

        Example 1:

        Input: nums = [2,2,3,4]
        Output: 3
        Explanation: Valid combinations are: 
        2,3,4 (using the first 2)
        2,3,4 (using the second 2)
        2,2,3
        Example 2:

        Input: nums = [4,2,3,4]
        Output: 4
        

        Constraints:

        1 <= nums.length <= 1000
        0 <= nums[i] <= 1000


        Approach 3: Linear Scan
        Algorithm

        As discussed in the last approach, once we sort the given numsnumsnums array, we need to find the right limit of the index kkk for a pair of indices (i,j)(i, j)(i,j) chosen to find the countcountcount of elements satisfying nums[i]+nums[j]>nums[k]nums[i] + nums[j] > nums[k]nums[i]+nums[j]>nums[k] for the triplet (nums[i],nums[j],nums[k])(nums[i], nums[j], nums[k])(nums[i],nums[j],nums[k]) to form a valid triangle.

        We can find this right limit by simply traversing the index kkk's values starting from the index k=j+1k=j+1k=j+1 for a pair (i,j)(i, j)(i,j) chosen and stopping at the first value of kkk not satisfying the above inequality. Again, the countcountcount of elements nums[k]nums[k]nums[k] satisfying nums[i]+nums[j]>nums[k]nums[i] + nums[j] > nums[k]nums[i]+nums[j]>nums[k] for the pair of indices (i,j)(i, j)(i,j) chosen is given by k−j−1k - j - 1k−j−1 as discussed in the last approach.

        Further, as discussed in the last approach, when we choose a higher value of index jjj for a particular iii chosen, we need not start from the index j+1j + 1j+1. Instead, we can start off directly from the value of kkk where we left for the last index jjj. This helps to save redundant computations.

        Thus, if we are able to find this right limit value of kkk(indicating the element just greater than nums[i]+nums[j]nums[i] + nums[j]nums[i]+nums[j]), we can conclude that all the elements in numsnumsnums array in the range (j+1,k−1)(j+1, k-1)(j+1,k−1)(both included) satisfy the required inequality. Thus, the countcountcount of elements satisfying the inequality will be given by (k−1)−(j+1)+1=k−j−1(k-1) - (j+1) + 1 = k - j - 1(k−1)−(j+1)+1=k−j−1.


            public class Solution {
                public int triangleNumber(int[] nums) {
                    int count = 0;
                    Arrays.sort(nums);
                    for (int i = 0; i < nums.length - 2; i++) {
                        int k = i + 2;
                        for (int j = i + 1; j < nums.length - 1 && nums[i] != 0; j++) {
                            while (k < nums.length && nums[i] + nums[j] > nums[k])
                                k++;
                            count += k - j - 1;
                        }
                    }
                    return count;
                }
            }




-29) 3 POINTER PERFORMANCE OPTIMIZATION FOR O(N^2)

     3SUM LEETCODE QUESTION/TRIANGLE NUMBERS LEETCODE 
    -> You can use 3 pointers, with middle pointer iterating through array, 
       and other 2 pointers moving left and right to find different numbers
       to GET AN O(N^2) SOLUTION INSTEAD OF AN O(N^3) SOLUTION.
                


-28) REVIEW BIDIRECTIONAL BFS PYTHON SOLUTION FOR Open the Lock in important questions. 

-27) 2 Pointers to delimit sequence ranges, and enforce loop invariants: 
     Use pointers to delimit correct and incorrect regions in sequences, and swap elements/process
     to correct the incorrect sequence

     MOVE ZEROS:
     Given an array nums, write a function to move all 0's to the end of it while 
     maintaining the relative order of the non-zero elements.
     OPTIMAL:
     
     the code will maintain the following invariant:
     All elements before the slow pointer (lastNonZeroFoundAt) are non-zeroes.
     All elements between the current and slow pointer are zeroes.
 
     Therefore, when we encounter a non-zero element, we need to swap elements 
     pointed by current and slow pointer, then advance both pointers. 
     If it's zero element, we just advance current pointer.
 
     void moveZeroes(vector<int>& nums) {
         for (int lastNonZeroFoundAt = 0, cur = 0; cur < nums.size(); cur++) {
             if (nums[cur] != 0) {
                 swap(nums[lastNonZeroFoundAt++], nums[cur]);
             }
         }
     }
                

-26) LINKED LIST CHANGE VALUES INSTEAD OF NODE RELATIONSHIPS STRATEGY 

     Delete a linked list node you have access to, but linked list is 
     singly linked and you dont have access to  parent. 
     
     Soln: copy the value of the next node to our node. Then delete the next node. 


-25) Check cycle in directed graph (REMEMBER TO PUSH AND POP OFF THE RUNNING
     RECURISIVE STACK/VISITED SET TO BE ABLE TO REPROCESS NODES ON DIFFERNT PATHS IN GRAPH):
     (TREASURY PRIME QUIZ)

    def isThereCycle(self, node, visited=set()):        
        visited.add(node)          
        kids = self.graph.get(node)
        if kids: 
            for kid in kids:
                # print("For parent, examining child", (node, kid))
                if kid in visited:
                    return True
                else:
                    result = self.isThereCycle(kid, visited)
                    if result == True:
                        return True 
        visited.remove(node)
        return False



-24) Different ways to backtrack:
    1) Use colors, either 2 with visited set, or 3 when you need to record times in DFS, 
    2) 3 colors is also similar to pushing and popping off a recursive stack to reprocess elements 
       you saw. This is used when you want to get all paths from source to target, so you need to
       reuse nodes/detect all cycles for instance.  

    ALL Paths from source to target:
    Given a directed, acyclic graph of N nodes.  
    Find all possible paths from node 0 to node N-1, and return them in any order.
    
    # DYNAMIC PROGRAMMING SOLUTION TOP-DOWN!! BY USING @lru_cache(maxsize=None)
    # THIS SOLUTION IS BAD BECAUSE WE ARE NOT USING DEQUE AND APPENDLEFT, 
    # LIST MERGING AND INSERTION TO FRONT IS O(N)!!!
    
    #The two approach might have the same asymptotic time 
    #complexity. However, in practice the DP approach is 
    #slower than the backtracking approach, since we copy the intermediate paths over and over.

    #Note that, the performance would be degraded further, 
    #if we did not adopt the memoization technique here.

    class Solution:
        def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
            
            # apply the memoization
            @lru_cache(maxsize=None)
            def dfs(node):
                
                if node == len(graph) - 1:
                    return [[len(graph) - 1]]
                
                kids  = graph[node]
                
                # all paths from node to target. 
                paths = []
                
                for kid in graph[node]:
                    res = dfs(kid)
                    # add node to front of each result!
                    for result in res:
                        paths.append([node] + result)
                
                return paths
            return dfs(0)

    # BETTER, ALSO USES A DIFF APPROACH OF PUSHING AND POPPING EACH PATH PIECE IN FOR LOOP
    class Solution:
        def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:

            target = len(graph) - 1
            results = []

            def backtrack(currNode, path):
                # if we reach the target, no need to explore further.
                if currNode == target:
                    results.append(list(path))
                    return
                # explore the neighbor nodes one after another.
                for nextNode in graph[currNode]:
                    path.append(nextNode)
                    backtrack(nextNode, path)
                    path.pop()
            # kick of the backtracking, starting from the source node (0).
            path = deque([0])
            backtrack(0, path)

            return results



-23) Work with differences and gradients instead of RAW VALUES!!
    It helps to solve the problem. Simplify problems by allocating 
    everything to one group, then pulling the correct ones to the other group. 
    Use slopes, intercepts, and think of problems geometrically when 
    preprocessing AKA two city scheduling


-22) Flatten binary tree to linked list. 
     Given a binary tree, flatten it to a linked list in-place.
     Use right nodes when creating linked list. 
     CAN DO THIS WITH O(1) SPACE LIKE SO:
  
     So what this solution is basically doing is putting the 
     right subtree next to the rightmost node on the left subtree 
     and then making the left subtree the right subtree and 
     then making the left one null. Neat!
     
    class Solution:
        # @param root, a tree node
        # @return nothing, do it in place
        def flatten(self, root):
            if not root:
                return
            
            # using Morris Traversal of BT
            node=root
            
            while node:
                if node.left:
                    pre=node.left
                    while pre.right:
                        pre=pre.right
                    pre.right=node.right
                    node.right=node.left
                    node.left=None
                node=node.right

-21.5) DP SOLUTION TO LIS:

    class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    Compare this with nlogn soln for LIS...



-21) THE DUTCH PARTITIONING ALGORITHM WITH 3 POINTERS. 
    Using 3 pointers Algorithm  + Abusing LOOP invariants 
    The trick is to move the MIDDLE POINTER FROM LEFT TO RIGHT AND 
    use the left and right pointers to delimit the correctly processed sequence!

    Otherwise, try moving the left pointer right to left or the left pointer, 
    or try adding another pointer. Add as many pointers until the problem seems simple 
    and we have untangled all the SEPERATE CONCERNS. 

    Given an array with n objects colored red, white or blue, 
    sort them in-place so that objects of the same color 
    are adjacent, with the colors in the order red, white and blue.

    Here, we will use the integers 0, 1, and 2 to represent the 
    color red, white, and blue respectively.

    Note: One pass algorithm with constant space only. 

    Example:
    Input: [2,0,2,1,1,0]
    Output: [0,0,1,1,2,2]    


    def sortColors(self, nums: List[int]) -> None:
        i = 0
        l = 0         
        j = len(nums) - 1
        
        while l != len(nums):
            if nums[l] == 1:
                l += 1
            elif nums[l] == 0 and i == l:
                l += 1
            elif nums[l] == 0:
                nums[l], nums[i] = nums[i], nums[l]
                i += 1 
            elif nums[l] == 2 and l >= j:
                l += 1
            elif nums[l] == 2:
                nums[l], nums[j] = nums[j], nums[l]
                j -= 1
        return nums


    INVARIANTS ABUSE FOR SIMPLIFICATION: 

    nums[0:red] = 0, nums[red:white] = 1, nums[white:blue + 1] = unclassified, 
    nums[blue + 1:] = 2.
    The code is written so that either 
    (red < white and nums[red] == 1) or (red == white) at all times.
    Think about the first time when white separates from red. 
    That only happens when nums[white] == 1, 
    so after the white += 1, 
    we have nums[red] == 1, and notice that nums[red] 
    will continue to be 1 as long as 
    red != white. This is because red only gets 
    incremented in the first case 
    (nums[white] == 0), so we know that we are swapping nums[red] == 1 with nums[white] == 0.

 
    def sortColors(self, nums):
        red, white, blue = 0, 0, len(nums)-1
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                # We could have never swapped blue in because we are going left to right and the blue if statement would hit first.
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1


-20) Hill finding part 1: Best time to buy and sell stock I
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        min_i = prices[0]
        max_profit = 0
        for i,x in enumerate(prices):
            min_i = min(min_i, x)
            profit = x - min_i
            max_profit = max(max_profit, profit)
        return max_profit

    CAN ALSO SOLVE WITH KADANES ALGORITHM (Max subarray sum):
    if the interviewer twists the question slightly by giving the difference 
    array of prices, Ex: for {1, 7, 4, 11}, 
    if he gives {0, 6, -3, 7}, you might end up being confused. Do this:

        public int maxProfit(int[] prices) {
        int maxCur = 0, maxSoFar = 0;
        for(int i = 1; i < prices.length; i++) {
            maxCur = Math.max(0, maxCur += prices[i] - prices[i-1]);
            maxSoFar = Math.max(maxCur, maxSoFar);
        }
        return maxSoFar;
    }


-19)VISULIZING PROBLEMS, GET GEOMETRIC UNDERSTANDING OF TEST CASES, AND DO NOT 
    RUSH THE ALGORITHM DESIGN PHASE. TEST YOUR ALGO ON TEST CASES BEFORE WRITING. 
    ESP IF YOU ARE UNSURE!!

    BEST TIME TO BUY AND SELL STOCK 2
    ALWAYS TRY TO TEST YOUR ALGORITHM AGAINST TEST CASES!!!! for UNDERSTANDING
    before coding out!! YOUR ALGORITHM WILL BE INCORRECT SOMETIMES. 
    ALSO think of invarients that should be true, and dont break invarients you invent. 
     
    When you design an algo, make sure you understand 
    all the details of algo before you code it out. It may look like a hill finder, 
    but is actually a RIEMAN SUM INTEGRAL algorithm!! 
    Like with this qustion which you
    messed up the first time, doing. correct soln under it. 

    Say you have an array prices for which the ith element is the price of a given stock on day i.
    Design an algorithm to find the maximum profit. You may complete as many transactions as you like

    class Solution:
        def maxProfitWRONG(self, prices: List[int]) -> int:
            '''
            IF YOU WERE SMART, YOU WOULD realized you have to consider 
            every peak and valley to create a profit value. And this realization
            comes from playing around with test cases and writing out problem ideas,
            and then testing problem ideas. 
            '''

            i = 0
            buyPrice = None
            sellPrice = None            
            profit = 0

            while i < len(prices):
                if buyPrice is None:
                    buyPrice = prices[i]
                elif prices[i] < buyPrice:
                    if sellPrice:
                        profit += (sellPrice - buyPrice)                    
                    buyPrice = prices[i]
                    sellPrice = None
                else:
                    if not sellPrice:
                        sellPrice = prices[i]
                    elif sellPrice < prices[i]:
                        sellPrice = prices[i]
                i += 1
            if buyPrice and sellPrice:
                profit += (sellPrice - buyPrice)
            return profit
        
        def maxProfit(self, prices: List[int]) -> int:
                # CORRECT WAY WITH INTEGRAL SUM INSTEAD OF HILL FINDING
                profit = 0
                prev = prices[0]
                for i in range(1, len(prices)):
                    if prices[i] > prev:
                        profit += (prices[i] - prev)
                    prev = prices[i]
                return profit



-18) Sorting algorithms and true space complexity. 

    Quicksort: For quicksort, your intuition about recursion requiring O(log(n)) space is correct. 
    Since quicksort calls itself on the order of log(n) times (in the average case, worst 
    case number of calls is O(n)), at each recursive call a new stack frame of constant 
    size must be allocated. Hence the O(log(n)) space complexity.

    Mergesort: Since mergesort also calls itself on the order of log(n) times, 
    why the O(n) space requirement? The extra space comes from the merge operation. 
    Most implementations of merge use an auxiliary array with length equal to the 
    length of the merged result, since in-place merges are very complicated. 
    In other words, to merge two sorted arrays of length n/2, most merges will 
    use an auxiliary array of length n. The final step of mergesort 
    does exactly this merge, hence the O(n) space requirement.


    Merge sort on linked lists can be executed using only O(1) extra space if 
    you take a bottom-up approach by counting where the boundaries of 
    the partitions are and merging accordingly.


-17.9)  For the GREEDY CANDY PROBLEM. You are given constraints to respect. 
        The constraints are hard unless you break them up!
        ALSO try to understand the problem before you start:
        review it in most important problems.
        
        There are N children standing in a line. 
        Each child is assigned a rating value.

        You are giving candies to these children 
        subjected to the following requirements:

        Each child must have at least one candy.
        Children with a higher rating get more candies than their neighbors.
        What is the minimum candies you must give?
        Input: [1,0,2] 
        Output: 2, 1, 2  -> 5
        Input: [1,2,2]
        Output: 1, 2, 1 -> 4

        def candy(self, ratings):
            if not ratings:
                return 0

            n = len(ratings)
            candy = [1] * n
            for i in range(1, n):
                if ratings[i] > ratings[i - 1]:
                    candy[i] = candy[i - 1] + 1
                
            for i in range(n - 2, -1, -1):
                if ratings[i] > ratings[i + 1] and candy[i] <= candy[i + 1]:
                    candy[i] = candy[i + 1] + 1

            return sum(candy)

        You can also solve with constant space by 
        looking at rising and falling slopes

        EXPLANATION:

        Approach 4: Single Pass Approach with Constant Space
        Algorithm

        This approach relies on the observation (as demonstrated in the figure below as well) that in order to distribute the candies as per the given criteria using the minimum number of candies, the candies are always distributed in terms of increments of 1. Further, while distributing the candies, the local minimum number of candies given to a student is 1. Thus, the sub-distributions always take the following form: 1, 2, 3, ..., n\text{1, 2, 3, ..., n}1, 2, 3, ..., n or n,..., 2, 1\text{n,..., 2, 1}n,..., 2, 1. Which, can simply be added using the formula n(n+1)/2n(n+1)/2n(n+1)/2.

        Now, we can view the given rankings as some rising and falling slopes. Whenever the slope is rising, the distribution takes the form: 1, 2, 3, ..., m\text{1, 2, 3, ..., m}1, 2, 3, ..., m. Similarly, a falling slope takes the form: k,..., 2, 1\text{k,..., 2, 1}k,..., 2, 1. A challenge that arises now is that the local peak point can be included in only one of the slopes. Should we include the local peak point, n, in the rising slope or the in falling slope?

        In order to decide, we can observe that in order to satisfy both the right neighbor and the left neighbor criteria, the peak point's count needs to be the max. of the counts determined by the rising and the falling slopes. Thus, in order to determine the number of candies required, the peak point needs to be included in the slope which contains more number of points. The local valley point can also be included in only one of the slopes, but this issue can be resolved easily, since the local valley point will always be assigned a candy count of 1 (which can be subtracted from the next slope's count calculations).

        Coming to the implementation, we maintain two variables oldSlope and newSlope to determine the occurrence of a peak or a valley. We also use up and down variables to keep a track of the count of elements on the rising slope and on the falling slope respectively (without including the peak element). We always update the total count of candies at the end of a falling slope following a rising slope (or a mountain). The leveling of the points in rankings also works as the end of a mountain. At the end of the mountain, we determine whether to include the peak point in the rising slope or in the falling slope by comparing the up and down variables up to that point. Thus, the count assigned to the peak element becomes: max(up, down) + 1. At this point, we can reset the up and down variables indicating the start of a new mountain.

        class Solution:
            def candy(self, ratings):
                if not ratings:
                    return 0

                count = up = down = 1

                for i in range(1, len(ratings)):
                    if ratings[i] >= ratings[i - 1]:
                        if down > 1:
                            count -= min(down, up) - 1
                            up, down = 1, 1
                        up = ratings[i] == ratings[i - 1] or up + 1
                        count += up
                    else:
                        down += 1
                        count += down

                if down > 1:
                    count -= min(down, up) - 1

                return count

        Another constant space soln:

            def candy(self, ratings):
                if len(ratings) <= 1:
                    return len(ratings)
                candies = 0
                up = 0
                down = 0
                oldSlope = 0
                for i in range(1, len(ratings)):
                    newSlope = (
                        1
                        if ratings[i] > ratings[i - 1]
                        else (-1 if ratings[i] < ratings[i - 1] else 0)
                    )
                    # slope is changing from uphill to flat or downhill
                    # or from downhill to flat or uphill
                    if (oldSlope > 0 and newSlope == 0) or (
                        oldSlope < 0 and newSlope >= 0
                    ):
                        candies += self.count(up) + self.count(down) + max(up, down)
                        up = 0
                        down = 0
                    # slope is uphill
                    if newSlope > 0:
                        up += 1
                    # slope is downhill
                    elif newSlope < 0:
                        down += 1
                    # slope is flat
                    else:
                        candies += 1
                    oldSlope = newSlope
                candies += self.count(up) + self.count(down) + max(up, down) + 1
                return candies


-17.8)  BE SMART ABOUT GRAPH ROOT FINDING, AND ITERATING:
        
        1.   Longest Consecutive Sequence
        
        Given an unsorted array of integers, find the length of the 
        longest consecutive elements sequence.
        Your algorithm should run in O(n) complexity.


        def longestConsecutive(self, nums):
            nums = set(nums)
            best = 0
            for x in nums:
                if x - 1 not in nums:
                    y = x + 1
                    while y in nums:
                        y += 1
                    best = max(best, y - x)
            return best


-17.7) Intersection of 2 linked lists. 
    Write a program to find the node at which the 
    intersection of two singly linked lists begins.
    (Constant space)

    Find the different in 2 lists, then traverse longer one 
    shifted by difference, and other, one node at a time.
    When nodes are equal that is the intersection node. 

    Other soln:
        def getIntersectionNode(self, headA, headB):
            if headA is None or headB is None:
                return None

            pa = headA # 2 pointers
            pb = headB

            while pa is not pb:
                # if either pointer hits the end, 
                # switch head and continue the second traversal, 
                # if not hit the end, just move on to next
                pa = headB if pa is None else pa.next
                pb = headA if pb is None else pb.next

            return pa 
            # only 2 ways to get out of the loop, 
            # they meet or the both hit the end=None

    the idea is if you switch head, the possible difference 
    between length would be countered. On the second traversal, 
    they either hit or miss. if they meet, pa or pb would 
    be the node we are looking for, 
    if they didn't meet, they will hit the end at 
    the same iteration, pa == pb == None, 
    return either one of them is the same,None




-17.6) Copy list with random pointer 
       (associate input structure with output structure)
       then recover both after trick. 
    
    A linked list is given such that each node contains an 
    additional random pointer which could point to any node in the list or null.

    Return a deep copy of the list.

    We need a hash map here to map to random nodes in 
    our new linked list. This requires O(n) space

    We can use constant space (if we do not consider space
    for output)
    
    The idea is to associate the original node with its 
    copy node in a single linked list. In this way, 
    we don't need extra space to keep track of the new nodes.

    The algorithm is composed of the follow three steps which are also 3 iteration rounds.

    Iterate the original list and duplicate each node. The duplicate
    of each node follows its original immediately.
    Iterate the new list and assign the random pointer for each
    duplicated node.
    Restore the original list and extract the duplicated nodes.

    def copyRandomList(self, head):

        # Insert each node's copy right after it, already copy .label
        node = head
        while node:
            copy = RandomListNode(node.label)
            copy.next = node.next
            node.next = copy
            node = copy.next

        # Set each copy's .random
        node = head
        while node:
            node.next.random = node.random and node.random.next
            node = node.next.next

        # Separate the copied list from the original, (re)setting every .next
        node = head
        copy = head_copy = head and head.next
        while node:
            node.next = node = copy.next
            copy.next = copy = node and node.next

        return head_copy


    @DrFirestream OMG is that a mindfuck :-). But a nice thing is that the original 
    list's next structure is never changed, so I can write a helper generator to 
    visit the original list with a nice for loop encapsulating the while loop 
    and making the loop bodies a little simpler:

    '''
    def copyRandomList(self, head: 'Node') -> 'Node':
        def nodes():
            node = head
            while node:
                yield node
                node = node.next
        # create new nodes
        for node in nodes():
            node.random = Node(node.val, node.random, None)
        # populate random field of the new node
        for node in nodes():
            node.random.random = node.random.next and node.random.next.random
        # restore original list and build new list
        head_copy = head and head.random
        for node in nodes():
            node.random.next, node.random = node.next and node.next.random, node.random.next
        return head_copy

-17.5) AVOID LINKED LIST LOOPS IN YOUR CODE. ALWAYS 
       NULLIFY YOUR POINTERS IF YOU ARE REUSING THE 
       DATASTRUCTURE/ DOING THINGS IN PLACE!!!!!!
       SUCH AS HERE by saving nxt pointer as tmp

       1.   Odd Even Linked List
       Given a singly linked list, group all odd nodes 
       together followed    by the even nodes. 
   
       You should try to do it in place. The program should run in O(1)    
       space complexity and O(nodes) time complexity.

        def oddEvenList(self, head: ListNode) -> ListNode:
            oddH = ListNode(0)
            evenH = ListNode(0)
            
            odd = oddH
            even = evenH
            
            isOdd = True
            node = head
            
            while node:
                nxt = node.next  # SAVE THE NEXT NODE FOR FUTURE USE THEN NULLIFY IT TO STOP LOOPS
                node.next = None # STOP THE LOOPS
                if isOdd:
                    odd.next = node
                    odd = odd.next
                    isOdd = False
                else:
                    even.next = node
                    even = even.next
                    isOdd = True
                node = nxt
            
            odd.next = evenH.next
            return oddH.next




-17.4) IMPLEMENTED QUICK SORT FOR LINKED LISTS:
            
    # USE MORE DUMMY NODES TO SEPERATE CONCERNS, AND REDUCE MISTAKES
    # I HAD A PROBLEM WITH CYCLES BECAUSE I WAS REUSING NODES.

    def sortList(self, head: ListNode) -> ListNode:
        '''
        random.randint(a, b)¶
        Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1).
        '''
        
        node = head
        l = 0
        while node:
            l += 1
            node = node.next
            
        def partition(head, start, end, pivot):
            if start == end:
                return head
            if head is None:
                return None
            
            pivotVal = pivot.val
            before = ListNode(0)
            after = ListNode(0)
            afterCopy = after
            beforeCopy = before
            
            temp = head
            left_len = 0
            
            while temp:       
                # print("processing temp", temp.val)
                if temp == pivot:
                    temp = temp.next
                    continue
                    
                if temp.val < pivotVal: 
                    left_len += 1
                    before.next = temp
                    before = before.next
                    temp = temp.next
                else:
                    after.next = temp
                    after = after.next
                    temp = temp.next
                        
            before.next = None
            after.next = None
            return beforeCopy.next, left_len, afterCopy.next
 
        def quicksort(head, start, end):
            if head is None:
                return None
            
            if end-start <= 1:
                return head 
            
            pivotLoc = random.randint(start, end-1)            
            pivot = head
            i = 0
            while i < pivotLoc:
                pivot = pivot.next
                i += 1
                
            if pivot is None:
                return None
               
            left, left_len, right = partition(head, start, end, pivot) 
            sorted_left = quicksort(left, 0, left_len)
            sorted_right = quicksort(right, 0, end - left_len - 1)
            
            if sorted_left:
                temp = sorted_left
                while temp and temp.next:
                    temp = temp.next
                temp.next = pivot
            else:
                sorted_left = pivot

            pivot.next = sorted_right
            return sorted_left
        
        return quicksort(head, 0, l)


-17.3 Bottom Up Merge Sort for Linked List (O(1) space )
    class Solution {
    public:
        ListNode *sortList(ListNode *head) {
            if(!head || !(head->next)) return head;
            //get the linked list's length
            ListNode* cur = head;
            int length = 0;
            while(cur){
                length++;
                cur = cur->next;
            }
            
            ListNode dummy(0);
            dummy.next = head;
            ListNode *left, *right, *tail;
            for(int step = 1; step < length; step <<= 1){
                cur = dummy.next;
                tail = &dummy;
                while(cur){
                    left = cur;
                    right = split(left, step);
                    cur = split(right,step);
                    tail = merge(left, right, tail);
                }
            }
            return dummy.next;
        }
    private:
        ListNode* split(ListNode *head, int n){
            //if(!head) return NULL;
            for(int i = 1; head && i < n; i++) head = head->next;
            
            if(!head) return NULL;
            ListNode *second = head->next;
            head->next = NULL;
            return second;
        }
        ListNode* merge(ListNode* l1, ListNode* l2, ListNode* head){
            ListNode *cur = head;
            while(l1 && l2){
                if(l1->val > l2->val){
                    cur->next = l2;
                    cur = l2;
                    l2 = l2->next;
                }
                else{
                    cur->next = l1;
                    cur = l1;
                    l1 = l1->next;
                }
            }
            cur->next = (l1 ? l1 : l2);
            while(cur->next) cur = cur->next;
            return cur;
        }
    };



-17) Storing 2 integer values at same index in an array:

    First we have to find a value greater than 
    all the elements of the array. Now we can store the 
    original value as modulus and the second value as division. 
    Suppose we want to store arr[i] and arr[j] both at index 
    i(means in arr[i]). First we have to find a ‘maxval’ 
    greater than both arr[i] and arr[j]. Now we can store 
    as arr[i] = arr[i] + arr[j]*maxval. Now arr[i]%maxval 
    will give the original value of arr[i] and arr[i]/maxval 
    will give the value of arr[j].


-16) Modified Bin Search: Find Minimum in Rotated Sorted Array
    
    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
    (i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).
    Find the minimum element.
    You may assume no duplicate exists in the array.
    
    This soln is abusing the fact that 
    left side is bigger than right side,
    for all rotated cases. 
    [3,4,5,1,2]

    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while nums[left] > nums[right]:
            middle  = (left + right) // 2
            if nums[middle] < nums[right]:
                right = middle
            else:
                left = middle + 1
        return nums[left]

    My soln didnt use this fact. it instead took the first 
    element as a reference, and then cut the middle, check left
    and right element for pivot, and binary searched left or right
    based on comparing referenced element to middle element. 
    In other words. I gutta SLOW DOWN, UNDERSTAND PROBLEM,
    EXPLOIT MORE CONSTRAINTS, LOOK AT PROBLEM STRUCTURE,
    IDENTIFY CORE TRUTHS FOR ALL CASES!!!

    




-15) Wierd~Modified~Customized Memoization / Tree Memoization Pattern:      
     Path Sum III
     You are given a binary tree in which 
     each node contains an integer value.
     Find the number of paths that sum to a given value.
     The path does not need to start or end at the root or 
     a leaf, but it must go downwards 
     (traveling only from parent nodes to child nodes).
     
    IDEA 1: You need two recursive functions. SEPERATE CONCERNS. 
          DONT TRY TO DO IT ALL IN ONE RECURSIVE FUNCTION!
          WATCH FOR WEIRD BASE CASES -> ESP when you get errors
          When debugging look at base cases -> think about if you
          need more recursive functions -> because having it all in 
          one recursive function may lead 4 cases and 2 cases are undesirable
    
        class Solution:
            def verifySum(self, node, val):         
                if node is None:
                    return 0
                count = 0
                if node.val == val:
                    count = 1    
                verifyRight = self.verifySum(node.right, val - node.val)
                verifyLeft = self.verifySum(node.left, val - node.val)
                count += (verifyRight + verifyLeft)    
                return count
            
            def pathSum(self, root: TreeNode, sum: int) -> int: 
                if root is None:
                    return 0   
                count = 0
                count = self.verifySum(root, sum)
                right = self.pathSum(root.right, sum)
                left =  self.pathSum(root.left, sum)
                count += right
                count += left
                return count     
        
    IDEA 2: Memoization -> When do tree recursion, check for repeat solutions.
            Here solution is like 2-SUM, and hash map abuse. 
            The 50% Top 50 % Bottom Memoization Pattern

         why initialize cache = {0:1} why not just empty dictionary?
            Initializing the cache to {0:1} allows us to consider the path starting 
            from root. Another way to consider this path 
            would be to check the value of currPathSum directly:

        class Solution:
            def dfs(self,node,isum,target):
                if node ==None:
                    return
                nxtSum = isum + node.val
                
                # THIS IS EXACTLY 2 SUM. WE ARE LOOKING FOR THE DIFFERENCE!
                # BECAUSE WE WANT TO TRY THE SUMS BETWEEN ANY 2 NODES, SO WE 
                # GUTTA COUNT EM!
                if nxtSum - target in self.map:
                    self.count += self.map[nxtSum - target]
                
                if nxtSum not in self.map:
                    self.map[nxtSum] = 1
                else:    
                    self.map[nxtSum] += 1
                
                self.dfs(node.left,nxtSum,target)
                self.dfs(node.right,nxtSum,target)
                
                # WE NO LONGER HAVE THIS NODE TO CREATE A PATH SEGMENT, SUBTRACT IT OUT. 
                # BECAUSE WE SWITCHED BRANCHES
                self.map[nxtSum] -= 1
    
            def pathSum(self, root: TreeNode, sum: int) -> int:
                self.map = {}
                self.map[0] = 1
                self.count = 0
                self.dfs(root,0,sum)
                return self.count


-14) Learn to use iterators: Serialize and Deserialize bin tree preorder style:

    class Codec:
        def serialize(self, root):
            def doit(node):
                if node:
                    vals.append(str(node.val))
                    doit(node.left)
                    doit(node.right)
                else:
                    vals.append('#')
            vals = []
            doit(root)
            return ' '.join(vals)

        def deserialize(self, data):
            def doit():
                val = next(vals)
                if val == '#':
                    return None
                node = TreeNode(int(val))
                node.left = doit()
                node.right = doit()
                return node
            vals = iter(data.split())
            return doit()




-13) GREEDY HILL FINDING WITH REVERSE POINTERS, 
     AKA MOST IMPORTANT INDEXES ONLY FINDING AND USING SMARTLY 
     AKA MONOQUEUE EXTENSION

    Some problems require you to find optimal hills, to get answer. 
    These hills are valid for certain indexes, and then you have to use new hills
    They have a sort of max min aura to them, and seem similar to monoqueue type 
    problems.
    When you see a max-min type optimization pattern, then you have to find HILLS:
    
    For instance:
    Input a = [21,5,6,56,88,52], output = [5,5,5,4,-1,-1] . 

    Output array values is made up of indices of the 
    element with value greater than the current element 
    but with largest index. So 21 < 56 (index 3), 
    21 < 88 (index 4) but also 21 < 52 (index 5) 
    so we choose index 5 (value 52). 
    Same applies for 5,6 and for 56 its 88 (index 4).
    
    Algorithm 1: Find the hills, and binsearch the indexes: 

    need to keep track of biggest element on right side. 
    on the right side, keep the hills!
    52, is a hill, 
    then 88, because its bigger than 52,
    not 56, not 6, not 5, not 21, because you can just use 52, or 88 
    so elements check against 52 first, then against 88. 
    
    import bisect
    def soln(arr):
        hills = []
        hill_locations = []
        running_max = float("-inf")
        for i in range(len(arr)-1, -1, -1):
            if running_max < arr[i]:
                running_max = arr[i]
                hills.append(arr[i])
                hill_locations.append(i)
        hill_locations_pop_idx = hill_locations[-1]
        ans = []

        def bin_search(arr, val):
            l = 0
            r = len(arr) 
            mid = None
            while l != r:
                mid = l + (r-l)//2
                if arr[mid]  == val:
                    return mid 
                elif arr[mid] > val:
                    r = mid 
                else:
                    l = mid  + 1
            return l
        
        for i in range(len(arr)):
            if i == hill_locations_pop_idx:
                # you have to invalidate indexes because you dont want to 
                # invalid indexes to be found in bin search.
                hill_locations.pop()
                hills.pop()
                hill_locations_pop_idx = -1 if len(hill_locations) == 0 else hill_locations[-1]
            # Locate the insertion point for x in a to maintain sorted order.
            x = bisect.bisect_left(hills, arr[i], lo=0, hi=len(hills))
            y = bin_search(hills, arr[i])
            print("x, y", (x, y)) # will be same
            if y < len(hill_locations):
                ans.append(hill_locations[x])
            else:
                ans.append(-1)
        return ans  

    Algorithm 2: Insert everything in pq. Pop off 1 by 1, check running max idx. and assign idx. 






-12) SIMULATE BINARY SEARCH INSERTION POINT FINDER 
     AKA bisect.bisect_left(arr, val, lo=0, hi=len(arr)) 
    # Locate the insertion point for x in a to maintain sorted order.
    # REMEMBER THAT THE FINAL ANSWER IS LOW NOTTTTT MID

    # HERE WE INITIALIZED RIGHT AS LEN(NUMS) - 1
    def searchInsert(self, nums, target):
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low

    # HERE WE INITIALIZED RIGHT AS LEN(NUMS) KNOW THE DIFFERENCE. 
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        l = 0
        r = len(nums)
        mid = None
        
        while l != r:            
            mid = l + (r-l)//2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid

        # DO NOT RETURN MID, RETURN L
        return l




-11) Merkle Hashing and Tree to String Methods:

    Given two non-empty binary trees s and t, check whether tree 
    t has exactly the same structure and node values with a subtree of s.
    A subtree of s is a tree consists of a node in s and all of this node's 
    descendants. The tree s could also be considered as a subtree of itself.

    Normal way runtime is O(|s| * |t|)
    Runtime O(|s| + |t|) (Merkle hashing):

    For each node in a tree, we can create node.merkle, 
    a hash representing it's subtree. This hash is formed by hashing the 
    concatenation of the merkle of the left child, the node's value, 
    and the merkle of the right child. Then, two trees are identical if 
    and only if the merkle hash of their roots are equal (except when 
    there is a hash collision.) From there, finding the answer is straightforward: 
    we simply check if any node in s has node.merkle == t.merkle

    def isSubtree(self, s, t):
        from hashlib import sha256
        def hash_(x):
            S = sha256()
            S.update(x)
            return S.hexdigest()
            
        def merkle(node):
            if not node:
                return '#'
            m_left = merkle(node.left)
            m_right = merkle(node.right)
            node.merkle = hash_(m_left + str(node.val) + m_right)
            return node.merkle
            
        merkle(s)
        merkle(t)
        def dfs(node):
            if not node:
                return False
            return (node.merkle == t.merkle or 
                    dfs(node.left) or dfs(node.right))
                        
        return dfs(s)
    
    QA below:
    Soln doesnt check for hash collisions but we use hash resistant function:
    For practical purposes, we can assume that there will not be a hash collision, 
    as the probability of a collision will be in the order of |S|^2 / 2^256. 
    A computer can do a sha256 hash in about 1 microsecond, 
    but sha256 hashes are technically proportional to their input length, 
    and you would be hashing hex digest (32 bytes each) as well as the node.val strings.

    For this problem though, collision resistant hash functions like sha256 
    are not necessary, from performance perspective. You can use some 
    computationally cheaper hash functions. With an addition of O(|T|) 
    checking every time hash values match, correctness is also made sure.

    Convert Trees to Strings Method F strings:
    Basically we convert our tree into string representation, 
    then just check whether substring exists in target string.
    
    >>> f"Hello, {name}. You are {age}."
    'Hello, Eric. You are 74.'

    class Solution:
        def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
            string_s = self.traverse_tree(s)
            string_t = self.traverse_tree(t)
            if string_t in string_s:
                return True
            return False
        
        
        def traverse_tree(self, s):
            if s:
                return f"#{s.val} {self.traverse_tree(s.left)} {self.traverse_tree(s.right)}"
            return None




-10) Learn how to index for binsearch. 
     Left index, getting mid, and right index is always boundary (so len(arr))

    # Given an array where elements are sorted in ascending order, 
    # convert it to a height balanced BST.

    class Solution:
        def sortedArrayToBST(self, nums: List[int]) -> TreeNode:            
            def build_tree(l, r):
                if(l == r):
                    return None
                
                mid =  l + (r-l)//2
                root = nums[mid]
                
                # you never include right value
                left = build_tree(l, mid)
                right = build_tree(mid+1, r)
                return TreeNode(val=root, left=left, right=right)
                
            return build_tree(0, len(nums))



-9 Remember that you can do in-order and post-order to help you do
   tree problems such as validate bst: 
   (i did it pre-order, by also keep track of the range)

    def isValidBST(self, root):
        res, self.flag = [], True
        self.helper(root, res)
        return self.flag
    
    def helper(self, root, res):
        if root:
            self.helper(root.left, res)
            if res and root.val <= res[-1]:
                self.flag = False
                return
            res.append(root.val)
            self.helper(root.right, res)

    
-8) Dynamic programming -> check if they want permutations or combinations.  
    The DP needs to change so that this invarient is 
    maintained such as in Coin Change 2,
    
    THIS IS A COMBINATIONS DP PROBLEM. 
    Input: amount = 5, coins = [1, 2, 5]
    Output: 4
    Explanation: there are four ways to make up the amount
    with the denominations. 
    DONT DO PERMUTATIONS DP. 
    2 + 2 + 1 IS THE SAME AS 2 + 1 + 2, so forloop over coins first
    so we dont reuse the same denomiation twice aka:

    class Solution(object):
        def change(self, amount, coins):
            dic = {0: 1}
            for coin in coins:
                for j in range(amount + 1):
                    dic[j] = dic.get(j, 0) +  dic.get(j - coin, 0)
            return dic.get(amount, 0)
    
    THIS IS A PERMUATIONS DP PROBLEM, DONT DO COMBINATIONS:
    Given an integer array with all positive numbers and no duplicates, 
    find the number of possible combinations that add up to a positive integer target.

    nums = [1, 2, 3]
    target = 4

    The possible combination ways are:
    (1, 1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 3), (2, 1, 1), (2, 2), (3, 1)

    def combinationSum4(self, nums: List[int], target: int) -> int:        
        amounts = [0 for _ in range(target + 1)]
        amounts[0] = 1 # When you reach amount 0, yield 1, for the coin, base case
        
        for amt in range(1, target+1):
            for coin in nums: 
                if coin <= amt: 
                    amounts[amt] += amounts[amt-coin]
                    
        return amounts[target]

    What if negative numbers are allowed in the given array?
    How does it change the problem?
    What limitation we need to add to the question to allow negative numbers?
    We should bound the length because solutions can have infinite length!!!
    Code to follow up: 
 
    def combinationSum4WithLength(self, nums, target, length, memo=collections.defaultdict(int)):
        if length <= 0: return 0
        if length == 1: return 1 * (target in nums)
        if (target, length) not in memo: 
            for num in nums:
                memo[target, length] += self.combinationSum4(nums, target - num, length - 1)
        return memo[target, length]




    
-7) CONTINUE TO PUSH FACTS AND DETAILS INTO A SOLUTION, ABUSE TO MAKE IT FASTER. 
    Longest increasing subsequence can be solved with patience sort using NlogN. 
    logN time to find pile and insert in pile. To come up with this method, 
    look at your algorithm, realize what facts its using, realize if there 
    are facts that you know the algorithm is not using but are true, 
    then use those facts to ENHANCE YOUR SOLUTION, permutate the facts, look
    for directionality in the facts, force it in with datastructures, and try to 
    be clean as you do so. 
    
    # Lacks bin search below but one after adds it. 
    The idea is to make piles as you traverse the array. The first pile is the 
    first number in the array. As you iterate over the array, you check from 
    the left most pile and if the current number is <= to the pile it can be 
    placed there (greedy if I fits I sits rule) Otherwise, continue down the pile. 
    If it can not fit on any pile you create a new one.

    class Solution:
        def lengthOfLIS(self, nums: List[int]) -> int:
            if not nums:
                return 0
            # start with first pile as first num
            piles = [nums[0]]
            
            for num in range(1, len(nums)):
                # keep track if current number is placed
                placed= False
                # traverse the piles being greedy left to right. If it fits it sits
                for x in range(len(piles)):
                    if nums[num] <= piles[x]:
                            piles[x] = nums[num]
                            placed = True
                            break
                # Make home for the number if it didn't find one :(
                if not placed:
                    piles.append(nums[num])
            return len(piles)

        Look at this problem: [10, 9, 2, 5, 3, 7, 101, 18]
        ->  Soln -> [2, 3, 7, 101]
            
    # SOLUTION WITH BINSEARCH      
    def lengthOfLIS(self, l: List[int]) -> int:
        if not l: return 0
        
        # Create a placeholder for each pile. In the worst case, 
        # the number of piles is the number of items in the list.
        topOfEachPile = [0] * len(l)
        
        # From the deck/videos, we should know that  the Patience Algorithm is Greedy. 
        # This results in the fewest number of piles possible.
        # The LIS is then the number of piles that exist.
        # Here we create a variable that describes the number 
        # of piles that we have initialised from our placeholder above.
        numberOfPiles = 0
        
        # Iterate over each number. For each number, do binary 
        # search to figure out which of the piles to place the number.
        for n in l:
            # These variables set the range of the binary search. 
            # We only want to do BS on the piles that have been initialised.
            # We include, at the very right, a new pile. This is useful 
            # because if the n can't fit into any of the existing 
            # piles we have to add it into this new pile.
            beg, end = 0, numberOfPiles
        
            # This BS is where we are greedy. If n is the same as l[middle] or less, we go left. 
            while beg != end:
                middle = (beg + end) // 2
                if n > topOfEachPile[middle]:
                    beg = middle + 1
                else:
                    end = middle
            
            # Update the top card at this pile.
            topOfEachPile[beg] = n
            
            # If we did end up using a new pile, then beg == numberOfPiles. 
            if beg == numberOfPiles: numberOfPiles += 1
        
        return numberOfPiles

-6) Review linked list 2, reversing a linked list between integers m and n 
   and how to use recursive stack and nonlocal variables to
   access backpointers in singly linked list. 
   Also how to use dummy pointers to simplify code 
   at the start.
   and always check before doing .next to stop null errors. 
   Iterative soln:

        '''
        When we are at the line pre.next.next = cur 
        the LL looks like this for [1,2,3,4,5] m = 2, n = 4
        we want: 1->4->3->2->5
        we have: 1 -> 2 <- 3 <- 4 5
 
        Note that there is no connection between 4 and 5, 
        here pre is node 1, reverse is node 4, cur is node 5; 
        So pre.next.next = cur is basically linking 2 with 5; 
        pre.next = reverse links node 1 with node 4.
        '''
        
        def reverseBetween(self, head, m, n):
            if m == n:
                return head
            p = dummy = ListNode(0)
            dummy.next = head
            for _ in range(m - 1):
                p = p.next
            cur = p.next
            pre = None
            for _ in range(n - m + 1):
                cur.next, pre, cur = pre, cur, cur.next
            p.next.next = cur # 2's next is 5
            p.next = pre # connect 1 to 4. 
            return dummy.next

-5) How to use nonlocals in python3 to make code easier:
    (check if palindrome exists in singly linked list)
        def isPalindrome(self, head):
            """
            :type head: ListNode
            :rtype: bool
            """
            
            if(head == None):
                return True
            
            n = head
            l = 0      
            while n:
                n = n.next
                l += 1
            
            lp = head
            rp = head        
            rpCounter = (l+1)//2
            lpCounter = (l//2 -1)
            left_counter = 0
            
            for i in range(rpCounter):
                rp = rp.next
                
            def check_palin(lp): 
                # We only need these 2 as nonlocals. 
                # because we modify in the closure. 
                # Also cant use rp as argument 
                # to function call. unless you wrap in []. Why?

                nonlocal rp 
                nonlocal left_counter

                if (left_counter < lpCounter):
                    left_counter += 1
                    result = check_palin(lp.next)
                    if result == False:
                        return False
                
                if(rp == None):
                    return True
                
                if(rp.val == lp.val):
                    rp = rp.next # check next rp. 
                    return True # needed when there are only 2 nodes in linked list. 
                else:
                    return False
            return check_palin(lp)

-4.5) find middle of linked list:
    class Solution(object):
        def middleNode(self, head):
            slow = fast = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow

-4) Python generator for converting binary to value, but 
    binary is encoded as a linked list:
    
    class Solution(object):
        def yield_content(self, head):
            current = head
            yield current.val
            while current.next != None:
                current = current.next
                yield current.val

        def getDecimalValue(self, head):
            bin_number = ''
            generator = self.yield_content(head)
            while True:
                try:
                    bin_number += str(next(generator))
                except StopIteration:
                    break
            return int(bin_number, 2)

-3) WHEN GIVEN CONSTRAINTS TO A PROBLEM
    NEGATE THE CONsTRAINTS TO EXPLOIT PROBLEM STRUCTURE. think combinatorically 
    about how to use constraints, whether that means to do there exists, or there 
    doesnt exist such that the constrain is satisfied. especially for greedy questions. 
    think in positive space and negative space.

-2) For sliding window, remember that you can do optimized sliding window 
    by skipping multiple indexes ahead instead of skipping one at a time. 
    COMPRESS THE STEPS TO FURTHER OPTIMIZE SLIDING WINDOW!
    OR USE MULTIPLE POINTERS. 

-1)     DFS, BFS + COLORS IS POWERFUL!
        Another way to check if graph is bipartionable. 
        ALGORITHM:
        CAN DO BIPARTITION WITH DFS AND 2 COLORING. 

        For each connected component, we can check whether 
        it is bipartite by 
        just trying to coloring it with two colors. How to do this is as follows: 
        color any node red, then all of it's neighbors blue, 
        then all of those neighbors 
        red, and so on. If we ever color a red node blue 
        (or a blue node red), then we've reached a conflict.

+
+
+
+
+
+

+-108) Linked list cycle detection start algo with treess:
+
+        LeetCode 1650. Lowest Common Ancestor of a Binary Tree III
+        Tree
+        Given two nodes of a binary tree p and q, return their lowest common ancestor (LCA).
+        Each node will have a reference to its parent node. The definition for Node is below:
+        class Node {
+            public int val;
+            public Node left;
+            public Node right;
+            public Node parent;
+        }
+
+        /*
+        // Definition for a Node.
+        class Node {
+        public:
+            int val;
+            Node* left;
+            Node* right;
+            Node* parent;
+        };
+        */
+            Node* lowestCommonAncestor(Node* p, Node * q) {
+                Node* a = p, *b = q;
+                while (a != b) {
+                    a = (a == nullptr ? q : a->parent);
+                    b = (b == nullptr ? p : b->parent);
+                return a;
+        };
+
+
+
+-107) PRINTING QUESTIONS -> good pattern is to enumerate all the indices you are going
+    to print to make it easier to figure out a good way to traverse the array 
+    Diagonal Matrix:
+    98. Diagonal Traverse
+
+    Given an m x n matrix mat, return an array of all the 
+    elements of the array in a diagonal order.
+        
+    class Solution:
+        def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
+            '''
+            LC Pattern:
+            One trick for these questions is to enumerate how the indexes should be visited and find 
+            patttersn in that enumeration.
+            '''
+            i = 0 
+            j = 0 
+            go_up = True
+            collect = []
+            N = len(mat)
+            M = len(mat[0])
+            while True:
+                print("i, j, go_up", i, j, go_up)
+                collect.append(mat[i][j])
+                if(i == N -1 and j == M-1):
+                    return collect
+                if go_up:
+                    if(i-1 < 0 or j + 1 >= M):
+                        if( j + 1 >= M):
+                            i += 1
+                        else:
+                            j += 1
+                        go_up = False
+                    else:
+                        i -= 1
+                        j += 1
+                else:
+                    if(j - 1 < 0 or i + 1 >= N ):
+                        if(i + 1  >= N ):
+                            j += 1
+                        else:    
+                            i += 1
+                        go_up = True
+                    else:
+                        i += 1
+                        j -=  1
+-106) Research this math problem (Hidden bit manipulation): 
+    Convert 0 into N in minimum steps by multiplying with 2 or by adding 1.
+    Input: 19;  Output: 6
+    Medium level problem
+    Explained: Recursion -> DP -> Better{ O(n) } -> Optimal{ O(log(n)} }
+    After that question was updated with if you are only allowed to multiply by 2, K times.
+    Explained: Optimal{ O(logK) }
+    This is a bit manipulation question (possibly???). 
+    N -> we need to check how many set bits are there. 
+    For instnace N = 6  -> 0110
+    Add 1          –> 0 + 1 = 1.
+    Multiply 2  –> 1 * 2 = 2.
+    Add 1          –> 2 + 1 = 3. 
+    Multiply 2  –> 3 * 2 = 6.
+    Therefore number of operations = 4.  
+    divide by 2, subtract 1, divide by 2 then subtract 1?
+    0 + 1 = 1 * 2 = 2 + 1 = 3 * 2 = 6
+    # QUESTION DOES THE BELOW MODIFIED GEEKS FOR GEEKS SOLN ALWAYS 
+    # WORK OR DO YOU ALWAYS NEED TO DO SOME TYPE OF DP??
+    def minimumOperation(N):
+    
+        # Stores the count of set bits
+        count = 0
+    
+        while (N):
+    
+            # If N is odd, then it
+            # a set bit
+            if (N & 1 == 1):
+                count += 1
+    
+            N = N >> 1
+            count += 1
+        # Return the result
+        return count
+-105) Read leftmost column with at least a one:
+      Abuse the fact that pointers can move both row and column in a 2d array 
+      and try to continue to be greedy as you optimize for the soln. 
+    
+      Think about wierd traversals in a  matrix as well!!
+-104) SOMETIMES YOU CAN ANSWER GOING BOTH LEFT TO RIGHT, OR RIGHT TO LEFT
+        You are given an integer num. You can swap two digits at 
+        most once to get the maximum valued number.
+        Return the maximum valued number you can get.
+       
        LEETCODE SOLN:

        class Solution(object):
            def maximumSwap(self, num):
                A = map(int, str(num))
                last = {x: i for i, x in enumerate(A)}
                for i, x in enumerate(A):
                    for d in xrange(9, x, -1):
                        if last.get(d, None) > i:
                            A[i], A[last[d]] = A[last[d]], A[i]
                            return int("".join(map(str, A)))
                return num

        SOLN 2;
        class Solution:
            def maximumSwap(self, num):
                """
                :type num: int
                :rtype: int
                """
                num = [int(x) for x in str(num)]
                max_idx = len(num) - 1
                xi = yi = 0
                for i in range(len(num) - 1, -1, -1):
                    if num[i] > num[max_idx]:
                        max_idx = i
                    elif num[i] < num[max_idx]:
                        xi = i
                        yi = max_idx
                num[xi], num[yi] = num[yi], num[xi]
                return int(''.join([str(x) for x in num]))

+        
+        class Solution:
+            def maximumSwap(self, num: int) -> int:
+                return self.maximumSwapLtoR(num)
+                # this soln also works:
+                #return self.maximumSwapRtoL(num)
+            # SOLN THAT GOES RIGHT TO LEFT!
+            def maximumSwapRtoL(self, num: int) -> int:
+                '''
+                2736
+                ^
+                Go right to left soln
+                '''
+                
+                nums_arr = [int(i) for i in str(num)]
+                
+                j = len(nums_arr) - 1
+                biggest = -1
+                biggest_idx = -1
+                
+                left_idx = -1
+                
+                viable_soln = None
+                while j > -1:
+                    if nums_arr[j] > biggest:
+                        # SAVE THE PREVIOUS VIABLE SOLN!
+                        if(biggest_idx != -1 and left_idx != -1):
+                            viable_soln = (biggest_idx, left_idx)  
+                            
+                        # keep track of previous viable soln, in case we cant find a better one?
+                        biggest= nums_arr[j]  
+                        biggest_idx = j
+                        left_idx = -1
+                    elif nums_arr[j] < biggest:
+                        left_idx = j
+                    j -= 1
+                
+                def create_soln(i, j):
+                    nums_arr[i], nums_arr[j] = nums_arr[j], nums_arr[i]
+                    return int("".join([str(i) for i in nums_arr]))
+                
+                ans = num
+                if left_idx == -1:
+                    if viable_soln != None:
+                        return create_soln(viable_soln[0], viable_soln[1])
+                else:
+                    return create_soln(biggest_idx, left_idx)
+                        
+                return ans
+            # SOLN THAT GOES LEFT TO RIGHT
+            def maximumSwapLtoR(self, num: int) -> int:
+                '''
+                Left to right
+                just make sure its descending,
+                when its not descending fix max valu to the right,
+                then swap it with something in the left. 
+                '''
+                nums_arr = [int(i) for i in str(num)]
+                prev = float("inf")
+                break_idx = None
+                
+                for idx, i in enumerate(nums_arr):
+                    if i <= prev:
+                        prev = i
+                    else:
+                        break_idx = idx
+                        break
+                
+                if break_idx == None:
+                    return num
+                biggest = nums_arr[break_idx]
+                biggest_idx = break_idx
+                
+                for i in range(break_idx+1, len(nums_arr)):
+                    if nums_arr[i] >= biggest:
+                        biggest = nums_arr[i]
+                        biggest_idx = i
+                
+                def create_soln(i, j):
+                    nums_arr[i], nums_arr[j] = nums_arr[j], nums_arr[i]
+                    return int("".join([str(i) for i in nums_arr]))
+                
+                # ok now we need a left idx...
+                for i in range(len(nums_arr)):
+                    if biggest > nums_arr[i]:
+                        # then swap it and return 
+                        return create_soln(i, biggest_idx)
+                  
+

+---------------------------------------------------------------------------------------------
+------------------------------------------------------------------------------------------------------------------------
+---------------------------------------------------------------------------------------------
+------------------------------------------------------------------------------------------------------------------------
+Start reading notes from here, when you are done going forwards, go backwards from here. 
+ALGO README PART 1


1)  For problems like parenthesis matching. You can use a stack to solve the matching. But you can also
    do matching by incrementing and decrementing an integer variable. Or you can use colors or 
    other types of INDICATOR VARIABLE TYPE SOLUTIONS that contain meta information on the problem. 
    Also remember that as you see each element, you can push multiple times to stack, not just once
    in case u need to keep count of something before a pop occurs. 
    
0.05) To solve a difficult 3D problem or 2D problem. Solve the lower dimension first, 
     and then use it to guide your solution for higher dimensions. 
     Such as max area of rectangle of 1's.
     Also think of other leetcode problems you did and just copy paste the ideas lol 
    Maximal Rectangle can use strategies from largest histogram leetcode problem. 
    

0.1) When doing string splitting, there are helper functions 
     but sometimes its better to for loop through the 
     string character by character because you have more granularity 
     which can help to solve the problem easily. 

0.15)   One-line Tree in Python
        Using Python's built-in defaultdict we can easily define a tree data structure:

        def tree(): return defaultdict(tree)
        
        users = tree()
        users['harold']['username'] = 'hrldcpr'
        users['handler']['username'] = 'matthandlersux'

0.2) Learnings from interval coloring. Sometimes we care a lot about tracking our current running value
    such as current running intersection to generate a solution output. However it can be better to
    focus on the negative space, and care about what will be evicted first, as our main tracking 
    concern using a priority queue. Look at problems with both positive and negative space in mind. 
    Know whats best to track, and track it. dont be fooled by the question. 

0.25) Look to see if the question is a
      in the negative space of questions you've 
      seen before!
      For instance, Non-overlapping intervals LC, where
      you find the minimum number of intervals to remove 
      to make the rest of the intervals non-overlapping
      is the negative of max interval scheduling using
      earliest finish time.

      But you can also solve directly:
      Sort the intervals by their start time. 
      If two intervals overlap, the interval 
      with larger end time will be removed 
      so as to have as little impact on 
      subsequent intervals

0.26) Introduce multiple running variables, if maintaining one 
        running variable makes the updates too difficult or tricky
        (SEPERATION OF CONCERNS USING MULTIPLE VARS OR CONTAINERS TRICK)

        For instance: LC -> 
        Given an integer array nums, find the contiguous 
        subarray within an array (containing at least one number) 
        which has the largest product.

        Example 1:

        Input: [2,3,-2,4]
        Output: 6
        Explanation: [2,3] has the largest product 6.
        
        Solution:
        int maxProduct(int A[], int n) {
            // store the result that is the max we have found so far
            int r = A[0];

            // imax/imin stores the max/min product of
            // subarray that ends with the current number A[i]
            for (int i = 1, imax = r, imin = r; i < n; i++) {
                // multiplied by a negative makes big number smaller, small number bigger
                // so we redefine the extremums by swapping them
                if (A[i] < 0)
                    swap(imax, imin);

                // max/min product for the current number is either the current number itself
                // or the max/min by the previous number times the current one
                imax = max(A[i], imax * A[i]);
                imin = min(A[i], imin * A[i]);

                // the newly computed max value is a candidate for our global result
                r = max(r, imax);
            }
            return r;
        }


0.27) Binary Tree Max Path Sum: the binary tree version of max subarray sum:

    Given a non-empty binary tree, find the maximum path sum.
    For this problem, a path is defined as any sequence of nodes from 
    some starting node to any node in the tree along the parent-child connections. 
    The path must contain at least one node and does not need to go through the root.
    
    def maxPathSum(self, root: TreeNode) -> int:
        m = float(-inf)
        def helper(node):
            nonlocal m
            if node is None:
                return 0
            maxRightPath =  helper(node.right)
            maxLeftPath = helper(node.left)           
            right = maxRightPath + node.val
            left = maxLeftPath + node.val
            connected = maxRightPath + maxLeftPath + node.val
            m = max(m, right, left, connected, node.val)
            maxPath = max(right, left, node.val, 0)
            return maxPath
        helper(root)
        return m


0.28) Given an unbalacned binary search tree, write a function 
      kthSmallest to find the kth smallest element in it.
    # GENERATOR SOLN:
    def traverse(node):
        if node:
            yield from traverse(node.left)
            yield node
            yield from traverse(node.right)
        
    def kthSmallest(root, k):
        k -= 1
        for i, node in enumerate(traverse(root)):
            if i == k:
                return node.val


    # RECURSIVE
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # LETS DO INORDER THIS TIME. 
        found = None
        count = 0

        def inorder(node):
            nonlocal found
            nonlocal count
            
            # Ok found left side. 
            if node is None:
                return 
            
            inorder(node.left)
            count += 1
            if count == k:
                found = node.val
                return 
            inorder(node.right)
        inorder(root)
        return found
    
    # ITERATION:
    def kthSmallest(self, root, k):
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
    
    What if the BST is modified (insert/delete operations) 
    often and you need to find the kth smallest frequently?
    How would you optimize the kthSmallest routine?
    
    Seems like a database description, isn't it? Let's use here 
    the same logic as for LRU cache design, and combine an 
    indexing structure (we could keep BST here) with a double linked list.
    Such a structure would provide:

    O(H) time for the insert and delete.
    O(K) for the search of kth smallest.

    You could also add a count field to each node but this would make performance:
    It's not O(log n). O(h) instead. h is the height of the BST.


0.3) Granularity === Optimization. Break up variables and track everything. Structurd things like
      for loops, and functional structures like reduction, and map, and filter 
      that dont fit the required granlarity should be thrown away if it interferes. 

0.4) HOW TO do cieling using python floor int truncation:

    def ceiling_division(n, d):
        return -(n // -d)

0.45) When you are solving a problem, and it seems like DP. 
      take a step back and see if you can use  hash map abuse + greedy,
      to solve the problem. DP uses hash maps too, but maybe you can
      be smart and add direction, and running values, which will allow  
      hash map abuse to work (try thinking of it bottom up, base case, direction,
      to help you think of hash map abuse or a greedy soln). 

      also TRY binary SEARCH, BINARY SEARCH CAN fit into anywhere!!



0.5) Preprocess and do a running comparison between 2 containers. 
    For instance, to do certain problems you need to sort a list and then compare the sorted list 
    to another list to determine ways to do sliding window/2pointer type techniques. 

0.55) GREEDY AND 2 POINTER SOLUTION GUIDE: 
      
      For an optimization problem, to do it greedily and efficiently, do not enumerate 
      all states, only ones that you are sure could possibily be better under the problems
      constraints. 
      Do this first:
      
      DISCOVER ALL THE CONSTRIANTS INTRODUCED BY THE PROBLEM FIRST!
      THEN THINK OF THEOREMS THAT MUST BE TRUE AS A RESULT OF THE CONSTRAINTS.
      RUN THROUGH EXAMPLES, TO ENSURE THEOREMS ARE TRUE, and then step through a 
      solution to see how they work: 

      Do both 1 and 2 at same time to come up with a solution:
      1) EXPLOT THEOREMS TO CREATE the OPTIMIZATION PATTERNS AND STEPS TO EXECUTE.
         Thinking of the problem using DP can help with greedy creation. 
      2) THINK OF WHAT A PROOF TO THE GREEDY PROBLEM COULD BE GIVEN THE THEREMS;
         use proof to guide creation of greedy.

      Example: Container with most water: 
        
        Find two lines, which together with x-axis forms a container, 
        such that the container contains the most water.
        Input: [1,8,6,2,5,4,8,3,7]
        Output: 49
        
        class Solution:
            def maxArea(self, height):
                i, j = 0, len(height) - 1
                water = 0
                while i < j:
                    water = max(water, (j - i) * min(height[i], height[j]))
                    if height[i] < height[j]:
                        i += 1
                    else:
                        j -= 1
                return water

0.56) SLIDING WINDOW ALGO DESIGN PATTERN:
      Max Sum Contiguous Subarray: 
      
    # Function to find the maximum contiguous subarray 
    from sys import maxint 
    def maxSubArraySum(a,size): 
        
        max_so_far = -maxint - 1
        max_ending_here = 0
        
        for i in range(0, size): 
            max_ending_here = max_ending_here + a[i] 
            if (max_so_far < max_ending_here): 
                max_so_far = max_ending_here 
    
            if max_ending_here < 0: 
                max_ending_here = 0   
        return max_so_far 


0.57) Interval Coloring:
    
    from heapq import *
    import itertools

    def minMeetingRooms(self, intervals):
        sorted_i = sorted(intervals, key=lambda x: x.start)
        
        pq = []
        counter = itertools.count()
        active_colors = 0
        max_colors = 0
        
        for i in sorted_i:
            iStart = i.start
            iEnd = i.end
            
            while len(pq) != 0:
                
                min_end_time, _, interval_to_be_popped = pq[0]                
                if(iStart <= min_end_time):
                    break                
                active_colors -= 1
                _ = heappop(pq)
                            
            c = next(counter)
            item = [iEnd, c, i]
            heappush(pq, item)
            print("increment active colors")
            active_colors += 1
            max_colors = max(active_colors, max_colors)
        return max_colors

+0.58) Min Meeting rooms ii Approach 2) : Chronological Ordering
+    class Solution:
+        """
+        @param intervals: an array of meeting time intervals
+        @return: the minimum number of conference rooms required
+        """
+        def minMeetingRooms(self, intervals):
+            # Write your code here
+            '''
+            start < end
+            get all start times, get all end times, sort them together. 
+            If you see starts increment, if you see an end, decrement. 
+            Keep track of max overlapping rooms
+            '''
+            all_times = []
+            for i in intervals:
+                all_times.append( ("s", i.start) )
+                all_times.append( ("e", i.end))
+            sorted_times = sorted(all_times, key=lambda x: x[1])
+            
+            cur = 0
+            ans = 0
+            for i in sorted_times:
+                if(i[0] == "s"):
+                    cur += 1
+                else:
+                    cur -= 1
+                ans = max(ans, cur)
+            return ans


0.6) To delete from a list in O(1), any index, you can swap the middle indexed element with
    the last element in the array. then call array.pop(). This is O(1). You could also use a linked
    list. The problem is, this will mess up the sorting of your array if you do this. 
    so dont do it if your result needs to be sorted. 


0.65) EMULATE DO WHILE IN PYTHON:

        i = 1

        while True:
            print(i)
            i = i + 1
            if(i > 3):
                break

0.67) PYTHON AND BINARY Enumerate all subsets:
    class Solution:
        def subsets(self, nums):
            numberOfSubsets = len(nums)
            subsets = []
            for i in range(0, 2 ** numberOfSubsets):
                bits = bin(i)
                subset = []
                for j in range(0, numberOfSubsets):
                    if i >> j & 1:  # Check if the first bit is on, 
                                    # then check if second bit is on, 
                                    # then check third bit is on, and keep going
                        subset.append(nums[j])

                subsets.append(subset)
            return subsets

        Iterate through all subsets of a 
        subset y (not including empty set) (TODO WRITE IN PYTHON):

        given a set of numbers, we want to find the sum of all subsets.

            Sol: This is easy to code using bitmasks. we can use an array to store all the results.

            int sum_of_all_subset ( vector< int > s ){
                        int n = s.size() ;
                        int results[ ( 1 << n ) ] ;     // ( 1 << n )= 2^n

                    //initialize results to 0
                        memset( results, 0, sizeof( results ) ) ;

                    // iterate through all subsets
                    // for each subset, O(2^n)
                    for( int i = 0 ; i < ( 1 << n ) ; ++ i ) {    
                            // check membership, O(n)
                            for ( int j = 0; j < n ; ++ j ) {       
                                // test bit
                                if ( ( i & ( 1 << j ) ) ! = 0 )    
                                    results[i] += s [j] ;          
                                }
                }
            }

0.68) When you are doing a question that requires modifying a list as you go
      dont save pointers to the list and reprocess list and other stuff. 
      Do all the modifications in one loop as you go.
      Such as for Insert Interval (LC 57)
      
0.69) When the question has products involved. Use Logs to turn it into a sums question. 


0.7) Iterate backwards through array using python for dp:
    for i in range(len(arr) - 1, -1, -1):
        print(i)

0.71) Remember that its constant space when you are enumerating over the alphabet!!

0.75) To get the fastest greedy solution possible, you must keep getting more
     and more greedy and breaking assumptions you think you have. Only 
     care about the answer, not the details of the question. Focus on 
     what you REALLY NEED!!! when you keep breaking the questions rules
     you thought were rules, find the true constraints!
     Look at 621) Task Scheduler. It looked complicated but we just 
     kept getting more greedy to get the most optimal MATHEMATICAL SOLUTION.
     USE MATH AND ANALYSIS TO GET THE BEST SOLUTION! + use binary SEARCHHCHCHC


0.8) Sometimes you will get TLE with the bottom up solution. 
     
     This is because bottom up is slower since it is performing a BFS, 
     rather than going directly to the solution unlike DFS + memoization, 
     that tries to solve as quickly as possible. 

     => If you only need a subset of the possible outputs 
     from the algorithm, then the answer could also be yes. 
     You only calculate the outputs you need, and so you avoid unneeded work.
    => Jump 


0.85) When their is a max or min problem. 
      TRY GREEDY first before doing GRAPH SEARCH + DP
      Be smart first and do 
      GREEDY + EXPLOIT PROBLEM STRUCTURE before anything 
      else. 

0.9) Bidirectional BFS Reasons to use:

    Bi-directional BFS will yield much better 
    results than simple BFS in most cases. Assume the 
    distance between source and target is k, and the 
    branching factor is B (every vertex has on average B edges).

    BFS will traverse 1 + B + B^2 + ... + B^k vertices.
    Bi-directional BFS will traverse 2 + 2B^2 + ... + 2B^(k/2) vertices.
    For large B and k, the second is obviously much faster the the first.

1) Exploit problem structure

    a) This means using a sliding window
    b) This means keeping track of running values, and updating permanent values such as 
        keeping tracking of curr_max_from_left to update overall_max when you are running through an array
        -> running variables, running maps, running sets
    c) Use 2 pointer solutions. Two pointers can be nodes or indexes in an array.

1.25)   When the problem says sorted order, you can use
        binary search or a very smart version of
        2 pointer/2 index solutions. For instance,
        2 SUM for ordered arrays can be 
        solved in O(N) (OR even O(log N) if you implement 
        binary searching 2 pointers -> ACTUALLY no i think its still o(N))

        # binary search        
        def twoSum(self, numbers, target):
            for i in xrange(len(numbers)):
                l, r = i+1, len(numbers)-1
                tmp = target - numbers[i]
                while l <= r:
                    mid = l + (r-l)//2
                    if numbers[mid] == tmp:
                        return [i+1, mid+1]
                    elif numbers[mid] < tmp:
                        l = mid+1
                    else:
                        r = mid-1

1.3) Count set bits in integer:
      (Log N!!) if N represents size of number

    # Function to get no of set bits in binary 
    # representation of positive integer n */ 
    def  countSetBits(n): 
        count = 0
        while (n): 
            count += n & 1
            n >>= 1
        return count 

1.5) LOOK AT PROBLEM IN ALL POSSIBLE DIRECTIONS to apply your techniques, whether its 2 pointer, 
    sliding window, or Dynamic programming
    a) think about left to right
    b) right to left
    c) 2 pointer on either side and you close into the middle
    d) 2 pointers, one that traverses even indexes, and the other that traverses odd indexes
    e) Linked list pointers, second moves twice as fast as the first. When second gets to end, first is at halfway node. 
    f) Be creative in how you see the DIRECTIONALITY of the solution for a given problem. 


+1.56) Linked List Palindrome Question:
+    Solution 1:
+        def isPalindrome(self, head):
+            rev = None
+            slow = fast = head
+            while fast and fast.next:
+                fast = fast.next.next
+                rev, rev.next, slow = slow, rev, slow.next
+            if fast:
+                slow = slow.next
+            while rev and rev.val == slow.val:
+                slow = slow.next
+                rev = rev.next
+            return not rev
+        
+        Expand rev, rev.next, slow = slow, rev, slow.next in C++ for easier understanding.
+
+        ListNode* tmp = rev;
+        rev = slow;
+        slow = slow -> next;
+        rev -> next = tmp;
+
+    Solution 2: Play Nice
+
+    Same as the above, but while comparing the two halves, restore the 
+    list to its original state by reversing the first half back. 
+
+    def isPalindrome(self, head):
+        rev = None
+        fast = head
+        while fast and fast.next:
+            fast = fast.next.next
+            rev, rev.next, head = head, rev, head.next
+        tail = head.next if fast else head
+        isPali = True
+        while rev:
+            isPali = isPali and rev.val == tail.val
+            head, head.next, rev = rev, head, rev.next
+            tail = tail.next
+        return isPali
+
+
+    Solution 3:
+    How to use nonlocals in python3 to make code easier:
+    (check if palindrome exists in singly linked list)
+        def isPalindrome(self, head):
+            """
+            :type head: ListNode
+            :rtype: bool
+            """
+            
+            if(head == None):
+                return True
+            
+            n = head
+            l = 0      
+            while n:
+                n = n.next
+                l += 1
+            
+            lp = head
+            rp = head        
+            rpCounter = (l+1)//2
+            lpCounter = (l//2 -1)
+            left_counter = 0
+            
+            for i in range(rpCounter):
+                rp = rp.next
+                
+            def check_palin(lp): 
+                # We only need these 2 as nonlocals. 
+                # because we modify in the closure. 
+                # Also cant use rp as argument 
+                # to function call. unless you wrap in []. Why?
+                nonlocal rp 
+                nonlocal left_counter
+                if (left_counter < lpCounter):
+                    left_counter += 1
+                    result = check_palin(lp.next)
+                    if result == False:
+                        return False
+                
+                if(rp == None):
+                    return True
+                
+                if(rp.val == lp.val):
+                    rp = rp.next # check next rp. 
+                    return True # needed when there are only 2 nodes in linked list. 
+                else:
+                    return False
+            return check_palin(lp)
+1.57) Python generator for converting binary to value, but 
+    binary is encoded as a linked list:
+    
+    class Solution(object):
+        def yield_content(self, head):
+            current = head
+            yield current.val
+            while current.next != None:
+                current = current.next
+                yield current.val
+        def getDecimalValue(self, head):
+            bin_number = ''
+            generator = self.yield_content(head)
+            while True:
+                try:
+                    bin_number += str(next(generator))
+                except StopIteration:
+                    break
+            return int(bin_number, 2)



+1.58) WHEN GIVEN CONSTRAINTS TO A PROBLEM
+    NEGATE THE CONsTRAINTS TO EXPLOIT PROBLEM STRUCTURE. think combinatorically 
+    about how to use constraints, whether that means to do there exists, or there 
+    doesnt exist such that the constrain is satisfied. especially for greedy questions. 
+    think in positive space and negative space.
+1.59) For sliding window, remember that you can do optimized sliding window 
+    by skipping multiple indexes ahead instead of skipping one at a time. 
+    COMPRESS THE STEPS TO FURTHER OPTIMIZE SLIDING WINDOW!
+    OR USE MULTIPLE POINTERS. 
+1.6)     DFS, BFS + COLORS IS POWERFUL!
+        Another way to check if graph is bipartionable. 
+        ALGORITHM:
+        CAN DO BIPARTITION WITH DFS AND 2 COLORING. 
+        For each connected component, we can check whether 
+        it is bipartite by 
+        just trying to coloring it with two colors. How to do this is as follows: 
+        color any node red, then all of it's neighbors blue, 
+        then all of those neighbors 
+        red, and so on. If we ever color a red node blue 
+        (or a blue node red), then we've reached a conflict.

1.6) Coin Change Bottom Up DP:
        You are given coins of different denominations and a total amount of money amount. 
        Write a function to compute the fewest number of coins that you need to 
        make up that amount. If that amount of money cannot be made 
        up by any combination of the coins, return -1.

        Example 1:

        Input: coins = [1, 2, 5], amount = 11
        Output: 3 
        Explanation: 11 = 5 + 5 + 1
        Example 2:

        Input: coins = [2], amount = 3
        Output: -1
        Note:
        You may assume that you have an infinite number of each kind of coin.  
        
        class Solution(object):
            def coinChange(self, coins, amount):
                """
                :type coins: List[int]
                :type amount: int
                :rtype: int
                """
                rs = [amount+1] * (amount+1)
                rs[0] = 0
                for i in xrange(1, amount+1):
                    for c in coins:
                        if i >= c:
                            rs[i] = min(rs[i], rs[i-c] + 1)

                if rs[amount] == amount+1:
                    return -1
                return rs[amount]


1.7) Sliding window: Common problems you use the sliding window pattern with:
        -> Maximum sum subarray of size ‘K’ (easy)
        -> Longest substring with ‘K’ distinct characters (medium)
        -> String anagrams (hard)

1.75) Transpose matrix:
      Switch (i, j) with (j,i) either by 
      iterating over upper triangle or lower triangle:

        n = len(A)
        for i in range(n):
            for j in range(i):
                A[i][j], A[j][i] = A[j][i], A[i][j]


1.8) Two Pointers is often useful when searching pairs in a 
        sorted array or linked list; for example, 
        when you have to compare each element 

        Here are some problems that feature the Two Pointer pattern:
        of an array to its other elements.
        Squaring a sorted array (easy)
        Triplets that sum to zero (medium)
        Comparing strings that contain backspaces (medium)

1.9) Fast and slow pointer:
        The Fast and Slow pointer approach, also known as the Hare & Tortoise algorithm, 
        is a pointer algorithm that uses two pointers which move through the array 
        (or sequence/linked list) at different speeds. This approach is quite useful 
        when dealing with cyclic linked lists or arrays.
        By moving at different speeds (say, in a cyclic linked list), the 
        algorithm proves that the two pointers are bound to meet. The fast 
        pointer should catch the slow pointer once both the pointers are in a cyclic loop.
        
        Linked List Cycle (easy)
        Linked List Cycle || (medium)
        Palindrome Linked List (medium)
        Cycle in a Circular Array (hard)


1.95) Use pointer on the fly construction !!
      Combining running 2 pointers, running 2 container concepts, and space-efficient
      dynamic programming concepts to get O(N) speed, O(1) space except for output container.
      Think in terms of containers to implement fast 2 pointer solution!
      Then think in terms of DP to reduce to a 1 pointer solution!
      

        FOR THIS PROBLEM, TRY LAGGING YOUR RUNNING VARIABLES AND INSERTIONS TO MAKE THE ALGO WORK.
        
        238. Product of Array Except Self

        Given an array nums of n integers where n > 1,  
        return an array output such that output[i] is 
        equal to the product of all the elements of nums except nums[i].

        Example:

        Input:  [1,2,3,4]
        Output: [24,12,8,6]
        Note: Please solve it without division and in O(n).

        Follow up:
        Could you solve it with constant space complexity? 

        class Solution:
            # @param {integer[]} nums
            # @return {integer[]}
            def productExceptSelf(self, nums):
                p = 1
                n = len(nums)
                output = []
                for i in range(0,n):
                    output.append(p)
                    p = p * nums[i]
                p = 1
                for i in range(n-1,-1,-1):
                    output[i] = output[i] * p
                    p = p * nums[i]
                return output

1.96)
    https://leetcode.com/problems/subarray-sum-equals-k/description/ 
    Variation to just give true or false Did both solutions using Prefix sum and sliding window.

    Nested List weighted sum
    Coding1:

    K closest points to Origin solved in 20 mins using quick select.
    Variation of Robot room cleaner - This is where I struggled. 
    I only had 15 mins left to answer this, Interviewer was not letting me 
    finish my code unless I give them a clear explanation. I could not finish the code. 
    Gave the answer verbally that I will use DFS and backtracking to go back when the robot is stuck.

    Coding 2:

    https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/
    https://leetcode.com/problems/powx-n/description/
    Gave optimal answers for all of them and explained my thought process thoroughly. 
    I think I made it up for my earlier coding round here.

1.97)  GREEDY SEARCHING INSTEAD OF BINARY SEARCH! 
       ABUSING INCREASING VALUES IN ONE DIRECT AND DECREASING IN OTHER DIRECTION. 
       AKA ABUSING CRITICAL POINTS.  
        
        -> TO COME UP WITH THESE SOLUTIONS YOU MUST TRACE OUR A BUNCH OF EXAMPLES 
        AND SEE IF THE ALGO WORKS IN YOUR HEAD BEFORE YOU WRITE IT OUT!!
        -> USE ALL YOUR TECHNIQUIES AND THINKING AS YOU TRACE..
        -> HERE CRITICIAL POINTS WERE BOTTOM LEFT AND TOP RIGHT..

        Write an efficient algorithm that searches for a value target in an m x n 
        integer matrix matrix. This matrix has the following properties:

        Integers in each row are sorted in ascending from left to right.
        Integers in each column are sorted in ascending from top to bottom.

        1   4  7 11 15
        2   5  8 12 19
        3   6  9 16 22
        10 13 14 17 24
        18 21 23 26 30

        ^ search for 5 in here. 


        This isnt binary search. 


    Actually its abusing the sort in both the rows and cols to achieve O(N+M) complexity
    instead of binary seraching each row to do O(n*log(m)) complexity which is worse. 
    Both the top right and bottm left cols will have 


    We start search the matrix from top right corner, initialize the current 
    position to top right corner, if the target is greater than the value in current 
    position, then the target can not be in entire row of current position because 
    the row is sorted, if the target is less than the value in current position, 
    then the target can not in the entire column because the column is sorted too. 
    We can rule out one row or one column each time, so the time complexity is O(m+n).

    public class Solution {
        public boolean searchMatrix(int[][] matrix, int target) {
            if(matrix == null || matrix.length < 1 || matrix[0].length <1) {
                return false;
            }
            int col = matrix[0].length-1;
            int row = 0;
            while(col >= 0 && row <= matrix.length-1) {
                if(target == matrix[row][col]) {
                    return true;
                } else if(target < matrix[row][col]) {
                    col--;
                } else if(target > matrix[row][col]) {
                    row++;
                }
            }
            return false;
        }
    }



2) Back tracking
    => For permutations, need some intense recursion 
        (recurse on all kids, get all their arrays, and append our chosen element to everyones array, return) 
        and trying all posibilities
    => For combinations, use a binary tree. Make the following choice either: CHOOSE ELEMENT. DONT CHOOSE ELEMENT. 
        Recurse on both cases to get all subsets
    => To get all subsets, count from 0 to 2^n, and use bits to choose elements.
    => When doing DP, check to see if you are dealing with permutations or combinations type solutions, and 
        adjust your DP CAREFULLY ACCORDING TO THAT -> AKA CHECK COIN CHANGE 2

2.1) How do you do LCA? explain now. 

    class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
        return root

        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        if l and r:
            return root
        return l or r


2.3) Graphs =>
    Try BFS/DFS/A star search
    Use dist/parent/visited maps to get values
    -> Cycle detection -> use visited map. 
        Actually need to carry parent in param as well in dfs 
        for undirected graph
    -> shortest path = BFS
    -> Do parent stuff with parents map (such as common ancestors).
    -> Cool graph techniques is coloring nodes, or flagging nodes 
        if you are trying to get multiple paths in graph and looking for path intersections. 
    -> To do topological sort, USE DFS and get start end times. also can count in-nodes and out-nodes to sort them !
    -> Reverse the graph to get the shortest path starting from all other nodes to your node!
    -> Sometimes in a problem, you cant bfs/dfs once. you need to bfs/dfs every vertex!
    -> Minimum spanning tree -> use prims algorithm or kruskals algorithm
    -> Find strongly connected components => use kosarju's algo which does dfs on graph and the reverse of the graph from a vertex.
    -> Topological SORT: dfs, process nodes children first, then add node to list. then reverse entire list at end
     REMOVING CYCLES, DFS, AND BFS using colors: DONE IN GRAPHA ALGO REVIEW SECTION BELOW. 


2.31) PRIM VS KRUSKAL
    If you implement both Kruskal and Prim, in their optimal form : with a union find and a 
    finbonacci heap respectively, then you will note how Kruskal is easy to implement compared to Prim.

    Prim is harder with a fibonacci heap mainly because you have to maintain a book-keeping 
    table to record the bi-directional link between graph nodes and heap nodes. With a Union Find, 
    it's the opposite, the structure is simple and can even produce directly the mst at almost no additional cost.



    Use Prim's algorithm when you have a graph with lots of edges.

    For a graph with V vertices E edges, Kruskal's algorithm runs in O(E log V) time 
    and Prim's algorithm can run in O(E + V log V) amortized time, if you use a Fibonacci Heap.

    Prim's algorithm is significantly faster in the limit when you've got a 
    really dense graph with many more edges than vertices. Kruskal performs better in 
    typical situations (sparse graphs) because it uses simpler data structures.


    Kruskal's algorithm will grow a solution from the cheapest edge by 
    adding the next cheapest edge, provided that it doesn't create a cycle.

    Prim's algorithm will grow a solution from a random vertex by adding 
    the next cheapest vertex, the vertex that is not currently in the 
    solution but connected to it by the cheapest edge.    


2.3101) Meta onsite know these thigns. 
        Phone
        (1) https://leetcode.com/problems/max-area-of-island/editorial/
        (2) https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/ -> REQUIRES bfs + tracking left and right with -1 and +1
        Virtual Onsite
        Coding 1
        https://leetcode.com/problems/powx-n/description/
        
        https://leetcode.com/problems/kth-largest-element-in-an-array/description/ -> quick select or heap OR COUNTING SORT?
        Time complexity:
        QUICK SELECT:
        O(N) on average
        O(N^2) in the worst case, always choose the smallest or biggest, function runs in O(N) and need to call N times -> N * O(N) = O(N^2)
        COUNTING SORT:
        O(N + M), N is the len of nums, and M is maxVal - minVal
        class Solution:
            def findKthLargest(self, nums: List[int], k: int) -> int:
                """
                if heap len > k and heap[0] < num, heapop and then heappush
                """
                
                max_n = max(nums)
                min_n = min(nums)
                counts = [0] * (max_n - min_n  + 1)
                for num in nums:
                    counts[num - min_n] += 1
                delta = k
                for idx in range(len(counts) - 1, -1, -1):
                    delta -= counts[idx]
                    if delta <= 0:
                        return idx + min_n
                raise "SHEEET"




        Coding 2
        https://leetcode.com/problems/binary-search-tree-iterator/description/
        Slight Variation of https://leetcode.com/problems/shortest-path-in-binary-matrix/description/ to return the path, the lc question is to return distance.

        System design
        Design a system to search statuses

        Behavioral
        usual fluff about project, manager and peers.

        I see that the editorial for some of the questions like:

        Find kth largest number
        Find k closest points to origin
        Suggest using quick select algorithm which has O(n^2) time complexity in the worst case. 
        Do they still prefer that over a heap solution? What's the acceptance criteria 
        for such questions for people who interviewed at META?


2.311) KRUSKALS WITH AND WITHOUT Disjoint set union
+        Creating Minimum Spanning Tree Using Kruskal Algorithm
+        You will first look into the steps involved in Kruskal’s Algorithm to generate a minimum spanning tree:
+        Step 1: Sort all edges in increasing order of their edge weights.
+        Step 2: Pick the smallest edge.
+        Step 3: Check if the new edge creates a cycle or loop in a spanning tree.
+        Step 4: If it doesn’t form the cycle, then include that edge in MST. Otherwise, discard it.
+        Step 5: Repeat from step 2 until it includes |V| - 1 edges in MST.
+    Kruskal's algorithm initially places all the nodes of the original graph isolated from each other, 
+    to form a forest of single node trees, and then gradually merges these trees, combining at each 
+    iteration any two of all the trees with some edge of the original graph. Before the execution of 
+    the algorithm, all edges are sorted by weight (in non-decreasing order). 
+    Then begins the process of unification: pick all edges from the first to 
+    the last (in sorted order), and if the ends of the currently picked edge 
+    belong to different subtrees, these subtrees are combined, 
+    and the edge is added to the answer. After iterating through 
+    all the edges, all the vertices will belong to the same sub-tree, 
+    and we will get the answer.
+    The simplest implementation
+    The following code directly implements the algorithm described above, 
+    and is having O(MlogM+N^2) time complexity. Sorting edges requires O(MlogN) 
+    (which is the same as O(MlogM)) operations. Information regarding the subtree to 
+    which a vertex belongs is maintained with the help of an array tree_id[] - 
+    for each vertex v, tree_id[v] stores the number of the tree , to which v belongs. 
+    For each edge, whether it belongs to the ends of different trees, 
+    can be determined in O(1). Finally, the union of the two trees is carried 
+    out in O(N) by a simple pass through tree_id[] array. Given that the 
+    total number of merge operations is N−1, we obtain 
+    the asymptotic behavior of O(MlogN+N^2).
+    NON DSU IMPL:
+        struct Edge {
+            int u, v, weight;
+            bool operator<(Edge const& other) {
+                return weight < other.weight;
+            }
+        };
+        int n;
+        vector<Edge> edges;
+        int cost = 0;
+        vector<int> tree_id(n);
+        vector<Edge> result;
+        for (int i = 0; i < n; i++)
+            tree_id[i] = i;
+        sort(edges.begin(), edges.end());
+        for (Edge e : edges) {
+            if (tree_id[e.u] != tree_id[e.v]) {
+                cost += e.weight;
+                result.push_back(e);
+                int old_id = tree_id[e.u], new_id = tree_id[e.v];
+                for (int i = 0; i < n; i++) {
+                    if (tree_id[i] == old_id)
+                        tree_id[i] = new_id;
+                }
+            }
+        }



    NON DSU IMPL:
        struct Edge {
            int u, v, weight;
            bool operator<(Edge const& other) {
                return weight < other.weight;
            }
        };

        int n;
        vector<Edge> edges;

        int cost = 0;
        vector<int> tree_id(n);
        vector<Edge> result;
        for (int i = 0; i < n; i++)
            tree_id[i] = i;

        sort(edges.begin(), edges.end());

        for (Edge e : edges) {
            if (tree_id[e.u] != tree_id[e.v]) {
                cost += e.weight;
                result.push_back(e);

                int old_id = tree_id[e.u], new_id = tree_id[e.v];
                for (int i = 0; i < n; i++) {
                    if (tree_id[i] == old_id)
                        tree_id[i] = new_id;
                }
            }
        }

    DSU implementation:

    Just as in the simple version of the Kruskal algorithm, we 
    sort all the edges of the graph in non-decreasing order of weights. 
    Then put each vertex in its own tree (i.e. its set) via calls to the make_set 
    function - it will take a total of O(N). We iterate through all the edges (in sorted order) 
    and for each edge determine whether the ends belong to different trees (with two find_set 
    calls in O(1) each). Finally, we need to perform the union of the two trees (sets), for 
    which the DSU union_sets function will be called - also in O(1). So we get the total 
    time complexity of O(MlogN+N+M) = O(MlogN).

    Here is an implementation of Kruskal's algorithm with Union by Rank:

        vector<int> parent, rank;

        void make_set(int v) {
            parent[v] = v;
            rank[v] = 0;
        }

        int find_set(int v) {
            if (v == parent[v])
                return v;
            return parent[v] = find_set(parent[v]);
        }

        void union_sets(int a, int b) {
            a = find_set(a);
            b = find_set(b);
            if (a != b) {
                if (rank[a] < rank[b])
                    swap(a, b);
                parent[b] = a;
                if (rank[a] == rank[b])
                    rank[a]++;
            }
        }

        struct Edge {
            int u, v, weight;
            bool operator<(Edge const& other) {
                return weight < other.weight;
            }
        };

        int n;
        vector<Edge> edges;

        int cost = 0;
        vector<Edge> result;
        parent.resize(n);
        rank.resize(n);
        for (int i = 0; i < n; i++)
            make_set(i);

        sort(edges.begin(), edges.end());

        for (Edge e : edges) {
            if (find_set(e.u) != find_set(e.v)) {
                cost += e.weight;
                result.push_back(e);
                union_sets(e.u, e.v);
            }
        }

        Notice: since the MST will contain exactly N−1 edges, 
        we can stop the for loop once we found that many.


2.312) Prims Impl:
    minimum spanning tree is built gradually by adding edges one at a time. 
    At first the spanning tree consists only of a single vertex (chosen arbitrarily). Then the 
    minimum weight edge outgoing from this vertex is selected and added to the spanning tree. 
    After that the spanning tree already consists of two vertices. Now select and add the edge 
    with the minimum weight that has one end in an already selected vertex (i.e. a vertex 
    that is already part of the spanning tree), and the other end in an 
    unselected vertex. And so on, i.e. every time we select and add the edge 
    with minimal weight that connects one selected vertex with one unselected vertex. 
    The process is repeated until the spanning tree contains all vertices (or equivalently until we have n−1 edges).

    In the end the constructed spanning tree will be minimal. If 
    the graph was originally not connected, then there doesn't 
    exist a spanning tree, so the number of selected edges will be less than n−1.

    Two impls discussed: O(N^2) and O(mlogn)


    Dense Graph Implementation: O(N^2)

    We approach this problem for a different side: for every not yet 
    selected vertex we will store the minimum edge to an already selected vertex.

    Then during a step we only have to look at these 
    minimum weight edges, which will have a complexity of O(n).

    After adding an edge some minimum edge pointers have to be recalculated. 
    Note that the weights only can decrease, i.e. the minimal weight edge of 
    every not yet selected vertex might stay the same, or it will be 
    updated by an edge to the newly selected vertex. 
    Therefore this phase can also be done in O(n).

    Thus we received a version of Prim's algorithm with the complexity O(n^2).

    In particular this implementation is very convenient for the Euclidean Minimum Spanning 
    Tree problem: we have n points on a plane and the distance between each pair 
    of points is the Euclidean distance between them, and we want to find a minimum 
    spanning tree for this complete graph. This task can be solved by the described 
    algorithm in O(n^2) time and O(n) memory, which is not possible with Kruskal's algorithm.

    The adjacency matrix adj[][] of size n×n stores the weights of the edges, and it 
    uses the weight INF if there doesn't exist an edge between two vertices. The 
    algorithm uses two arrays: the flag selected[], which indicates which vertices 
    we already have selected, and the array min_e[] which stores the edge with minimal 
    weight to an selected vertex for each not-yet-selected vertex (it stores the weight and the end vertex). 
    The algorithm does n steps, in each iteration the vertex with the smallest 
    edge weight is selected, and the min_e[] of all other vertices gets updated.


    int n;
    vector<vector<int>> adj; // adjacency matrix of graph
    const int INF = 1000000000; // weight INF means there is no edge

    struct Edge {
        int w = INF, to = -1;
    };

    void prim() {
        int total_weight = 0;
        vector<bool> selected(n, false);
        vector<Edge> min_e(n);
        min_e[0].w = 0;

        for (int i=0; i<n; ++i) {
            int v = -1;
            for (int j = 0; j < n; ++j) {
                if (!selected[j] && (v == -1 || min_e[j].w < min_e[v].w))
                    v = j;
            }

            if (min_e[v].w == INF) {
                cout << "No MST!" << endl;
                exit(0);
            }

            selected[v] = true;
            total_weight += min_e[v].w;
            if (min_e[v].to != -1)
                cout << v << " " << min_e[v].to << endl;

            for (int to = 0; to < n; ++to) {
                if (adj[v][to] < min_e[to].w)
                    min_e[to] = {adj[v][to], v};
            }
        }

        cout << total_weight << endl;
    }


+2.313) Prims With PriorityQueue C++:
+        // STL implementation of Prim's algorithm for MST
+        #include<bits/stdc++.h>
+        using namespace std;
+        # define INF 0x3f3f3f3f
+        
+        // iPair ==>  Integer Pair
+        typedef pair<int, int> iPair;
+        
+        // This class represents a directed graph using
+        // adjacency list representation
+        class Graph
+        {
+            int V;    // No. of vertices
+        
+            // In a weighted graph, we need to store vertex
+            // and weight pair for every edge
+            list< pair<int, int> > *adj;
+        
+        public:
+            Graph(int V);  // Constructor
+        
+            // function to add an edge to graph
+            void addEdge(int u, int v, int w);
+        
+            // Print MST using Prim's algorithm
+            void primMST();
+        };
+        
+        // Allocates memory for adjacency list
+        Graph::Graph(int V)
+        {
+            this->V = V;
+            adj = new list<iPair> [V];
+        }
+        
+        void Graph::addEdge(int u, int v, int w)
+        {
+            adj[u].push_back(make_pair(v, w));
+            adj[v].push_back(make_pair(u, w));
+        }
+        
+        // Prints shortest paths from src to all other vertices
+        void Graph::primMST()
+        {
+            // Create a priority queue to store vertices that
+            // are being preinMST. This is weird syntax in C++.
+            // Refer below link for details of this syntax
+            // http://geeksquiz.com/implement-min-heap-using-stl/
+            priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
+        
+            int src = 0; // Taking vertex 0 as source
+        
+            // Create a vector for keys and initialize all
+            // keys as infinite (INF)
+            vector<int> key(V, INF);
+        
+            // To store parent array which in turn store MST
+            vector<int> parent(V, -1);
+        
+            // To keep track of vertices included in MST
+            vector<bool> inMST(V, false);
+        
+            // Insert source itself in priority queue and initialize
+            // its key as 0.
+            pq.push(make_pair(0, src));
+            key[src] = 0;
+        
+            /* Looping till priority queue becomes empty */
+            while (!pq.empty())
+            {
+                // The first vertex in pair is the minimum key
+                // vertex, extract it from priority queue.
+                // vertex label is stored in second of pair (it
+                // has to be done this way to keep the vertices
+                // sorted key (key must be first item
+                // in pair)
+                int u = pq.top().second;
+                pq.pop();
+                
+                //Different key values for same vertex may exist in the priority queue.
+                //The one with the least key value is always processed first.
+                //Therefore, ignore the rest.
+                if(inMST[u] == true){
+                    continue;
+                }
+            
+                inMST[u] = true;  // Include vertex in MST
+        
+                // 'i' is used to get all adjacent vertices of a vertex
+                list< pair<int, int> >::iterator i;
+                for (i = adj[u].begin(); i != adj[u].end(); ++i)
+                {
+                    // Get vertex label and weight of current adjacent
+                    // of u.
+                    int v = (*i).first;
+                    int weight = (*i).second;
+        
+                    //  If v is not in MST and weight of (u,v) is smaller
+                    // than current key of v
+                    if (inMST[v] == false && key[v] > weight)
+                    {
+                        // Updating key of v
+                        key[v] = weight;
+                        pq.push(make_pair(key[v], v));
+                        parent[v] = u;
+                    }
+                }
+            }
+        
+            // Print edges of MST using parent array
+            for (int i = 1; i < V; ++i)
+                printf("%d - %d\n", parent[i], i);
+        }
+        
+2.314) PRIM VS KRUSKAL
+    If you implement both Kruskal and Prim, in their optimal form : with a union find and a 
+    finbonacci heap respectively, then you will note how Kruskal is easy to implement compared to Prim.
+    Prim is harder with a fibonacci heap mainly because you have to maintain a book-keeping 
+    table to record the bi-directional link between graph nodes and heap nodes. With a Union Find, 
+    it's the opposite, the structure is simple and can even produce directly the mst at almost no additional cost.
+    Use Prim's algorithm when you have a graph with lots of edges.
+    For a graph with V vertices E edges, Kruskal's algorithm runs in O(E log V) time 
+    and Prim's algorithm can run in O(E + V log V) amortized time, if you use a Fibonacci Heap.
+    Prim's algorithm is significantly faster in the limit when you've got a 
+    really dense graph with many more edges than vertices. Kruskal performs better in 
+    typical situations (sparse graphs) because it uses simpler data structures.
+    Kruskal's algorithm will grow a solution from the cheapest edge by 
+    adding the next cheapest edge, provided that it doesn't create a cycle.
+    Prim's algorithm will grow a solution from a random vertex by adding 
+    the next cheapest vertex, the vertex that is not currently in the 
+    solution but connected to it by the cheapest edge.  


2.32) ORDERED SET/BST IN ACITION: (optimally done with fibonnaci heaps however) 
    PRIMS ALGORITHM WITH RED BLACK TREES + SET!
    (usally done with HEAP)

    n the above described algorithm it is possible to interpret the 
    operations of finding the minimum and modifying some values as set 
    operations. These two classical operations are supported by many 
    data structure, for example by set in C++ (which are implemented via red-black trees).

    The main algorithm remains the same, but now we can find the minimum 
    edge in O(logn) time. On the other hand recomputing the pointers 
    will now take O(nlogn) time, which is worse than in the previous algorithm.

    But when we consider that we only need to update O(m) times in total, 
    and perform O(n) searches for the minimal edge, then the total 
    complexity will be O(mlogn). For sparse graphs this is better 
    than the above algorithm, but for dense graphs this will be slower.

    Here the graph is represented via a adjacency list adj[], where adj[v] 
    contains all edges (in form of weight and target pairs) for the vertex v. 
    min_e[v] will store the weight of the smallest edge from vertex v to an 
    already selected vertex (again in the form of a weight and target pair). 
    In addition the queue q is filled with all not yet selected vertices in 
    the order of increasing weights min_e. The algorithm does n steps, on each 
    of which it selects the vertex v with the smallest weight min_e (by extracting 
    it from the beginning of the queue), and then looks through all the edges 
    from this vertex and updates the values in min_e (during an update we also 
    need to also remove the old edge from the queue q and put in the new edge).



    const int INF = 1000000000;

    struct Edge {
        int w = INF, to = -1;
        bool operator<(Edge const& other) const {
            return make_pair(w, to) < make_pair(other.w, other.to);
        }
    };

    int n;
    vector<vector<Edge>> adj;

    void prim() {
        int total_weight = 0;
        vector<Edge> min_e(n);
        min_e[0].w = 0;
        set<Edge> q;
        q.insert({0, 0});
        vector<bool> selected(n, false);
        for (int i = 0; i < n; ++i) {
            if (q.empty()) {
                cout << "No MST!" << endl;
                exit(0);
            }

            int v = q.begin()->to;
            selected[v] = true;
            total_weight += q.begin()->w;
            q.erase(q.begin());

            if (min_e[v].to != -1)
                cout << v << " " << min_e[v].to << endl;

            for (Edge e : adj[v]) {
                if (!selected[e.to] && e.w < min_e[e.to].w) {
                    q.erase({min_e[e.to].w, e.to});
                    min_e[e.to] = {e.w, v};
                    q.insert({e.w, e.to});
                }
            }
        }

        cout << total_weight << endl;
    }




2.322) 

    US Based role, recruiter reached out and practised around 3 week prior to phone screen.

    Got asked around 15 min behavior questions before the coding round. Only one question was asked. 
    The exact question cannot be found in leetcode but it's similer to this:

    https://leetcode.com/problems/sliding-window-median/description/
    Instead of a 1D sliding window, I was asked to apply a 2D sliding window as if the 1D array
    was a (m * n) array, with a k sliding window on the pixals around it.

    It took me around 40 minutes to write everything. The result is only 20 lines eventually 
    so initially I was quite defeted as I thought I missed a easy question. The code was 
    passing all the dry runs during the interview, but after the interview I realized that some 
    special cases it would fail. I was stopped around 40 minute markd and he continued 
    asking around 10 minutes about it's time complexity which I answered about right but he also hinted a lot.

    The feedback from the recruiter mentioned that I communicated well to the interviewer and was quick on grabbing the hints.
    

    Normal median finding is with 2 heaps right? yes. 
    or wait.. 


    Giving back to community.

    I gave my Meta Phone Screen today. Waiting for the result.

    Questions I received :
    https://leetcode.com/problems/nested-list-weight-sum/
    https://leetcode.com/problems/valid-word-abbreviation/description/




2.3222) READ THIS QUICKLY DP:
            Unlocking the Power of Dynamic Programming: Principles, Practice, and Significance
            Dynamic Programming (DP) stands as a cornerstone technique in algorithm design, offering a 
            systematic approach to solving optimization problems efficiently. Its essence lies in breaking 
            down complex problems into simpler subproblems, solving them independently, and then combining 
            their solutions to derive the optimal solution for the main problem. 

            Understanding the Inner Workings of Dynamic Programming:
            1. Optimal Substructure:

            DP problems exhibit optimal substructure, implying that an optimal solution for the main problem can be constructed from optimal solutions to its subproblems.

            This characteristic allows us to decompose the main problem into smaller, 
            manageable subproblems, facilitating a divide-and-conquer strategy for problem-solving.
            2. Overlapping Subproblems:

            Another critical aspect of DP is the presence of overlapping subproblems, 
            where the same subproblem is encountered and solved multiple times during the computation process.
            DP leverages this repetition by storing the solutions to overlapping subproblems in a data structure, 
            thus avoiding redundant computations and enhancing efficiency.
            
            Core Steps in Dynamic Programming:
            1. Identify Subproblems:

            Begin by identifying the subproblems that exhibit optimal substructure, 
            breaking down the main problem into smaller, solvable components.
            These subproblems are usually interrelated and contribute to solving the overarching problem.
            
            2. Formulate Recurrence Relations:

            Define recurrence relations that express the solution to each subproblem in terms of solutions to its smaller subproblems.
            These recurrence relations serve as the foundation for DP solutions, providing a 
            structured way to compute optimal solutions iteratively.
            
            3. Memoization or Tabulation:

            DP implementations often employ memoization or tabulation techniques.
            Memoization involves storing the solutions to subproblems in a cache (such as a dictionary or array) to avoid recomputation.
            Tabulation, on the other hand, is a bottom-up approach where solutions to subproblems are computed iteratively and stored in a table-like structure.

            4. Construct Optimal Solution:

            Once all subproblems are solved, and their solutions are stored via memoization or tabulation, construct the optimal solution to the main problem using these precomputed solutions.
            This step typically involves tracing back through the DP table or cache to derive the optimal path or outcome.
            Memoization Approach:
            Memoization is a top-down approach to DP, where solutions to subproblems are stored in a cache (such as a dictionary or an array) to avoid redundant computations. It's particularly effective when solving recursive problems with overlapping subproblems. Here's a step-by-step guide to the memoization approach:

            1. Identify Subproblems:

            Begin by identifying the subproblems within the main problem that exhibit optimal substructure.
            These subproblems should be distinct and reusable, contributing to the solution of the main problem.
            2. Define Recursive Function:

            Create a recursive function that represents the problem-solving logic, 
            taking parameters that define the current state or subproblem.
            Within the recursive function, incorporate base cases to handle trivial or terminating conditions.
            3. Implement Memoization Cache:

            Initialize a cache (e.g., a dictionary or an array) to store solutions to subproblems.
            Before computing the solution to a subproblem, check if it already exists in the cache. 
            If so, return the cached solution instead of recomputing.
            4. Recursion with Memoization:

            Modify the recursive function to utilize memoization. Upon computing the solution to a subproblem, store it in the cache for future reference.
            When encountering a subproblem that has already been solved and cached, retrieve the solution from the cache to avoid redundant computations.
            5. Return Optimal Solution:

            As the recursive calls unwind, the optimal solution to the main problem is constructed using the solutions stored in the memoization cache.
            Return the final optimal solution computed by the memoization-enhanced recursive function.

            Tabulation Approach:
            Tabulation is a bottom-up approach to DP, where solutions to subproblems are computed 
            iteratively and stored in a table-like structure (such as an array or matrix). 
            It's suitable for problems with well-defined states and optimal substructure. 
            Here's a detailed guide to the tabulation approach:

            1. Define DP Table:

            Begin by defining a DP table, typically a multi-dimensional array or matrix, to store solutions to subproblems.
            Determine the dimensions of the table based on the problem's states or parameters.
            2. Initialize Base Cases:

            Populate the initial rows or columns of the DP table with base case values that represent trivial or starting conditions.
            Base cases serve as the foundation for computing solutions to larger subproblems.
            3. Iterative Computation:

            Iterate through the DP table in a systematic order, filling in entries based on solutions to smaller subproblems.
            Follow a specific order of computation that ensures dependencies between subproblems are addressed correctly.
            4. Update DP Table:

            As solutions to subproblems are computed iteratively, update the DP table with the computed values.
            Ensure that each entry in the DP table represents the optimal solution to the corresponding subproblem.
            5. Derive Optimal Solution:

            Once the DP table is fully populated with solutions to all subproblems, 
            the optimal solution to the main problem is derived from the final entries of the DP table.
            Traverse the DP table or follow a predefined path to extract the optimal solution.
            Comparison and Use Cases:
            Memoization: Ideal for problems with recursive structures and overlapping subproblems, 
            where the focus is on solving specific subproblems efficiently and reusing their solutions.

            Tabulation: Suited for problems with well-defined states or parameters, where a systematic and 
            iterative approach to computing solutions is preferred, leading to a structured 
            DP table representing optimal solutions.

            Both memoization and tabulation are powerful techniques in DP, offering different approaches to 
            problem-solving based on the problem's characteristics and requirements. Understanding when to
            apply each technique is key to leveraging Dynamic Programming effectively across a diverse range of optimization problems.

            DP ALSO USES FINITE STATE MACHIENS TO TRAVERSE DIFFERENT STATES WITHIN SUBPROBLEMS, LEVERAGE IT. 
            
            Practice Questions for Dynamic Programming Mastery:
            70. Climbing Stairs
            322. Coin Change
            300. Longest Increasing Subsequence
            53. Maximum Subarray
            198. House Robber
            62. Unique Paths
            72. Edit Distance
            5. Longest Palindromic Substring
            123. Best Time to Buy and Sell Stock III
            122. Best Time to Buy and Sell Stock II
            139. Word Break
            91. Decode Ways
            96. Unique Binary Search Trees
            55. Jump Game
            1143. Longest Common Subsequence
            10. Regular Expression Matching
            13. Paint House
            518. Coin Change II
            338. Counting Bits
            647. Palindromic Substrings
            64. Minimum Path Sum










2.325) DFS ANALYSIS START AND END TIMES!
    The parenthesis theorem says that the discovery 
    and finish time intervals are either disjoint or nested.

    With the graph version of DFS, only some edges 
    (the ones for which visited[v] is false) will be traversed. 
    These edges will form a tree, called the depth-first-search 
    tree of G starting at the given root, and the edges in this
    tree are called tree edges. The other edges of G can be 
    divided into three categories:

    Back edges point from a node to one of its ancestors in the DFS tree.
    Forward edges point from a node to one of its descendants.
    Cross edges point from a node to a previously visited 
    node that is neither an ancestor nor a descendant.

    AnnotatedDFS(u, parent):
        parent[u] = parent
        start[u] = clock; clock = clock + 1
        visited[u] = true
        for each successor v of u:
            if not visited[v]:
            AnnotatedDFS(v, u)
        end[u] = clock; clock = clock + 1

    we will show in a moment that a graph is acyclic if and 
    only if it has no back edges. But first: how do we tell 
    whether an edge is a tree edge, a back edge, a forward edge, 
    or a cross edge? We can do this using the clock mechanism 
    we used before to convert a tree into a collection of intervals.


    Tree edges are now easy to recognize; uv is a tree edge 
    if parent[v] = u. For the other types of edges, 
    we can use the (start,end) intervals to tell 
    whether v is an ancestor, descendant, or distant cousin of u:

    Edge type of uv | start times         | end times
    Tree edge       | start[u] < start[v] | end[u] > end[v]
    Back edge       | start[u] > start[v] | end[u] < end[v]  (THIS IS JUST THE OPPOSITE TIMESTAMPS AS TREE/FORWARD ENDGE)
    Forward edge    | start[u] < start[v] | end[u] > end[v]   (START AND FORWARD HAVE THE SAME TIME STAMPS!!!)
    Cross edge      | start[u] > start[v] | end[u] > end[v] (THIS IS JUST ALL OTHER CASES! BUT THE EDGE U VISITED STARTED BEFORE U. )

    TREE/FORWARD EDGE -> STARTED AFTER YOU AND ENDED EARLIER THAN YOU.

    BACK EDGE -> STARTED BEFORE YOU, AND ENDED AFTER YOU.

    CROSS EDGE -> STARTED BEFORE YOU, AND ENDED BEFORE YOU.


+2.33) Storing 2 integer values at same index in an array:
+    First we have to find a value greater than 
+    all the elements of the array. Now we can store the 
+    original value as modulus and the second value as division. 
+    Suppose we want to store arr[i] and arr[j] both at index 
+    i(means in arr[i]). First we have to find a ‘maxval’ 
+    greater than both arr[i] and arr[j]. Now we can store 
+    as arr[i] = arr[i] + arr[j]*maxval. Now arr[i]%maxval 
+    will give the original value of arr[i] and arr[i]/maxval 
+    will give the value of arr[j].



2.33) Construct the Rooted Tree by using start and finish time of its 
      DFS traversal: 


    Given start and finish times of DFS traversal of N vertices 
    that are available in a Rooted tree, the task is 
    to construct the tree (Print the Parent of each node).

    Parent of the root node is 0.

    Examples:

    Input: Start[] = {2, 4, 1, 0, 3}, End[] = {3, 5, 4, 5, 4}
    Output: 3 4 4 0 3
    Given Tree is -:
                        4(0, 5)
                        /   \
                (1, 4)3     2(4, 5)
                    /  \    
            (2, 3)1    5(3, 4)


    The root will always have start time = 0
    processing a node takes 1 unit time but backtracking 
    does not consume time, so the finishing time 
    of two nodes can be the same.

    Input: Start[] = {4, 3, 2, 1, 0}, End[] = {5, 5, 3, 3, 5}
    Output: 2 5 4 5 0

    Root of the tree is the vertex whose starting time is zero.

    Now, it is sufficient to find the descendants of a vertex, 
    this way we can find the parent of every vertex.

    Define Identity[i] as the index of the vertex with starting equal to i.
    As Start[v] and End[v] are starting and ending time of vertex v.
    The first child of v is Identity[Start[v]+1] and
    the (i+1)th is Identity[End[chv[i]]] where chv[i] is the ith child of v.
    Traverse down in DFS manner and update the parent of each node.
    
    Time Complexity : O(N)
    where N is the number of nodes in the tree.
    
    def Restore_Tree(S, E): 
    
        # Storing index of vertex with starting 
        # time Equal to i 
        Identity = N*[0]   
    
        for i in range(N): 
            Identity[Start[i]] = i 
    
        # Parent array 
        parent = N*[-1] 
        curr_parent = Identity[0] 
        
        for j in range(1, N): 
    
            # Find the vertex with starting time j 
            child = Identity[j] 
    
            # If end time of this child is greater than  
            # (start time + 1), then we traverse down and  
            # store curr_parent as the parent of child 
            if End[child] - j > 1: 
                parent[child] = curr_parent 
                curr_parent = child 
    
            # Find the parent of current vertex 
            # over iterating on the finish time 
            else:      
                parent[child] = curr_parent 
    
                # Backtracking takes zero time 
                while End[child]== End[parent[child]]: 
                    child = parent[child] 
                    curr_parent = parent[child] 
                    if curr_parent == Identity[0]: 
                        break
        for i in range(N): 
            parent[i]+= 1
    
        # Return the parent array 
        return parent 
    
    # Driver Code  
    if __name__=="__main__": 
        N = 5
    
        # Start and End time of DFS 
        Start = [2, 4, 1, 0, 3] 
        End = [3, 5, 4, 5, 4] 
        print(*Restore_Tree(Start, End)) 

2.35) In head recursion , the recursive call, when it happens, comes 
      before other processing in the function (think of it happening at the top, 
      or head, of the function). In tail recursion , it's the 
      opposite—the processing occurs before the recursive call.

2.37) Articulation Points Algorithm:
    A cut vertex is a vertex that when removed (with its boundary edges) 
    from a graph creates more components than previously in the graph. 
    A recursive function that find articulation points  using DFS traversal 
    
    In DFS tree, a vertex u is articulation point if one of the 
    following two conditions is true.
    
    1) u is root of DFS tree and it has at least two children.
       the first case is simple to detect. For every vertex, count children. 
       If currently visited vertex u is root (parent[u] is NIL) 
       and has more than two children, print it.
    
    2) u is not root of DFS tree and it has a child v such that no vertex 
       in subtree rooted with v has a back edge to one of the ancestors (in DFS tree) of u.

        How to handle second case? The second case is trickier. 
        We maintain an array disc[] to store discovery time of vertices. 
        For every node u, we need to find out the earliest visited vertex 
        (the vertex with minimum discovery time) that can be reached 
        from subtree rooted with u. So we maintain an additional array 
        low[] which is defined as follows.

        low[u] = min(disc[u], disc[w]) 
        where w is an ancestor of u and there is a back edge from 
        some descendant of u to w.


2.4) CUT VERTEX AKA ARTICULATION POINT finding:
        u --> The vertex to be visited next 
        visited[] --> keeps tract of visited vertices 
        disc[] --> Stores discovery times of visited vertices 
        parent[] --> Stores parent vertices in DFS tree 
        ap[] --> Store articulation points

        def AP(self): 
            visited = [False] * (self.V) 
            disc = [float("Inf")] * (self.V) 
            low = [float("Inf")] * (self.V) 
            parent = [-1] * (self.V) 
            ap = [False] * (self.V) #To store articulation points 
    
            # Call the recursive helper function 
            # to find articulation points 
            # in DFS tree rooted with vertex 'i' 
            for i in range(self.V): 
                if visited[i] == False: 
                    self.APUtil(i, visited, ap, parent, low, disc) 
    
            for index, value in enumerate (ap): 
                if value == True: print index, 

        def APUtil(self,u, visited, ap, parent, low, disc): 
            #Count of children in current node  
            children = 0
    
            # Mark the current node as visited and print it 
            visited[u]= True
    
            # Initialize discovery time and low value 
            disc[u] = self.Time 
            low[u] = self.Time 
            self.Time += 1
    
            #Recur for all the vertices adjacent to this vertex 
            for v in self.graph[u]: 
                # If v is not visited yet, then make it a child of u 
                # in DFS tree and recur for it 
                if visited[v] == False : 
                    parent[v] = u 
                    children += 1
                    self.APUtil(v, visited, ap, parent, low, disc) 
    
                    # Check if the subtree rooted with v has a connection to 
                    # one of the ancestors of u 
                    low[u] = min(low[u], low[v]) 
    
                    # u is an articulation point in following cases 
                    # (1) u is root of DFS tree and has two or more chilren. 
                    if parent[u] == -1 and children > 1: 
                        ap[u] = True
    
                    #(2) If u is not root and low value of one of its child is more 
                    # than discovery value of u. 
                    if parent[u] != -1 and low[v] >= disc[u]: 
                        ap[u] = True    
                        
                    # Update low value of u for parent function calls     
                elif v != parent[u]:  # found backedge! 
                    low[u] = min(low[u], disc[v]) 


2.5) CUT EDGE AKA BRIDGE finding. 
    
    '''A recursive function that finds and prints bridges 
    using DFS traversal 
    low[w] is the lowest vertex reachable in a subtree rooted at w

    u --> The vertex to be visited next 
    visited[] --> keeps tract of visited vertices 
    disc[] --> Stores discovery times of visited vertices 
    parent[] --> Stores parent vertices in DFS tree'''
    def bridge(self): 
   
        # Mark all the vertices as not visited and Initialize parent and visited,  
        # and ap(articulation point) arrays 
        visited = [False] * (self.V) 
        disc = [float("Inf")] * (self.V) 
        low = [float("Inf")] * (self.V) 
        parent = [-1] * (self.V) 
  
        # Call the recursive helper function to find bridges 
        # in DFS tree rooted with vertex 'i' 
        for i in range(self.V): 
            if visited[i] == False: 
                self.bridgeUtil(i, visited, parent, low, disc)

    def bridgeUtil(self,u, visited, parent, low, disc): 
  
        # Mark the current node as visited and print it 
        visited[u]= True
  
        # Initialize discovery time and low value 
        disc[u] = self.Time 
        low[u] = self.Time 
        self.Time += 1
  
        #Recur for all the vertices adjacent to this vertex 
        for v in self.graph[u]: 
            # If v is not visited yet, then make it a child of u 
            # in DFS tree and recur for it 
            if visited[v] == False : 
                parent[v] = u 
                self.bridgeUtil(v, visited, parent, low, disc) 
  
                # Check if the subtree rooted with v has a connection to 
                # one of the ancestors of u 
                # this value will become smaller if v has a backedge that goes to an 
                # ancestor of u
                low[u] = min(low[u], low[v]) 
  
  
                ''' If the lowest vertex reachable from subtree 
                under v is below u in DFS tree, then u-v is 
                a bridge'''
                if low[v] > disc[u]: 
                    print ("%d %d" %(u,v)) 
      
                      
            elif v != parent[u]: # Update low value of u for parent function calls. 
                low[u] = min(low[u], disc[v]) 




+2.36) Learn to use iterators: Serialize and Deserialize bin tree preorder style:
+
+    class Codec:
+        def serialize(self, root):
+            def doit(node):
+                if node:
+                    vals.append(str(node.val))
+                    doit(node.left)
+                    doit(node.right)
+                    vals.append('#')
+            vals = []
+            doit(root)
+            return ' '.join(vals)
+        def deserialize(self, data):
+            def doit():
+                val = next(vals)
+                if val == '#':
+                    return None
+                node = TreeNode(int(val))
+                node.left = doit()
+                node.right = doit()
+                return node
+            vals = iter(data.split())
+            return doit()
+2.37) GREEDY HILL FINDING WITH REVERSE POINTERS, 
+     AKA MOST IMPORTANT INDEXES ONLY FINDING AND USING SMARTLY 
+     AKA MONOQUEUE EXTENSION
+    Some problems require you to find optimal hills, to get answer. 
+    These hills are valid for certain indexes, and then you have to use new hills
+    They have a sort of max min aura to them, and seem similar to monoqueue type 
+    problems.
+    When you see a max-min type optimization pattern, then you have to find HILLS:
+    
+    For instance:
+    Input a = [21,5,6,56,88,52], output = [5,5,5,4,-1,-1] . 
+    Output array values is made up of indices of the 
+    element with value greater than the current element 
+    but with largest index. So 21 < 56 (index 3), 
+    21 < 88 (index 4) but also 21 < 52 (index 5) 
+    so we choose index 5 (value 52). 
+    Same applies for 5,6 and for 56 its 88 (index 4).
+    
+    Algorithm 1: Find the hills, and binsearch the indexes: 
+    need to keep track of biggest element on right side. 
+    on the right side, keep the hills!
+    52, is a hill, 
+    then 88, because its bigger than 52,
+    not 56, not 6, not 5, not 21, because you can just use 52, or 88 
+    so elements check against 52 first, then against 88. 
+    
+    import bisect
+    def soln(arr):
+        hills = []
+        hill_locations = []
+        running_max = float("-inf")
+        for i in range(len(arr)-1, -1, -1):
+            if running_max < arr[i]:
+                running_max = arr[i]
+                hills.append(arr[i])
+                hill_locations.append(i)
+        hill_locations_pop_idx = hill_locations[-1]
+        ans = []
+        def bin_search(arr, val):
+            l = 0
+            r = len(arr) 
+            mid = None
+            while l != r:
+                mid = l + (r-l)//2
+                if arr[mid]  == val:
+                    return mid 
+                elif arr[mid] > val:
+                    r = mid 
+                else:
+                    l = mid  + 1
+            return l # what happens if you returned mid here would that still work?
+                     # we check below. 
+                     # but i think it would be incorrect
+                     # because we always want the one index left of mid at the very end. 
+        for i in range(len(arr)):
+            if i == hill_locations_pop_idx:
+                # you have to invalidate indexes because you dont want to 
+                # invalid indexes to be found in bin search.
+                hill_locations.pop()
+                hills.pop()
+                hill_locations_pop_idx = -1 if len(hill_locations) == 0 else hill_locations[-1]
+            # Locate the insertion point for x in a to maintain sorted order.
+            x = bisect.bisect_left(hills, arr[i], lo=0, hi=len(hills))
+            y = bin_search(hills, arr[i])
+            print("x, y", (x, y)) # will be same
+            if y < len(hill_locations):
+                ans.append(hill_locations[x])
+            else:
+                ans.append(-1)
+        return ans  
+    Algorithm 2: Insert everything in pq. Pop off 1 by 1, check running max idx. and assign idx.
+    -> i dont get how this method works actually...  
+    // do you pop it off and push it back in or some shit?
+    // pq based on index?
+    if max val is too big its aight kepe it,
+    If its smaller, at a lower idx, throw it away, otherwise keep it?
+2.38) SIMULATE BINARY SEARCH INSERTION POINT FINDER  
+     AKA bisect.bisect_left(arr, val, lo=0, hi=len(arr)) PART 1
+    -> 
+    The returned insertion point i partitions the array a into two halves so that 
+    all(val < x for val in a[lo:i]) for the left side and all(val >= x for val in a[i:hi]) 
+    for the right side.
+    # BTW THIS CODE LOOKS DIFFERENT FROM THE BINARY SEARCH TEMPLATE SECTION BELOW
+    # Locate the insertion point for x in a to maintain sorted order.
+    # REMEMBER THAT THE FINAL ANSWER IS LOW NOTTTTT MID
+    # HERE WE INITIALIZED RIGHT AS LEN(NUMS) - 1
+    def searchInsert(self, nums, target):
+        low = 0
+        high = len(nums) - 1
+        while low <= high:
+            mid = (low + high) / 2
+            if nums[mid] == target:
+                return mid
+            elif nums[mid] < target:
+                low = mid + 1
+            else:
+                high = mid - 1
+        return low
+    # THE ABOVE SOLUTION WORKS ONLY IF WE RETURN LOW, NOT MEDIUM OR HIGH
+    # IN OTHER WORDS, WHAT YOU RETURN LOW/MID/HIGH IS PROBLEM SPECFIC!
+    Wrong Answer
+    Details 
+    Input
+    [1,3,5,6], 2
+    Output: 0
+    Expected: 1
+2.385) SIMULATE BINARY SEARCH INSERTION POINT FINDER  PART 2
+    # BTW THIS CODE LOOKS DIFFERENT FROM THE BINARY SEARCH TEMPLATE SECTION BELOW
+    Logic Flow of Solving Binary Search Problems
+        Choose lo & hi
+        Always double check what is the maximum range of possible values. For example, 
+        <LeetCode 35>, since it's possible to insert a value at the very end, 
+        the boundary for this problem is actually 0 - n.
+        Calculate mid
+        Always use the following, since it avoids overflow.
+        // when odd, return the only mid
+        // when even, return the lower mid
+        int mid = lo + ((hi - lo)/2);
+        // when odd, return the only mid
+        // when even, return the upper mid
+        int mid2 = lo + ((hi - lo + 1) / 2);
+        How to move lo and hi?
+        Always use a condition we are 100% sure of. It's always easier to eliminate 
+        options when we are 100% sure of something. For eample, if we are we are looking 
+        for target <= x, then for target>nums[mid] , we are 100% sure that our mid should 
+        never be considered. Thus we can type lo = mid + 1 with all the confidence.
+                if (100% sure logic) {
+                    left = mid + 1; // 100% sure target is to the right of mid
+                } else {
+                    right = mid; 
+                }
+                
+                if (100% sure logic) {
+                    right = mid - 1; // 100% sure target is to the left of mid
+                } else {
+                    left = mid;
+                }
+        while Condition
+        Always use while (lo < hi) so when the loop breaks, we are 100% sure that lo == hi
+        If it's possible that target doesn't exist, extra check needs to be performed.
+        🔥Avoid Infinite loop
+        // ❌ The following code results in inifite loop
+        let mid = lo + ((hi - lo)/2); // aka the lower mid
+        // We should use:
+        // let mid = lo + ((hi - lo + 1)/2) // aka the upper mid
+
+        if (100% sure logic) {
+            right = mid - 1
+        } else {
+            left = mid // <-- note here
+        }
+        Consider when there's only 2 elements left, if the if condition goes to the else statement, 
+        since left = mid, our left boundary will not shrink, 
+        this code will loop for ever. Thus, we should use the upper mid.
+        // ❌ The following code results in inifite loop
+        let mid = lo + ((hi - lo + 1)/2); // aka the upper mid
+        // We should use:
+        // let mid = lo + ((hi - lo)/2) // aka the lower mid
+        if (100% sure logic) {
+            left = mid + 1;
+        } else {
+            right = mid // <-- note here
+        }
+        
+        Consider when there's only 2 elements left, if the if condition goes to the else statement, 
+        since right = mid our right boundary will not shrink, this code will loop for ever. 
+        Thus, we should use the lower mid.
+
+        Take Away
+        * Always think of the situation where there's only 2 elements left!
+
+    ANSWER 1:
+        var searchInsert = function(nums, target) {
+            let lo = 0, hi = nums.length; // we might need to inseart at the end
+            while(lo < hi) { // breaks if lo == hi
+                let mid = lo + Math.floor((hi-lo)/2); // always gives the lower mid
+                if (target > nums[mid]) {
+                    lo = mid + 1 // no way mid is a valid option
+                } else {
+                    hi = mid // it might be possibe to inseart @ mid
+                }
+            }
+            return lo;
+        };
+2.39) SIMULATE BINARY SEARCH INSERTION POINT FINDER  PART 3 (Binary search with post processing)
+    First of all, we assume [left, right] is the possible answer range(inclusive) for this question. 
+    So initially left = 0; and right = n - 1;
+    we calculate int mid = left + (right - left)/2; rather than int mid = (left + right)/2; to avoid overflow.
+    Clearly, if A[mid] = target; return mid;
+    if A[mid] < target, then since we can insert target into mid + 1, so the minimum 
+    possible index is mid + 1. That's the reason why we set left = mid + 1;(1)
+    if A[mid] > target, then notice here(important!) that: we can insert
+    target into mid, so mid can be the potential candidate. For example:
+    Then how to determine the while loop condition?
+    left < right, left <= right，left < right - 1 are probally 
+    all the possible writings for a binary search problem.
+    Then how to determine the while loop condition?
+    left < right, left <= right，left < right - 1 are probally all the possible writings 
+    for a binary search problem.
+    The answer is that we need to test it out by ourselves with our left/right operation:
+    left = mid + 1;
+    right = mid;
+    You may find it very difficult and time consuming to figure it out. 
+    But if you are familiar with this analysis for while loop, 
+    you can give the answer very quickly, piece of cake.
+    Let's assume there are 3 elements left at last
+    5 	7 	9
+    l	m   h
+    we can see that left = mid + 1 and right = mid can shrink the size by 2 and 1, 
+    so 3 elements will not result in dead loop.
+    So we reduce it to 2 elements:
+    5 	  7
+    l/m   h
+    Same way, we can see that left = mid + 1 and right = mid can both shrink the size by 1, no dead loop as well.
+    So we can safely reduce it to only 1 element:
+    5
+    l/m/h
+    we can see that left = mid + 1 will not cause dead loop, but with right = mid 
+    we cannot shrink the size, so we will enter a dead loop if we goes to the case: right = mid.
+    So we can determine that we need break/jump out of the loop when there is 
+    only 1 element left, i.e. while(left < right)
+    At the end, we need to check the last element: nums[left/right] which has not 
+    been checked in binary search loop with target to determine the index. We call it the post processing part.
+    ANSWER:
+    class Solution {
+	public int searchInsert(int[] nums, int target) {
+		if(nums == null || nums.length == 0) return 0;
+		
+		int n = nums.length;
+		int left = 0;
+		int right = n - 1;
+		while(left < right){
+			int mid = left + (right - left)/2;
+			
+			if(nums[mid] == target) return mid;
+			else if(nums[mid] > target) right = mid; // right could be the result
+			else left = mid + 1; // mid + 1 could be the result
+		}
+		
+		// 1 element left at the end
+		// post-processing
+		return nums[left] < target ? left + 1: left;
+        }
+    }
+2.40)SIMULATE BINARY SEARCH INSERTION POINT FINDER 
+     AKA bisect.bisect_left(arr, val, lo=0, hi=len(arr))  PART 4
+    # HERE WE INITIALIZED RIGHT AS LEN(NUMS) KNOW THE DIFFERENCE. 
+    def searchInsert(self, nums: List[int], target: int) -> int:
+        
+        l = 0
+        r = len(nums)
+        mid = None
+        
+        while l != r:            
+            mid = l + (r-l)//2  # calculates lower mid/not upper mid
+            if nums[mid] == target:
+                return mid
+            elif nums[mid] < target:
+                l = mid + 1
+            else:
+                r = mid
+        # DO NOT RETURN MID, RETURN L
+        return l




2.55) MATRIX Problems Tips:
      Try reversing. Try transposing. Try circular sorting. 
      Flipping on x axis or y axis is just reversing. 

      MATRIX ROTATION:
        HARMAN SOLN:

        def rotate(self, matrix):
            n = len(matrix)
            N = len(matrix)
            indexN = N - 1
            
            for d in range(n//2):
                swaps_to_do_this_layer = len(matrix) - 2*d - 1
                # Swap everything except the last element. that is 
                # automatically swapped on the first swap in the loop
                for i in range( swaps_to_do_this_layer ):              
                    # CONSIDER D AS THE BOUNDARY (with the help of indexN) AND 
                    # I AS THE OFFSET TO THE ELEMENTS WITHIN BOUNDARY
                    # I should only be offsetting one side, either a row, or a column
                    
                    northR, northC = d, i+d
                    eastR, eastC = i + d, indexN - d
                    southR, southC = indexN - d, indexN - d - i
                    westR, westC = indexN - d - i, d
                    
                    matrix[northR][northC], matrix[eastR][eastC], matrix[southR][southC], matrix[westR][westC] =\
                        matrix[westR][westC], matrix[northR][northC], matrix[eastR][eastC], matrix[southR][southC]
        SMARTER WAY:
        def rotate(self, matrix):
            n = len(matrix)
            for l in xrange(n / 2):
                r = n - 1 - l
                for p in xrange(l, r):
                    q = n - 1 - p
                    cache = matrix[l][p]
                    matrix[l][p] = matrix[q][l]
                    matrix[q][l] = matrix[r][q]
                    matrix[r][q] = matrix[p][r]
                    matrix[p][r] = cache     

        REVERSE - TRANSPOSE:

        def rotate(self, matrix):
            n = len(matrix)
            matrix.reverse()
            for i in xrange(n): # top half triangle transpose
                for j in xrange(i + 1, n):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # walks over the "top-left quadrant" 
        # of the matrix and directly rotates each element with the 
        # three corresponding elements in the other three quadrants. 
        # Note that I'm moving the four elements in 
        # parallel and that [~i] is way nicer than [n-1-i].

        class Solution:
            def rotate(self, A):
                n = len(A)
                for i in range(n/2):
                    for j in range(n-n/2):
                        A[i][j], A[~j][i], A[~i][~j], A[j][~i] = \
                                A[~j][i], A[~i][~j], A[j][~i], A[i][j]

        # Flip Flip, all by myself - 48 ms

        # Similar again, but I first transpose and then flip 
        # left-right instead of upside-down, and do it all 
        # by myself in loops. This one is 100% in-place 
        # again in the sense of just moving the elements.

        class Solution:
            def rotate(self, A):
                n = len(A)
                for i in range(n): # bottom half triangle transpose
                    for j in range(i):
                        A[i][j], A[j][i] = A[j][i], A[i][j]
                for row in A:
                    for j in range(n/2):
                        row[j], row[~j] = row[~j], row[j]


2.57) To find the root nodes in a directed graph (NOT DAG):
      Reverse graph and find nodes with 0 children.
      However, there may not be root nodes!

2.575) REMEMBER CYCLE DETECING ON DIRECTED GRAPH ALWAYS NEEDS AT LEAST 3 COLORS!!!
     WHILE ON UNDIRECTED YOU COULD JUST USE A VISTED SET. (OR FOR DIRECTED,
     JUST REMEMBER TO POP THE PROCESSED ELEMENT FROM THE VISITED PATH, SO THAT A 
     DIFFERENT DIRECTED PATH CAN VISIT THE PROCESSED ELEMENT)
     


2.58) Cycle finding in directed graph and undirected graph is 
      completely different! Memorize details of each way. 

    Directed graph cycle finding: Course Schedule (LC):
    There are a total of n courses you have to take, 
    labeled from 0 to n-1.

    Some courses may have prerequisites, for example 
    to take course 0 you have to first take course 1, 
    which is expressed as a pair: [0,1]

    Given the total number of courses and a list of prerequisite 
    pairs, is it possible for you to finish all courses?
    
    Method 1 COLORED DFS:
        def canFinishWithColoredDFS(self, numCourses, prerequisites):        
            g = defaultdict(set)

            for req in prerequisites:
                g[req[1]].add(req[0])
            
            def has_cycle_directed(node, g, colors):
                colors[node] = "G" # Grey being processed
                
                for c in g[node]:
                    if colors.get(c) is None and has_cycle_directed(c, g, colors):
                        return True
                    elif colors.get(c) == "G":
                        # We are processing this node but we looped back around somehow
                        # so cycle
                        return True
                    else: 
                        # The node we are processing has already been processed. 
                        continue  
                colors[node] = "B" # Black
                return False
            
            colors = {}  # None -> white, Black -> Done, Grey -> Processing 
            for i in range(numCourses):
                # process each forest seperately
                # print("DFS ON", i)
                if(colors.get(i) is None and has_cycle_directed(i, g, colors)):
                    print("COLORS ARE, ", colors)
                    return False     
            return True

    METHOD 2 BFS + TOPOSORT WITH INORDER OUTORDER:
        def canFinish(self, numCourses, prerequisites):

            g = defaultdict(set)
            inorder_count = defaultdict(int)    
            # Init
            for c in range(numCourses):
                inorder_count[c] = 0
                
            for req in prerequisites:
                g[req[1]].add(req[0])
                inorder_count[req[0]] += 1
            
            print("inorder count")
            
            root_nodes = [k for (k,v) in  inorder_count.items() if v == 0]
            print("root nodes", root_nodes)
            
            print("G", g)
            print("Inorder count", inorder_count)
            
            # this is a bfs? do you need to do bfs with inorder stuff for topo to work?.. 
            # i dont think it matters..
            d = deque(root_nodes)
            visited = set()
            while d:
                node = d.popleft()
                
                visited.add(node)
                
                children = g[node]
                for c in children:
                    inorder_count[c] -= 1
                    if(inorder_count[c] == 0):
                        d.append(c)
                               
            # If you cant visit all nodes from root nodes, then there is a cycle 
            # in directed graph.      
            return len(visited) == numCourses


2.59) Cycle finding in undirected graph: 

        def undirected_has_cycle(G):
            color = {v: WHITE for v in G}
            cycle = False

            def visit(u, p):
                nonlocal cycle
                if cycle:
                    return

                color[u] = GREY
                for v in G[u]:
                    if color[v] == WHITE:
                        visit(v, u)
                    elif v != p and color[v] == GREY:
                        cycle = True
                color[u] = BLACK

            for s in G:
                if color[s] == WHITE:
                    visit(s, None)
                    if cycle:
                        return True

            return cycle



2.6) LRU Cache learnings and techniques=>
    Circular Doubly linked lists are better than doubly linked lists if you set up dummy nodes
    so you dont have to deal with edge cases regarding changing front and back pointers
    -> With doubly linked lists and maps, You can remove any node in O(1) time as well as append to front and back in O(1) time 
       which enables alot of efficiency

    -> You can also use just an ordered map for this question to solve it fast!! 
       (pop items and put them back in to bring them to the front technique to do LRU)

    from collections import OrderedDict
    class LRUCache:

        def __init__(self, capacity: int):
            self.max_capacity = capacity
            self.lru_cache = OrderedDict()
            
        def get(self, key: int) -> int:
            key = str(key)
            if(key not in self.lru_cache):
                return -1
            value = self.lru_cache[key]
            del self.lru_cache[key]
            self.lru_cache[key] = value
            return value

        def put(self, key: int, value: int) -> None:
            key = str(key)
            if(key not in self.lru_cache):
                if(len(self.lru_cache) < self.max_capacity):
                    self.lru_cache[key] = value
                else:
                    # last=False signals you want to delete first instead
                    # of last entry. 
                    self.lru_cache.popitem(last=False)
                    self.lru_cache[key] = value
            else:
                del self.lru_cache[key]
                self.lru_cache[key] = value

    # WITH A DOUBLY LINKED CIRCULAR LIST:
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.dic = dict()
            self.head = Node(0, 0)
            self.tail = Node(0, 0)
            self.head.next = self.tail
            self.tail.prev = self.head

        def get(self, key):
            if key in self.dic:
                n = self.dic[key]
                self._remove(n)
                self._add(n)
                return n.val
            return -1

        def set(self, key, value):
            if key in self.dic:
                self._remove(self.dic[key])
            n = Node(key, value)
            self._add(n)
            self.dic[key] = n
            if len(self.dic) > self.capacity:
                n = self.head.next
                self._remove(n)
                del self.dic[n.key]

        def _remove(self, node):
            p = node.prev
            n = node.next
            p.next = n
            n.prev = p

        def _add(self, node):
            p = self.tail.prev
            p.next = node
            self.tail.prev = node
            node.prev = p
            node.next = self.tail



2.7) When you DFS/BACKTRACK, one way to reduce space usage, is using grid itself
     as the visited set, and assigning and reverting it.  
     Additionally, RETURN ASAP. PRUNE, PRUNE PRUNE. 
     Do not aggregrate all the results then return.
     NO UNNECESSARY SEARCHING. Look at Word Search in leet folder. 

    Given a 2D board and a word, find if the word exists in the grid.

    class Solution:
        def exist(self, board: List[List[str]], word: str) -> bool:
        
            def CheckLetter(row, col, cur_word):
            #only the last letter remains
                if len(cur_word) == 1:
                    return self.Board[row][col] == cur_word[0]
                else:
                #mark the cur pos as explored -- None so that other can move here
                    self.Board[row][col] = None
                    if row+1<self.max_row and self.Board[row+1][col] == cur_word[1]:
                        if CheckLetter(row+1, col, cur_word[1:]):
                            return True
                    if row-1>=0 and self.Board[row-1][col] == cur_word[1]:
                        if CheckLetter(row-1, col, cur_word[1:]):
                            return True
                    if col+1<self.max_col and self.Board[row][col+1] == cur_word[1]:
                        if CheckLetter(row, col+1, cur_word[1:]):
                            return True
                    if col-1>=0 and self.Board[row][col-1] == cur_word[1]:
                        if CheckLetter(row, col-1, cur_word[1:]):
                            return True
                    #revert changes made
                    self.Board[row][col] = cur_word[0]
                    return False                  
        
            self.Board = board
            self.max_row = len(board)
            self.max_col = len(board[0])
            if len(word)>self.max_row*self.max_col:
                return False
            for i in range(self.max_row):
                for j in range(self.max_col):
                    if self.Board[i][j] == word[0]:
                        if CheckLetter(i, j, word):return True
            return False


2.8) ROLLING HASH USAGE: 
    Consider the string abcd and we have to find the hash values of 
    substrings of this string having length 3 ,i.e., abc and bcd.
    
    For simplicity let us take 5 as the base but in actual scenarios we should mod it 
    with a large prime number to avoid overflow.The highest 
    power of base is calculated as (len-1) where len is length of substring.

    H(abc) => a*(5^2) + b*(5^1) + c*(5^0) 
    = 97*25 + 98*5 + 99*1 = 3014

    H(bcd) => b*(5^2) + c*(5^1) + d*(5^0) 
    = 98*25 + 99*5 + 100*1 = 3045
    
    So, we do not need to rehash the string again. Instead, we can subtract 
    the hash code corresponding to the first character from 
    the first hash value,multiply the result by the considered 
    prime number and add the hash code corresponding to the next character to it.
    
    H(bcd)=(H(abc)-a*(5^2))*5 + d*(5^0)=(3014-97*25)*5 + 100*1 = 3045

    In general,the hash H can be defined as:-

    H=( c1a_{k-1} + c2a_{k-2} + c3a_{k-3}. . . . + cka_0 ) % m
    
    where a is a constant, c1,c2, ... ck are the input characters 
    and m is a large prime number, since the probability of 
    two random strings colliding is about ≈ 1/m.

    Then, the hash value of next substring,Hnxt using rolling hash can be defined as:-

    Hnxt=( ( H - c1ak-1 ) * a + ck+1a0 ) % m

    // computes the hash value of the input string s
    long long compute_hash(string s) {
        const int p = 31;   // base 
        const int m = 1e9 + 9; // large prime number
        long long hash_value = 0;
        long long p_pow = 1;
        for (char c : s) {
            hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
            p_pow = (p_pow * p) % m;  
        }
        return hash_value;
    }
    // finds the hash value of next substring given nxt as the ending character 
    // and the previous substring prev 
    long long rolling_hash(string prev,char nxt)
    {
        const int p = 31;
        const int m = 1e9 + 9;
        long long H=compute_hash(prev);
        long long Hnxt=( ( H - pow(prev[0],prev.length()-1) ) * p + (int)nxt ) % m;
        return Hnxt;
    }

    The various applications of Rolling Hash algorithm are:

    Rabin-Karp algorithm for pattern matching in a string in O(n) time
    Calculating the number of different substrings of a string in O(n2logn)
    Calculating the number of palindromic substrings in a str

    EXAMPLE:

    187. Repeated DNA Sequences
    Write a function to find all the 10-letter-long sequences 
    (substrings) that occur more than once in a DNA molecule.

    Example:

    Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"

    Output: ["AAAAACCCCC", "CCCCCAAAAA"]

    def findRepeatedDnaSequences(self, s):
        # We can use rolling hash to effeciently 
        # compute hash values of the pattern
        # hsh as counter for hash value => string
        hsh = {}
        def cal_hsh(string):
            assert(len(string) == 10)
            hsh_val = 0
            for i in range(10):
                hsh_val += ord(string[i]) * 7**i
            return hsh_val    
        def update_hsh(prev_val, drop_idx, add_idx):
            return (prev_val - ord(s[drop_idx]))//7 + ord(s[add_idx]) * 7 ** 9
        
        n = len(s)
        if n < 10: return []
        hsh_val = cal_hsh(s[:10])
        hsh[hsh_val] = s[:10]
        ret = set()
        # Notice this is n-9 since we want the last substring of length 10
        for i in range(1, n-9):
            hsh_val = update_hsh(hsh_val, i-1, i+9)
            if hsh_val in hsh:
                ret.add(s[i:i+10])
            else:
                hsh[hsh_val] = s[i:i+10]
        return list(ret)

    
3) 0-1 BFS
    we can use BFS to solve the SSSP (single-source shortest path) 
    problem in O(|E|), if the weights of each edge is either 0 or 1.

    append new vertices at the beginning if the corresponding edge 
    has weight 0, i.e. if d[u] = d[v], or at the end if the edge 
    has weight 1, i.e. if d[u]=d[v]+1. This way the queue still remains sorted at all time.

    vector<int> d(n, INF);
    d[s] = 0;
    deque<int> q;
    q.push_front(s);
    while (!q.empty()) {
        int v = q.front();
        q.pop_front();
        for (auto edge : adj[v]) {
            int u = edge.first;
            int w = edge.second;
            if (d[v] + w < d[u]) {
                d[u] = d[v] + w;
                if (w == 1)
                    q.push_back(u);
                else
                    q.push_front(u);
            }
        }
    }

    Dial's algorithm
    We can extend this even further if we allow the weights of the edges to be even bigger. 
    If every edge in the graph has a weight ≤k, than the distances of vertices 
    in the queue will differ by at most k from the distance of v to the source. 
    So we can keep k+1 buckets for the vertices in the queue, and
    whenever the bucket corresponding to the smallest distance gets 
    empty, we make a cyclic shift to get the bucket with the next higher 
    distance. This extension is called Dial's algorithm.



4) To do things inplace, such as inplace quick sort, or even binary search, 
    it is best to operate on index values in your recursion 
    instead of slicing and joining arrays.
    Always operate on the pointers for efficiency purposes.

5) If you need to keep track of a list of values instead of just 1 values 
    such as a list of maxes, instead of 1 max, 
    and pair off them off, use an ordered dictionary! 
    They will keep these values ordered for pairing purposes. 
    pushing and poping an element in an ordered map brings it to the front. 

6)  If you need to do range searches, you need a range tree. 
    if you dont have time to get a range tree, or a kd tree 
    use binary searching as the substitute!


7) if the problem is unsorted, try sorting and if you need to keeping 
    track of indexes, use reverse index map, to do computations. 

7.5) If the problem is already sorted, try binary search. 

8) Do preprocessing work before you start solving problem to improve efficiency

9) Use Counter in python to create a multiset. 

9.1) Basic Calculator 3 Review:

    The expression string contains only non-negative integers, +, -, *, / operators, 
    open ( and closing parentheses ) and empty spaces.


        class Solution(object):

            def calculate(self, s):
                """
                Time    O(n)
                Space   O(n)
                80 ms, faster than 22.22%
                """
                arr = []
                for c in s:
                    arr.append(c)
                return self.helper(arr)

            def helper(self, s):
                if len(s) == 0:
                    return 0
                stack = []
                sign = '+'
                num = 0
                while len(s) > 0:
                    c = s.pop(0)
                    if c.isdigit():
                        num = num*10+int(c)
                    if c == '(':
                        # do recursion to calculate the sum within the next (...)
                        num = self.helper(s)
                    if len(s) == 0 or (c == '+' or c == '-' or c == '*' or c == '/' or c == ')'):
                        if sign == '+':
                            stack.append(num)
                        elif sign == '-':
                            stack.append(-num)
                        elif sign == '*':
                            stack[-1] = stack[-1]*num
                        elif sign == '/':
                            stack[-1] = int(stack[-1]/float(num))
                        sign = c
                        num = 0
                        if sign == ')':
                            break
                return sum(stack)


10) Use dynamic programming for optimal substructure, subsequence questions
    -> Top down is easier to reason because you just memoize solutions youve seen before. 
    -> Check for optimal substructure ( A given problems has Optimal Substructure Property 
                                       if optimal solution of the given problem can be obtained 
                                       by using optimal solutions of its subproblems.)
    
    -> Check of overlapping solutions. Make problems overlap by processing it in a certian way!!! Look at directionality!!! 
        (DP cant help binary search because no overlapping)

    -> TO DO IT: Define Subproblems. Dynamic programming algorithms usually involve a recurrence involving
            some quantity OPT(k₁, …, kₙ) over one or more variables (usually, these variables
            represent the size of the problem along some dimension). Define what this quantity represents
            and what the parameters mean. This might take the form “OPT(k) is the maximum
            number of people that can be covered by the first k cell towers” or “OPT(u, v, i) is the
            length of the shortest path from u to v of length at most i.”
            • Write a Recurrence. Now that you've defined your subproblems, you will need to write
            out a recurrence relation that defines OPT(k₁, …, kₙ) in terms of some number of subproblems.
            Make sure that when you do this you include your base cases.

    -> The other key property is that there
            should be only a polynomial number of different subproblems. These two properties together allow
            us to build the optimal solution to the final problem from optimal solutions to subproblems.
            In the top-down view of dynamic programming, the first property above corresponds to being
            able to write down a recursive procedure for the problem we want to solve. The second property
            corresponds to making sure that this recursive procedure makes only a polynomial number of
            different recursive calls. In particular, one can often notice this second property by examining
            the arguments to the recursive procedure: e.g., if there are only two integer arguments that range
            between 1 and n, then there can be at most n^2 different recursive calls.
            Sometimes you need to do a little work on the problem to get the optimal-subproblem-solution
            property. For instance, suppose we are trying to find paths between locations in a city, and some
            intersections have no-left-turn rules (this is particulatly bad in San Francisco). Then, just because
            the fastest way from A to B goes through intersection C, it doesn’t necessarily use the fastest way
            to C because you might need to be coming into C in the correct direction. In fact, the right way
            to model that problem as a graph is not to have one node per intersection, but rather to have one
            node per <Intersection, direction> pair. That way you recover the property you need.


10.5) DP Construction:
        DP problems typically show up at optimization or counting problems 
        (or have an optimization/counting component). Look for words like 
        "number of ways", "minimum", "maximum", "shortest", "longest", etc.

        Start by writing your inputs. Identify which inputs are variable and which are constant.

        Now write your output. You output will be whatever you are optimizing or 
        counting. Because of this, the output might not match exactly what you
        are solving for (if counting / optimizing is only a component of the problem).

        Write a recurrence using your output as the function, your inputs
        as inputs to the function, and recursive calls in the function body. 
        The recursive calls will represent the "choices" that you can make, 
        so that means you'll have one recursive call per "choice". (You are usually optimizing 
        over choices are counting different types of choices). Think of ways to split up 
        your input space into smaller components. The type of input will dictate how this 
        might look. Array/string inputs usually peel one or two elements from the front 
        or back of the array and recurse on the rest of the array. Binary tree inputs 
        usually peel the root off and recurse on the two subtrees. Matrix inputs 
        usually peel an element off and recurse in both directions (up or down and right or left).

        Come up with base case(s) for the recurrence. When you make the recursive calls, 
        you decrease the problem size by a certain amount, x. You will probably need about x base cases.

        Write your code. I recommend top-down for interviews since you 
        don't have to worry about solving subproblems in the right order. 
        Your cache will have one dimension per non-constant input.

        After writing code, think about whether bottom-up is possible 
        (can you come up with an ordering of subproblems where smaller 
        subproblems are visited before larger subproblems?). If so, you 
        can decide whether it is possible to reduce space complexity by 
        discarding old answers to subproblems. If it's possible to reduce 
        space, mention it in the interview (and explain). You probably won't 
        have to code it. If you have time, feel free to code it bottom-up.

10.7) Memorize 0-1 Knapsack and strategy
      and space efficiency strategy:
    
    -> LEARN HOW TO USE A PARENTS {} MAP TO REVISIT THE OPTIMIZED DP STATES!
        -> And return the optimized solution
    -> Learn how to seperate concerns when creating DP GRIDS. 
    -> LEARN how to space optimize with PREV/NEXT Rolling optimizer,
        -> Learn how to also JUST have PREV without Next space optimization
           by being smart about how you iterate over data

    You have a weight W and you want to put items with wt and value to maximize total value!
    ...

        I think recurrence looks like this: process each item, and weights..
        Fill in lower weights first, then fille higher onces. Weight 0 = Value 0 base case. 
        OPT[W, i] = max { OPT[W - wt, i-1] + val, 
                        OPT[W, i-1] 
                        }

        return OPT[W, N] 
        (we are 1 index ahead here if you think about it, same in python code below, goes to n+1, and W+1 instead of n and W, 
        we are 1 index ahead by subtracting 1 for val[i-1], this way best soln is K[N][W])

        Also the reason W is inner loop and item is outer loop in code below is if you think about it, we neeed to know 
        all the weighhts for current item, before we can go to next item, to successuly process next item..

    # n is number of items. 
    def knapSack(W, wt, val, n): 
        K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
    
        # Build table K[][] in bottom up manner 
        for i in range(n + 1): 
            for w in range(W + 1): 
                if i == 0 or w == 0: 
                    K[i][w] = 0
                elif wt[i-1] <= w: 
                    K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]) 
                else: 
                    K[i][w] = K[i-1][w] 
    
        return K[n][W] 

    # SPACE EFFICIENT
    You can reduce the 2d array to a 1d array saving the values for the current iteration. 
    For this to work, we have to iterate capacity (inner for-loop) in the 
    opposite direction so we that we don't use the values that 
    were updated in the same iteration 
    
    from collections import namedtuple

    def knapsack(capacity, items):
        # A DP array for the best-value that could be achieved for each weight.
        best_value = [0] * (capacity + 1)
        # The previous item used to achieve the best-value for each weight.
        previous_item = [None] * (capacity + 1)
        for item in items:
            for w in range(capacity, item.weight - 1, -1):
                value = best_value[w - item.weight] + item.value
                if value > best_value[w]:
                    best_value[w] = value
                    previous_item[w] = item  # Always can save stuff like item chosen, when updates are made. 

        cur_weight = capacity
        taken = []
        while cur_weight > 0:
            taken.append(previous_item[cur_weight])
            cur_weight -= previous_item[cur_weight].weight

        return best_value[capacity], taken


1)  Know how to write BFS with a deque, and DFS explicitely with a list. 
    Keep tracking of function arguments in tuple for list. 

2)  If you need a priority queue, use heapq. Need it for djikistras. 
    Djikstras is general BFS for graphs with different sized edges. 

12.5) Know expand around center to find/count palindroms in a string:
    
    Count all even and odd palindromes in a string


    class Solution(object):
        def countSubstrings(self, s):
            # Find all even and odd substrings. 
            '''
            THis is also known as the expand around center solution.        
            '''
            
            i = 0
            count = 0
            for i in range(len(s)):            
                left = i
                right = i
                
                # Count odd palins
                def extend(left, right, s):
                    count = 0
                    while True:
                        if left < 0 or right >= len(s) or s[left] != s[right]:
                            break   
                        count += 1
                        left = left - 1
                        right = right + 1
                    return count
                
                count += extend(left, right, s)
                count += extend(left, right+1, s)
            
            return count
        

13) TOPO SORT -> if you dont know which node to start this from, start from any node.
                    topo sort will still figure out the sorting. keep visited set. 
                    
                    result = deque()
                    def topo_sort(node, visited):

                        visited.add(node)
                        children = g[node]
                        
                        for c in children:
                            if(c in visited):
                                continue

                            topo_sort(c, visited)
                        result.appendleft(node)
                    
                    for node in g.keys():
                        if node not in visited:
                            topo_sort(node, visited)



14) Use stacks/queues to take advantage of push/pop structure in problems 
    such as parentheses problems. Or valid expression problems.

15) When you have a problem and it gives you a Binary search tree, 
    make sure to exploit that structure and not treat it as a normal
    binary tree!!!


16) Diviide and conquer

17) Greedy algorithms => requires smart way of thinking about things

18) Bit magic -> Entire section done in math notes.

19) Dynamic programming with bitmasking

20) GRIDS THOUGHT PROCESS AND DP:
    When you do DP bottom up think of GRIDS. How many dimensions are your grid?
    How would you fill a grid if you were to do it by pencil? Which direction?
    Think about top down case -> HOW DOES THE MAP FILL UP. OK HOW SHOULD the GRID FILL UP
    Whats the base cases in the GRID? 
    Whats the recurrence relation?
    What locations need to be filled in GRID before other spots. 

21) 4. Merge Intervals
        The Merge Intervals pattern is an efficient technique to deal with 
        overlapping intervals. In a lot of problems involving intervals, you 
        either need to find overlapping intervals or merge intervals if they overlap. 
        The pattern works like this:
        Given two intervals (‘a’ and ‘b’), there will be six different ways the 
        two intervals can relate to each other:
         => a consumes b, b consumes a, b after a, a after b, b after a no overlap, a after b no overlap

        How do you identify when to use the Merge Intervals pattern?
            If you’re asked to produce a list with only mutually exclusive intervals
            If you hear the term “overlapping intervals”.
        Merge interval problem patterns:
            Intervals Intersection (medium)
            Maximum CPU Load (hard)


22) Cyclic Sort: 
        This pattern describes an interesting 
        approach to deal with problems involving arrays containing
        numbers in a given range. The Cyclic Sort pattern iterates over the array 
        one number at a time, and if the current number you are iterating is not 
        at the correct index, you swap it with the number at its correct index. You could 
        try placing the number in its correct index, but this will produce a complexity 
        of O(n^2) which is not optimal, hence the Cyclic Sort pattern.

        How do I identify this pattern?
        They will be problems involving a sorted array with numbers in a given range
        If the problem asks you to find the: 
        -> missing/duplicate/smallest number in an sorted/rotated array
           
        We one by one consider all cycles. We first consider the cycle that 
        includes first element. We find correct position of first element, 
        place it at its correct position, say j. We consider old value of arr[j] 
        and find its correct position, we keep doing this till all elements of current 
        cycle are placed at correct position, i.e., we don’t come back to cycle starting point.
    
        Problems featuring cyclic sort pattern:
        Find the Missing Number (easy)
        Find the Smallest Missing Positive Number (medium)

22.5) CYCLIC SORT EXAMPLE:

    class Solution(object):
        def rotate(self, nums, k):
            
            start = 0
            val = nums[start]
            
            i = start
            N = len(nums)
            swaps = 0
            
            while True:
                pivot = i + k
                pivot %= N
                
                temp = nums[pivot]
                nums[pivot] = val
                val = temp
                i = pivot
                
                swaps += 1
                if(swaps == N):
                    return 
                if pivot == start:
                    i = start + 1             
                    val = nums[start + 1]
                    start += 1        
            return nums
            
    Another Solution:
    def solution(A, K):
        N = len(A)
        
        if N == 0:
            return []
        
        i = 0
        nxtVal = A[0]
        swaps = 0
        lastLoc = 0

        while True:
            A[(i + K) % N], nxtVal = nxtVal, A[(i + K) % N]
            i = (i + K) % N
            
            swaps += 1
            if swaps == N:
                break 
            if i == lastLoc:
                lastLoc += 1
                i += 1
                nxtVal = A[i]
                # did we do all swaps?
        return A




22.6) Know in-place reverse linked list (MEMORIZE)
        # Function to reverse the linked list 
            def reverse(self): 
                prev = None
                current = self.head 
                while(current is not None): 
                    next = current.next
                    current.next = prev 
                    prev = current 
                    current = next
                self.head = prev 
        
        Reverse a Sub-list (medium)
        Reverse every K-element Sub-list (medium)

+22.61) Math & HashMap counting:
+    You are given n points in the plane that are all distinct,
+     where points[i] = [xi, yi]. A boomerang is a tuple of points (i, j, k) 
+     such that the distance between i and j equals the distance between 
+     i and k (the order of the tuple matters).
+    Return the number of boomerangs.
+
+ 
+
+    Example 1:
+
+    Input: points = [[0,0],[1,0],[2,0]]
+    Output: 2
+    Explanation: The two boomerangs are [[1,0],[0,0],[2,0]] 
+                 and [[1,0],[2,0],[0,0]].
+
+    Solution
+    for each point, create a hashmap and count all points with same distance. If for a point p, there are k points with distance d, number of boomerangs corresponding to that are k*(k-1). Keep adding these to get the final result.
+
+        res = 0
+        for p in points:
+            cmap = {}
+            for q in points:
+                f = p[0]-q[0]
+                s = p[1]-q[1]
+                cmap[f*f + s*s] = 1 + cmap.get(f*f + s*s, 0)
+            for k in cmap:
+                res += cmap[k] * (cmap[k] -1)
+        return res
+
+
+22.62) Question that doesnt look like DP but is
+    1048. Longest String Chain
+
+    You are given an array of words where each word consists of 
+    lowercase English letters.
+
+    wordA is a predecessor of wordB if and only if we can insert 
+    exactly one letter anywhere in wordA without changing the order 
+    of the other characters to make it equal to wordB.
+
+    For example, "abc" is a predecessor of "abac", while "cba" 
+    is not a predecessor of "bcad".
+
+    A word chain is a sequence of words [word1, word2, ..., wordk]
+    with k >= 1, where word1 is a predecessor of word2, word2 is a 
+    predecessor of word3, and so on. A single word is 
+    trivially a word chain with k == 1.
+
+    Return the length of the longest possible word chain with 
+    words chosen from the given list of words.
+
+    Input: words = ["a","b","ba","bca","bda","bdca"]
+    Output: 4
+    Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
+
+    Soln)
+    Sort the words by word's length. (also can apply bucket sort)
+    For each word, loop on all possible previous word with 1 letter missing.
+    If we have seen this previous word, update the longest chain for the current word.
+    Finally return the longest word chain.
+
+    Succint:
+    def longestStrChain(self, words):
+        dp = {}
+        for w in sorted(words, key=len):
+            dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in xrange(len(w)))
+        return max(dp.values())
+
+    class Solution:
+        def longestStrChain(self, words: List[str]) -> int:
+            dp = {}
+            result = 1
+
+            for word in sorted(words, key=len):
+                dp[word] = 1
+
+                for i in range(len(word)):
+                    prev = word[:i] + word[i + 1:]
+
+                    if prev in dp:
+                        dp[word] = max(dp[prev] + 1, dp[word])
+                        result = max(result, dp[word])
+
+            return result
+    
+    3 other solns TopDown/bottomup/LIS
+
+    LIS idea:
+    class Solution:
+        def longestStrChain(self, words: List[str]) -> int:
+            def isPredecessor(word1, word2):
+                if len(word1) + 1 != len(word2): return False
+                i = 0
+                for c in word2:
+                    if i == len(word1): return True
+                    if word1[i] == c:
+                        i += 1
+                return i == len(word1)
+            
+            words.sort(key=len)
+            n = len(words)
+            dp = [1] * n
+            ans = 1
+            for i in range(1, n):
+                for j in range(i):
+                    if isPredecessor(words[j], words[i]) and dp[i] < dp[j] + 1:
+                        dp[i] = dp[j] + 1
+                ans = max(ans, dp[i])
+            return ans
+
+
+    TOPDOWN (SEEMS to be fastest)
+
+    Let dp(word) be the length of the longest possible word chain end at word word.
+    To calculate dp(word), we try all predecessors of word
+    word and get the maximum length among them.
+
+    class Solution:
+        def longestStrChain(self, words: List[str]) -> int:
+            wordSet = set(words)
+
+            @lru_cache(None)
+            def dp(word):
+                ans = 1
+                for i in range(len(word)):
+                    predecessor = word[:i] + word[i + 1:]
+                    if predecessor in wordSet:
+                        ans = max(ans, dp(predecessor) + 1)
+                return ans
+
+            return max(dp(w) for w in words)
+
+
+
+22.69) Floyd's Loop detection algorithm:
+    Find cycle in linkedlist through tortoise and hare pointers.
+    If they meet again, there is a loop in the list.
+
+
+    For showing that they eventually must meet, 
+    consider the first step at which the tortoise enters the loop. If the hare is on that node, that 
+    is a meeting and we are done. If the hare is not on that node, note that on each subsequent step the distance the hare is ahead of the 
+    tortoise increases by one, which means that since they are on a loop the d
+    istance that the hare is BEHIND the tortoise decreases by one. 
+    Hence, at some point the distance the hare is behind the tortoise becomes zero and the meet
+
+    More detailed proof (modulus):
+
+    If the preliminary tail is length T and the cycle is length C (so in your picture, T=3, C=6), 
+    we can label the tail nodes (starting at the one farthest from the cycle) 
+    as −T,−(T−1),...,−1 and the cycle nodes 0,1,2,...,C−1 (with the cycle node 
+    numbering oriented in the direction of travel).
+
+    We may use the division algorithm to write T=kC+r where 0≤r<C.
+
+    After T clicks the tortoise is at node 0 and the hare is at node r (since hare has gone 2T 
+    steps, of which the first T were in the tail, leaving T steps in the cycle, and T≡r(modC)).
+
+    Assuming r≠0, after an additional C−r clicks, the tortoise is at node C−r; and the hare is at 
+    node congruent (modC) to r+2(C−r)=2C−r≡C−r(modC). Hence both critters are at node C−r. 
+    [In the r=0 case, you can check that the animals meet at the node 0.]
+
+    The distance from the start at this meeting time is thus T+C−r=(kC+r)+C−r=(k+1)C, a multiple of the cycle length, 
+    as desired. We can further note, this occurrence is at the first multiple of the cycle length that is greater than or equal to the tail length.
+



22.7) Find the start of a loop in a linked list:

       Consider the following linked list, where E is the cylce entry and X, the crossing point of fast and slow.
        H: distance from head to cycle entry E
        D: distance from E to X
        L: cycle length
                          _____
                         /     \
        head_____H______E       \
                        \       /
                         X_____/   
        
    
        If fast and slow both start at head, when fast catches slow, slow has traveled H+D and fast 2(H+D). 
        Assume fast has traveled n loops in the cycle, we have:
        2H + 2D = H + D + L  -->  H + D = nL  --> H = nL - D
        Thus if two pointers start from head and X, respectively, one first reaches E, the other also reaches E. 
        In my solution, since fast starts at head.next, we need to move slow one step forward in the beginning of part 2


        Intuition
        To resolve the problem of finding out the cycle’s starting point, 
        we can use the two-pointer technique, which is efficient and doesn't require extra memory for storage.

        The intuition behind this algorithm involves a faster runner (the fast pointer) 
        and a slower runner (the slow pointer), both starting at the head of the linked list. 
        The fast pointer moves two steps at a time while the slow pointer moves only one. 
        If a cycle exists, the fast pointer will eventually lap the slow pointer 
        within the cycle, indicating that a cycle is present.

        Once they meet, we can find the start of the cycle. To do this, we set up another pointer, 
        called ans, at the head of the list and move it at the same pace as the slow pointer. 
        The place where ans and the slow pointer meet again will be the starting node of the cycle.

        Why does this work? If we consider that the distance from the list head to the cycle 
        entrance is x, and the distance from the entrance to the meeting point is y, with the 
        remaining distance back to the entrance being z, we can make an equation. Since the fast 
        pointer travels the distance of x + y + n * (y + z) (where n is the number of laps made) 
        and slow travels x + y, and fast is twice as fast as slow, then we can deduce that x = n * (y + z) - y, -> x = nl - d
        which simplifies to x = (n - 1) * (y + z) + z. This shows that starting a pointer at the 
        head (x distance to the entrance) and one at the meeting point (z distance to the entrance) 
        and moving them at the same speed will cause them to meet exactly at the entrance of the cycle.


        Algorithm
        Use two references slow, fast, initialized to the head
        Increment slow and fast until they meet
        fast is incremented twice as fast as slow
        If fast.next is None, we do not have a circular list
        When slow and fast meet, move slow to the head
        Increment slow and fast one node at a time until they meet
        Where they meet is the start of the loop

        class MyLinkedList(LinkedList):

            def find_loop_start(self):
                if self.head is None or self.head.next is None:
                    return None
                slow = self.head
                fast = self.head
                while fast.next is not None:
                    slow = slow.next
                    fast = fast.next.next
                    if fast is None:
                        return None
                    if slow == fast:
                        break
                slow = self.head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                    if fast is None:
                        return None
                return slow





22.8) Find kth to last element of linked list

        Algorithm
        Setup two pointers, fast and slow
        Give fast a headstart, incrementing it once if k = 1, twice if k = 2, ...
        Increment both pointers until fast reaches the end
        Return the value of slow


1)  TREE DFS:
    Decide whether to process the current node now (pre-order), 
    or between processing two children (in-order) 
    or after processing both children (post-order).
    Make two recursive calls for both the children 
    of the current node to process them.
    -> You can also just use Tree DFS to process in bfs order

23.5) 2 STACKS == QUEUE TECHNIQUE!
      push onto one, if you pop and empty, 
      dump first one into second one!

      push into one stack, pop from the other stack. 


24) TWO HEAPS TECHNIQUE!!!

        In many problems, we are given a set of elements such that we can divide them
        into two parts. To solve the problem, we are interested in knowing 
        the smallest element in one part and the biggest element in the other 
        part. This pattern is an efficient approach to solve such problems.

        This pattern uses two heaps; A Min Heap to find the smallest element 
        and a Max Heap to find the biggest element. The pattern works by 
        storing the first half of numbers in a Max Heap, this is because 
        you want to find the largest number in the first half. You then 
        store the second half of numbers in a Min Heap, as you want to 
        find the smallest number in the second half. At any time, the 
        median of the current list of numbers can be calculated from 
        the top element of the two heaps.

        As youre pushing elements maintain the invariant, half the numbers in max heap half in min heap

        and max heap contains the smallest numbers, and min heap contains the largest numbers. 



        Ways to identify the Two Heaps pattern:
        Useful in situations like Priority Queue, Scheduling
        If the problem states that you need to find the 
        -> smallest/largest/median elements of a set
        Sometimes, useful in problems featuring a binary tree data structure
        Problems featuring
        -> Find the Median of a Number Stream (medium)


25) BFS CREATION OF SUBSETS! (instead of dfs choose/dont choose strat. 
    Can this strat also be used to do TOP DOWN DP with BFS ?)

    A huge number of coding interview problems involve 
    dealing with Permutations and Combinations of a 
    given set of elements. The pattern Subsets describes 
    an efficient Breadth First Search (BFS) approach to 
    handle all these problems.

    The pattern looks like this:
    Given a set of [1, 5, 3]
    Start with an empty set: [[]]
    Add the first number (1) to all the existing subsets to create new subsets: [[], [1]];
    Add the second number (5) to all the existing subsets: [[], [1], [5], [1,5]];
    Add the third number (3) to all the existing subsets: [[], [1], [5], [1,5], [3], [1,3], [5,3], [1,5,3]].

    Problems featuring Subsets pattern:
    Subsets With Duplicates (easy)
    String Permutations by changing case (medium)


+25.3) Given an array of numbers and an array of integer pairs, [i,j], increment the array of 
        numbers by one for each index between interval [i,j]
+        e.g.
+        a = [1,2,3,4]
+        intervals = [[1,3]]
+        output [1,3,4,5]
+        You're incrementing everything from index 1 to index 3.
+
+        a = [1,2,3,4]
+        intervals = [[1,3],[2,3]]
+        output [1,3,5,6]
+        You're incrementing everything from index 1 to index 3 then index 2 to index 3.

        You can get O(N + M). Keep an extra increment array B the same size of A initially empty (filled with 0). 
        If you need to increment the range (i, j) with value k then do B[i] += k and B[j + 1] -= k

        Now do a partial sum transformation in B, considering you're indexing from 0:

        for (int i = 1; i < N; ++i) B[i] += B[i - 1];
        And now the final values of A are A[i] + B[i]




+
+
+25.31) How does mod work on negatives?? whats binary/ternary representation of negative numbers?
+        
+        Python modulo operator always return the remainder having the same sign as the divisor. 
+        This can lead to some confusion with the output.
+
+        >>> -5 % 3
+        1
+        >>> 5 % -3
+        -1
+        >>> -10 % 3
+        2
+        >>> 
+
+        -5 % 3 = -5 + 3 + 3 % 3 == 1
+        
+        -5 % 3 = (1 -2*3) % 3 = 1
+        5 % -3 = (-1 * -2*-3) % 3 = -1
+        -10 % 3 = (2 -4*3) % 3 = 2
+
+
+        If, according to 2's Complement any binary string can be converted to it's negative counterpart by flipping each digit to it's 
+        opposite number (eg. 10012→01102) and then add 1, then how would you implement a "3's Complement"? So to speak.
+
+        Like, lets say I have 29, and want to write it as −29 in binary.
+        First I convert it into binary: 11101. And now to make it negative, we use 2's Complement. 
+        We change the binary number to 00010 and add 1→00011.
+
+        Now we have −29 in binary form.
+
+        But how to do for ternary?
+


+25.4) Covert Decimal to binary/ternary:
+
+        Similar to converting decimal to binary/or ternary?
+            -> 
+                i mean you can just find leftmost set bits...
+                how about ternary?
+                
+                find largest 3^x that can divide number. 
+                number // 3 -> remaineder
+                number = 3*q + r
+                divide by 3 -> remainder. 
+
+                10 in tern
+
+                1*3^2 + 0*3^1 + 1*3^0 -> 10 (101)
+                10 = 3*3 + 1
+
+                YOU ARE OVER COMPLICATING IT!
+                FIND THE least significant SET BIT first, then find the bigger bits!
+
+                Steps to Convert Decimal to Ternary: 
+
+                Divide the number by 3.
+                Get the integer quotient for the next iteration.
+                Get the remainder for the ternary digit.
+                Repeat the steps until the quotient is equal to 0.
+
+                How do you represent -3 decimal? how does mod work for negatives? (Research it and post it here)
+
+                    def convertToTernary(N):
+                        
+                        # Base case
+                        if (N == 0):
+                            return;
+                    
+                        # Finding the remainder
+                        # when N is divided by 3
+                        x = N % 3;
+                        N //= 3;
+                        if (x < 0):
+                            N += 1;
+                    
+                        # Recursive function to
+                        # call the function for
+                        # the integer division
+                        # of the value N/3
+                        convertToTernary(N);
+                    
+                        # Handling the negative cases
+                        if (x < 0):
+                            print(x + (3 * -1), end = "");
+                        else:
+                            print(x, end = "");
+                    
+                    
+                    # Function to convert the
+                    # decimal to ternary
+                    def convert(Decimal):
+                        
+                        print("Ternary number of ", Decimal,
+                            " is: ", end = "");
+                    
+                        # If the number is greater
+                        # than 0, compute the
+                        # ternary representation
+                        # of the number
+                        if (Decimal != 0):
+                            convertToTernary(Decimal);
+                        else:
+                            print("0", end = "");
+
+25.5) Counting and Similar to Next Permutation: Number plate
+
+        The number on the number plate of a vehicle has alphanumeric characters. The number is a string of 6 characters of 
+        which first 2 characters are alphabets while last 4 are digits. The first number generated is AA0000 while the last 
+        number generated is ZZ9999. Find the kth number generated.
+
+        Be greedy. subtract as much as possibly to get the leftmost letter, then get rest of letters. 
+
+        Subtract 26 * 9999 -> 
+        go up 10000 -> shifts the letter.
+
+        AA0000 -> AB0000
+
+        Can you divide by 10000 first, -> check the remainder. Set that. 
+        then use the rest to figure out the other 2 letters. 
+
+
+        260,004 -> Correct answer:
+        
+        BA0004
+        
+
+        -> 260,004 % 10,000 =   4
+        260,004//10000 = 26
+
+        ok now divide by 26? 
+
+        26 % 26 == 0 
+        26/26 = 1
+
+        1 % 26 = 1
+        1//26 == 0
+        done!
+
+        so we got
+
+        26^2 * 1  + 26^1 * 0 + 4*10,0000^0
+        
+        0 ->a, 1 -> B, ..., 25 -> Z
+        BA0004
+        (I think this soln works...)
+    
+
+
+25.55) Counting Plate 2:
+    Given below pattern of license plates (Pattern only, not the actual list of license plates), Find the nth license plate
+    All license plates no are of size 5 chars
+    Eg, if n is 3, ans is - 00002
+
+    00000
+    00001
+    00002
+    ........
+    ........
+    99999
+    0000A
+    0001A
+    0002A
+    ........
+    .........
+    9999A
+    0000B
+    0001B
+    0002B
+    .........
+    .........
+    9999B
+    0000C
+    ........
+    ........
+    9999Z
+    000AA
+    001AA
+    .........
+    .........
+    999AA
+    000AB
+    ..........
+    ..........
+    999ZZ
+    00AAA
+    ........
+    ........
+    ZZZZZ
+
+        Soln: Idea:
+        Example there are n = 5 charaters and find no-th license plate
+        All licenses can be listed and devided into 6 region:
+
+        Region 1	Region 2	Region 3	Region 4	Region 5	Region 6
+        00000	0000A	000AA	00AAA	0AAAAA	AAAAA
+        00001	0001A	001AA	01AAA	1AAAAA	AAAAB
+        ...	...	...	...	...	...
+        99999	9999Z	999ZZ	99ZZZ	9ZZZZZ	ZZZZZ
+        The first region has $10^5$ elements
+        The second region has $10^4*26$ elements
+        The third region has $10^3*26^2$ elements
+        The fourth region has $10^2*26^3$ elements
+        The fifth region has $10 * 26^4$ elements
+        The sixth region has $26^5$ elements\
+
+        We will find no-th license that belongs which region by compare no with the order of the first element in each region
+        After that, we find the pattern of this license.
+        Can see that each license has 2 part: left part only has number (like '123') and right part has only alphabet (like 'ABC').
+        If the no-th license belongs to Region X, left part has (5 - X + 1) numbers (call it by num0s) and (X - 1) alphabets.
+
+        Calculate the order from the first element in this region, call it distance
+        The left part is remainder of distance modulo $10^{num0s}$.
+        The right part is calculated from the quotient of distance devide $10^{num0s}$.
+
+25.57)  BFS VS Brute force:
+
+        Given a 2D grid with n rows and m columns. Some cells are blocked which you cannot pass through. From each cell you can go either up, 
+        down, left or right. Find the shortest path from (0, 0) to (n-1, m-1).
+
+        Followup:
+
+        Now assume that there exists no path from (0, 0) to (n-1, m-1) (All paths are blocked). Find the minimum number of cells that you need to 
+        unblock such that there exists a path from (0, 0) to (n-1, m-1). Can you solve it in O(n+m)
+
+        First part: BFS,
+
+        Second part: 
+
+        Brute force:
+            Try to dfs from start to other nodes. Then allow 1 crossing, and see if you make it. If you dont, try 2 crossings, until you get to N
+
+        Better:
+           dJIKSTRAS
+
+
+25.58) Cycle in linked list. One node moves 1 step at a time and the other node moves 2 steps at a time. If they meet, there is a cycle. 
+    If a pointer reaches theend of the linkedlist before the pointers are the same, then there is no cycle. 
     Actually, the pointers need not move one and two nodes at a time; it is only necessaary that 
     thepointers move at different rates. Sketch otu the proof for this. 
+
+
+25.59) Without a calculator, how many zeros are at the end of 100! (100 factorial)
+    factor out 100, factor out 10?
+    100 *90*80*70*60*50*40*30*20*10
+    
+    2*5 -> 10 
+
+    atleast 12....
+    
+    Answer:Whatyoudon'twanttodoisstartmultiplyingitallout!Thetrickis
+    rememberingthatthenumberofzerosattheendofanumberisequaltothe
+    numberoftimes"10"(or"2*5")appearswhenyoufactorthenumber.Therefore
+    thinkabouttheprimefactorizationof100!andhowmany2sand5sthereare.
+    Thereareabunchmore2sthan5s,sothenumberof5sisalsothenumberof10sin
+    thefactorization.Thereisone5foreveryfactorof5inourfactorialmultiplication
+    (1*2*...*5*...*10*...*15*...)andanextra5for25,50,75,and100.Thereforewehave
+    20+4=24zerosattheendof100!.
+    
+    Review Prime factorization
+
+25.6) Min cost to make string palindrome:
+    Given a string S and a cost matrix C.
+
+    C[i][j] denotes the cost to convert character i to character j. 
+
+    Goal is to convert the string into a palindromic string. In one operation you can choose a character of string and convert 
+    that character to any other character. You can do this operation any number of times. The cost to convert one character to another character 
+    is determined by the cost matrix. Find the minimum cost to convert a given string to a palindrome. 
+
+        The idea is to start comparing from the two ends of string. Let i be initialized as 0 index and j initialized as length – 1.
+        If characters at two indices are not same, a cost will apply. To make the cost minimum replace the character 
+        which is smaller. Then increment i by 1 and decrement j by 1. Iterate till i less than j. 
+
+        Also find the shortest cost from one character to anotehr using all pair shortest paths graph algo
+
+
+25.7) Efficient strategy for round robin?
+
+        Fair log pickups:
+        A log is defined as:
+
+        class Log {
+            String text;
+            String serverId;
+        }
+
+        You are given a list of logs and a number k. You need to pick up k logs in total from the servers. But while picking 
+        one must ensure that the pickup strategy is as fair as possible. By fair it means that it should not happen that all the logs 
+        are picked up from the same machine. Return list of log files. Implement the following method:
+
+        List<Log> optimalLogPickup(List<Log> logs, int k) {
+
+        }
+
+        Eg:
+
+
+        logs = [{"hello", "server#1"}, {"world", "server#1"}, {"Rishika Sinha", "server#2"}, {"best PM", "server#2"}], k = 2
+
+        possible output: 
+        logs = [{"hello", "server#1"}, {"Rishika Sinha", "server#2"}]
+        incorrect output:
+        logs = [{"hello", "server#1"}, {"world", "server#1"}] // Since I am picking both the logs from the same server (server#1) while being unfair to server#2.
+
+
+        Create map of server -> logs
+        
+        Use priority queue and pick one item at a time. 
+        
+        What if we want to pick more items at a time?
+
+        Determine min amount for particular server. 
+        Exhaust it, then keep track of min amount for next server.
+        Exhuast it, until you run out of servers. 
+
+        Count total logs in each server.
+        Sort it. 
+        [2,4,5,7,8] [5 servers]
+
+        Now check if k > 2*5, if it is, take 2 logs from each server. and kill off the exhausted server. 
+        Subtract 2 from every element in list and pop front. 
+
+        [2,3,5,6]
+        Repeat process, but if k <2 * 4 then just do round robin pick 1 at a time. 
+
+25.8) LAZY COMPUTING, PARTIAL SUMS,  AND Increment Intervals:
+
+    Given an array of numbers and an array of integer pairs, [i,j], increment the array of numbers by one for each index between interval [i,j]
+    e.g.
+    a = [1,2,3,4]
+    intervals = [[1,3]]
+    output [1,3,4,5]
+    You're incrementing everything from index 1 to index 3.
+
+    a = [1,2,3,4]
+    intervals = [[1,3],[2,3]]
+    output [1,3,5,6]
+    You're incrementing everything from index 1 to index 3 then index 2 to index 3.
+
+    Ok in a seperate LAZY array, put 1 and -1 in the locations where we are incrementing (1 for increment, and put -1 in the location 1 after the end of interval).
+    Lazy array will have to be 1 size bigger, to deal with end intervals. 
+
+    Process all the intervals, then at the end, do a cumulative sum array with that table. 
+    Cumulative sum array will include all your incrementing!
+
+    So for  [[1,3],[2,3]] we create the following:
+    [0, 1, 1, 0, -2]
+    -> Cum sum is:
+    [0,1,2,2,0]
+    Then sum it with original array a (ignore last element in lazy):
+
+    [1,3,5,6] -> which is our answer. 
+
+    Soln Code:
+        partial sum technique
+        we will increment the start position with 1 and decrease the (end + 1) position then will perform a prefix sum
+
+        vector<int> incrementIntervals(vector <int> array , vector<vector<int>> intervals) {
+        int n = (int)array.size();
+        vector <int> partialSum(n + 1 , 0);
+        for(auto index : intervals){
+        partialSum[index[0]]++;
+        partialSum[index[1] + 1]--;
+        }
+        for(int i = 1;i <= n;i++){
+            partialSum[i] += partialSum[i - 1];
+        }
+        vector <int> answer(n , 0);
+        for(int i = 0;i < n;i++){
+            answer[i] = array[i] + partialSum[i];
+        }
+        return answer;
+        }
+        hope this helps you
+        https://codeforces.com/blog/entry/15729 
+
+
+
+25.9)  SQRT Decomposition
+        Suppose we have an array a1, a2, ..., an and . We partition this array into k pieces each containing k elements of a.
+
+        Doing this, we can do a lot of things in . Usually we use them in the problems with modify and ask queries.
+
+        Problems : Holes, DZY Loves Colors, RMQ (range minimum query) problem
+
+25.95) Sparse Table
+        The main problem that we can solve is RMQ problem, we have an array a1, a2, ..., an and some queries. Each query gives you numbers 
+        l and r (l ≤ r) and you should print the value of min(al, al + 1, ..., ar) .
+
+        Solving using Sparse Table : For each i that 1 ≤ i ≤ n and for each j that 0 ≤ j and i + 2^j - 1 ≤ n, we keep the value 
+        of min(ai, ai + 1, ..., ai + (2^j - 1) ) in st[i][j] (preprocess) : (code is 0-based)
+
+        for(int j = 0;j < MAX_LOG; j++)
+            for(int i = 0; i < n; i ++) if(i + (1 << j) - 1 < n)
+                st[i][j] = (j ?   min(st[i][j-1], st[i + (1 << (j-1)) - 1][j-1])    :    a[i]);
+
+
+        And then for each query, first of all, find the maximum x such that 2^x ≤ r - l + 1 and answer is min(st[l][x], st[r - 2^x + 1][x]) .
+
+        So, the main idea of Sparse Table, is to keep the value for each interval of length 2^k (for each k).
+
+        You can use the same idea for LCA problem and so many other problems.
+        So preprocess will be in O(n.log(n)) and query will be in O(1)
+
+25.96) Median of Medians
+
+    The task is to find a median (the element with central index) in a sorted set of elements storing on 
+    multiple (here assuming the number is 1000) servers.
+
+    It’s pointless to do a full sort of set - it won’t fit in memory :) The algorithm is to sort arrays on each server and take a 
+    median from each other. Now we get a set of 1000 medians and it’s easy to find the result here.
+
+    More information: http://en.wikipedia.org/wiki/Selection_algorithm#Linear_general_selection_algorithm_-_Median_of_Medians_algorithm
+
+
+25.97) XOR Double Linked Lists:
+        Sometimes you implement linked list and think whether it’s needed to store 1 or 2 pointers in each node. 
+        The space really matters, especially if you store millions of records but sometimes it’s good to have a way to 
+        traverse back from a given node. There is one hack how to store two pointers using just half of the size 
+        (meaning that you’ll use size like for a single-linked list).
+
+        You store previous XOR next pointers.
+
+        Usage: when you traverse front (or back) the linked list, you just get the value and XOR it with the last element, taking the next value.
+
+        
+        A <-> B <-> C <-> D <-> E
+             A^C   B^D   C^E
+ 
+25.98) Recursively convert number to base x
+
+    Solution is incredibly simple and uses recursion:
+
+    //convert number to base X
+    public String convertToBase(int a, int x) {
+    if (a < x) return a;
+    return convertToBase(a/x, x) + (a%x);
+    }
+
+25.99) Matrix block (REMEMBER TO USE PRIORITY QUEUE VS DP ANALYSIS!)
+        Given a 2D grid with n rows and m columns. Some cells are blocked which you cannot pass through. From each cell you can go either up, 
+        down, left or right. Find the shortest path from (0, 0) to (n-1, m-1).
+        -> bfs
+
+        Followup:
+
+        Now assume that there exists no path from (0, 0) to (n-1, m-1) (All paths are blocked).
+        Find the minimum number of cells that you need to 
+        unblock such that there exists a path from (0, 0) to (n-1, m-1). Can you solve it in O(n+m)
+
+        Can follow up be solved with DP. I think so
+        
+        Also can we also just go right and down, why woudl we go up and left if we are starting at (0, 0)
+
+        To do follow up,
+        write dfs code that tries all paths to go from (0,0) to destination, and also passes through obstacles!
+        Keep track of path that took min # of obstacles to remove and memoize this in dfs
+
+        Then return min of left and down!
+
+        @lru_cache(None)
+        def helper(i, j)
+
+            if i == N-1 and J == M-1:
+                return 0
+            
+            isObstacle = 0
+            if maze[i][j] == "obstacle":
+                isObstacle = 1
+
+
+            return isObstacle + min(helper(i+1, j), helper(i, j+1)) 
+
+        This is linear time right!
+
+        Follow up can also be solved with PQ and exhausting all paths!, and priority is # of blocks you touched so far.
+
+
+
+
+
+
+25.999) FUNCTOOLS CACHE:
+
+    functools.cache was newly added in version 3.9.
+
+    The documentation states:
+
+    Simple lightweight unbounded function cache. Sometimes called “memoize”.
+
+    Returns the same as lru_cache(maxsize=None), creating a thin wrapper around a dictionary lookup for the 
+    function arguments. Because it never needs to evict old values, this is smaller and faster than lru_cache() with a size limit.
+
+    Example from the docs:
+
+    @cache
+    def factorial(n):
+        return n * factorial(n-1) if n else 1
+                
+
+


26) Modified Binary Search  
        First, find the middle of start and end. 
        An easy way to find the middle would be: 
        middle = (start + end) / 2.
        
        But this has a good chance of producing an integer overflow 
        so it’s recommended that you represent the middle as: 
        middle = start + (end — start) / 2

        Problems featuring the Modified Binary Search pattern:
        Order-agnostic Binary Search (easy)
        Search in a Sorted Infinite Array (medium)


+
+26.5) Basic Calculator 1,2,3 (I havent personally done this yet): [study other solutions too, and do it again.]
+
+
+    This algorithm works for Basic Calculator (BC I) problem, where we can have only + - ( ) operations, for Basic Calculator II (BC II), 
+    where we can have only + - * / operations and also for Basic Calculator III (BC III), where we can have all + - * / ( ) operations.
+
+    Stack of monomials
+    The idea is to use both stack and recursion (which can be seen as 2 stack, because recursion use implicit stack). First, let us consider, 
+    that we do not have any brackets. Then let us keep the stack of monomial, consider the example s = 1*2 - 3\4*5 + 6. 
+    Then we want our stack to be equal to [1*2, -3\4*5, 6], let us do it step by step:
+
+    Put 1 into stack, we have stack = [1].
+    We can see that operation is equal to *, so we pop the last element from our stack and put new element: 1*2, now stack = [1*2].
+    Now, operation is equal to -, so we put -3 to stack and we have stack = [1*2, -3] now
+    Now, operation is equal to \, so we pop the last element from stack and put -3\4 instead, stack = [1*2, -3\4]
+    Now, operation is equal to *, so we pop last element from stack and put -3\4*5 instead, stack = [1*2, -3\4*5].
+    Finally, operation is equal to +, so we put 6 to stack: stack = [1*2, -3\4*5, 6]
+    Now, all we need to do is to return sum of all elements in stack.
+
+    How to deal with brackets
+    If we want to be able to process the brackets properly, all we need to do is to call our calculator recursively! 
+    
+    When we see the open bracket (, we call calculator with the rest of our string, and when we see closed bracket ')', we give back 
+    the value of expression inside brackets and the place where we need to start when we go out of recursion.
+
+    Complexity
+    Even though we have stack and also have recursion, we process every element only once, so time complexity is O(n). However 
+    we pass slice of string as argument each time we meet bracket, so time complexity can go upto O(n^2) on example like (1+(1+(... +))) 
+    with O(n) open brackets. Space complexity is potentially O(n), because we need to keep stacks, but each element not more than once.
+
+    class Solution:
+        def calculate(self, s):
+            def update(op, v):
+                if op == "+": stack.append(v)
+                if op == "-": stack.append(-v)
+                if op == "*": stack.append(stack.pop() * v)           #for BC II and BC III
+                if op == "/": stack.append(int(stack.pop() / v))      #for BC II and BC III
+                
+            # the cool trick here is we assign + as our first sign, the operator becomes "post fix"        
+            it, num, stack, sign = 0, 0, [], "+"
+            
+            while it < len(s):
+                if s[it].isdigit():
+                    num = num * 10 + int(s[it])
+                elif s[it] in "+-*/":
+                    update(sign, num)
+                    num, sign = 0, s[it]
+                elif s[it] == "(":                                        # For BC I and BC III
+                    num, j = self.calculate(s[it + 1:])
+                    it = it + j
+                elif s[it] == ")":                                        # For BC I and BC III
+                    update(sign, num)
+                    return sum(stack), it + 1
+                it += 1
+            update(sign, num)
+            return sum(stack)
+        
+
+    Solution 2
+    The problem of previous code is that we pass slice of string as parameter. In python it works quite fast, because function is 
+    implemented in C and it works very fast. If we want to have honest linear time, we need to pass index as parameter. 
+    (there is alternative way like I used in problem 1896 https://leetcode.com/problems/minimum-cost-to-change-the-final-value-of-expression/discuss/1267304/Python-Recursion-dfs-solution-explained, 
+    where we can precalculate pairs of open and closing brackets)
+
+    Complexity
+    Now time complexity it is O(n), space is still O(n).
+
+    class Solution:
+        def calculate(self, s):    
+            def calc(it):
+                def update(op, v):
+                    if op == "+": stack.append(v)
+                    if op == "-": stack.append(-v)
+                    if op == "*": stack.append(stack.pop() * v)
+                    if op == "/": stack.append(int(stack.pop() / v))
+            
+                num, stack, sign = 0, [], "+"
+                
+                while it < len(s):
+                    if s[it].isdigit():
+                        num = num * 10 + int(s[it])
+                    elif s[it] in "+-*/":
+                        update(sign, num)
+                        num, sign = 0, s[it]
+                    elif s[it] == "(":
+                        num, j = calc(it + 1)
+                        it = j - 1
+                    elif s[it] == ")":
+                        update(sign, num)
+                        return sum(stack), it + 1
+                    it += 1
+                update(sign, num)
+                return sum(stack)
+
+            return calc(0)
+
+    Note:
+
+    Awsome solution, but it needs a little fix to pass this test case "14-3/2" in python (haven't tried in python3 tho), 
Update the function for -ve integer division as follows
+
+    if operation == "/":
+                    prev_value = stack.pop()
+                    if prev_value <0:
+                        prev_value = abs(prev_value)
+                        stack.append(-(int(prev_value/value)))
+                    else:
+                        stack.append(int(prev_value/value))
+
+921) Minimum Add to Make Parentheses Valid
+        Medium
+
+        2403
+
+        139
+
+        Add to List
+
+        Share
+        A parentheses string is valid if and only if:
+
+        It is the empty string,
+        It can be written as AB (A concatenated with B), where A and B are valid strings, or
+        It can be written as (A), where A is a valid string.
+        You are given a parentheses string s. In one move, you can insert a parenthesis at any position of the string.
+
+        For example, if s = "()))", you can insert an opening parenthesis to be "(()))" or a closing parenthesis to be "())))".
+        Return the minimum number of moves required to make s valid.
+
+
+        class Solution:
+            def minAddToMakeValid(self, s: str) -> int:
+            
+            
+                invalid_opens = 0
+                invalid_closes = 0
+                for i in s:
+                    if i == "(":
+                        invalid_opens += 1         
+                    elif i == ")":
+                        if(invalid_opens > 0):
+                            invalid_opens -= 1
+                        else:
+                            invalid_closes += 1
+                
+                return invalid_opens + invalid_closes 
+                 
+Given an m x n matrix mat, return an array of all the elements of the array in a diagonal order.\
+        Hey guys, super easy solution here, with NO DIRECTION CHECKS!!!
+        The key here is to realize that the sum of indices on all diagonals are equal.
+            -> Exploit property
+
+        class Solution(object):
+            def findDiagonalOrder(self, matrix):
+                """
+                :type matrix: List[List[int]]
+                :rtype: List[int]
+                """
+                d={}
+                #loop through matrix
+                for i in range(len(matrix)):
+                    for j in range(len(matrix[i])):
+                        #if no entry in dictionary for sum of indices aka the diagonal, create one
+                        if i + j not in d:
+                            d[i+j] = [matrix[i][j]]
+                        else:
+                        #If you've already passed over this diagonal, keep adding elements to it!
+                            d[i+j].append(matrix[i][j])
+                # we're done with the pass, let's build our answer array
+                ans= []
+                #look at the diagonal and each diagonal's elements
+                for entry in d.items():
+                    #each entry looks like (diagonal level (sum of indices), [elem1, elem2, elem3, ...])
+                    #snake time, look at the diagonal level
+                    if entry[0] % 2 == 0:
+                        #Here we append in reverse order because its an even numbered level/diagonal. 
+                        [ans.append(x) for x in entry[1][::-1]]
+                    else:
+                        [ans.append(x) for x in entry[1]]
+                return ans  
+
+
+26.6) Graph algo question series of queries:
+    You are given a weighted graph G that contains N nodes and M edges. 
+    Each edge has weight(w) associated to it. You are given Q queries of the following type:
+
+    -> x y W. Find if there exists a path in G between nodes x and y such that the weight of each edge in the 
        path is at most W. If such a path exists print 1, otherwise print 0.
+
+    Constraints:
+    1<=N,Q,M,<=10^5
+    1<=w,W<=10^5
+    1<=x,y<=N
+        Here is my solution with complexity: O(MlogN+QlogN)
+        Idea: sort the edges and queries by weights then join nodes and check if the nodes are connected i.e. have the same parent 
+        node in disjoint-sets.n the very beginning all nodes are disconnected. Then starting from edges with smallest weight the nodes are 
+        connected (Union) while the weight of edge is less or equal to the weight in the current query.
+        If the nodes have the same parent node for the current query then the query is counted.
+
+        Ex. edges = [
+        [0, 1, 5],
+        [1, 2, 6],
+        [2, 3, 7],
+        [0, 3, 4]
+        ]
+        queries = [
+        [0, 3, 5],
+        [1, 0, 3]
+        ]
+
+        Answer - 1,0
+
+        def solve(edges, queries):
+            def find(a):
+                if par[a] < 0:
+                    return a
+                par[a] = find(par[a])
+                return par[a]
+
+            def merge(a,b):
+                if a!=b:
+                    if rank[a]>rank[b]:
+                        par[b]=a
+                        rank[a]+=rank[b]
+                    else:
+                        par[a]=b;
+                        rank[b]+=rank[a]
+
+            n = len(edges)
+            if not edges or not queries:
+                return 0
+                        
+            par=[-1 for i in range(n+10)]
+            rank=[1 for i in range(n+10)]
+
+            edges.sort(key = lambda x:  x[2])
+            queries.sort(key = lambda x:    x[2])
+
+            pos = 0
+            for i,j,w in queries:
+                while pos < len(edges) and edges[pos][2] <= w:
+                    k = edges[pos]
+                    a = find(k[0])
+                    b = find(k[1])
+                    merge(a,b)
+                    pos+=1
+                print(1) if find(i) == find(j) else print(0)
+    
+
+
+26.7) MIN-MAX DP CARDS GOOGLE (same as stone game 3)
+        Two players are playing a card game where the deck of cards are layed out in a straight line and each card value is visible to both the players.
+        The value of each card can range from a reasonable [-ve to +ve] number and the length of deck in n.
+
+        The rules of the game are such:
+
+        Player 1 starts the game
+        Each player has 3 option:
+        (Option 1: Pick 1st card)
+        (Option 2: Pick 1st two cards)
+        (Option 3: Pick 1st three cards)
+        You're only allowed to pick cards from the left side of the deck
+        Both players have to play optimally.
+        Return the maximum sum of cards Player 1 can obtain by playing optimally.
+
+        Example 1:
+
+        Input: cards = [1, 2, -3, 8]
+        Output: 3
+        Explanation:
+        Turn 1: Player 1 picks the first 2 cards: 1 + 2 = 3 points
+        Turn 2: Player 2 gets the rest of the deck: -3 + 8 = 5 points
+        Example 2:
+
+        Input: cards = [1, 1, 1, 1, 100]
+        Output: 101
+        Explanation:
+        Turn 1: Player 1 picks cards[0] = 1 point
+        Turn 2: Player 2 picks cards[1] + cards[2] + cards[3] = 3 points
+        Turn 3: Player 1 picks cards[4] = 100 points
+
+        from functools import lru_cache
+        def max_score(cards):
+            @lru_cache(maxsize=None)
+            def minimax(idx, player1):
+                if idx >= len(cards):
+                    return 0
+                if player1:
+                    return max([sum(cards[idx:idx + o]) + minimax(idx + o, not player1) \
+                                for o in range(1, 4)])
+                else:
+                    return min([minimax(idx + o, not player1) for o in range(1, 4)])
+
+            return minimax(0, True)
+
+        Space O(N)
+        // dp[i] means the max possible score the 1st player can get starting from index i
+        // best strategy means we take the option that minimizes the max possible score for our opponent
+
+        def max_score_iterative(cards):
+            total, n = 0, len(cards)
+            dp = [0] * (n + 3)
+            
+            for i in range(n - 1, -1, -1):
+                total += cards[i]
+                // we then get (total - the minimized possible score for our opponent)
+                // we maximize our own score, by minning opponent, 
+                // but on the next iteration, the thing we compute now will be used as part of the "min"
+                dp[i] = total - min([dp[i + o] for o in range(1, 4)])
+            
+            return dp[0]
+
+        Optimized bottom-up:
+        Time: O(n); space: O(1).
+        from collections import deque
+        def max_score_iterative_opt(cards):
+            total, n = 0, len(cards)
+            dp = deque([0] * 3)
+            
+            for i in range(n - 1, -1, -1):
+                total += cards[i]
+                head = total - min(dp)
+                dp.pop()
+                dp.appendleft(head)
+            
+            return dp[0]
+
+
+
+
+26.8) Problems that seemingly cant improve but do (always attempt bin search theory!):
+
+        You are given a string that is grouped together by characters. For example a sample input could be: "hhzzzzaaa", 
+        and we need to output the most frequently occuring character so for our example we would output 'z'.
+
+        I was only asked this question on the phone screen.
+
+        Optimization 1:
+
+        Since the characters are kept in groups, we need to find the index where the character changes.
+        To get the count of a specific character, we need to subtract i - pivot.
+        Linear Time and Constant Space
+
+
+
+        I thought Linear time and constant space was the optimal solution, but it turns out the interviewer wanted me to optimize 
+        further into Log N time and constant space. This is where I struggled. 
+        After a hint I was able to code out the binary search solution.
+
+        Get the start from skipping and binary search for the end, k * log(n) but since k is capped at the number of 
+        different characters which should be a fixed amount. k is constant so log(n). Thanks for the detailed post, 
+        I think someone else mentioned this problem, but the description was very vague.
+
+        public class Main {
+            //returns position after the last instance of c, end variable is not needed but I just put it in for clarity
+            private static int findEnd(String s, char c, int start, int end){
+                while(start<=end){
+                    int mid = start+(end-start)/2;
+                    if(s.charAt(mid) == c){
+                        start = mid+1;
+                    }else{
+                        end = mid-1;
+                    }
+                }
+                return start;
+            }
+            private static char findMostFrequent(String s){
+                int i = 0;
+                int mostFreqCount = 0;
+                //assuming s is not blank
+                char mostFreq = ' ';
+                while(i<s.length()){
+                    char c = s.charAt(i);
+                    int end = findEnd(s, c, i, s.length()-1);
+                    int start = i;
+                    int count = end-start;
+                    if(count> mostFreqCount){
+                        mostFreqCount = count;
+                        mostFreq = c;
+                    }
+                    i = end;
+                }
+                return mostFreq;
+            }
+            public static void main(String[] args) {
+                System.out.println(findMostFrequent("hhzzzzaaa"));
+            }
+        }
+
+
+26.7) Creating a stock exchange:
+        1801. Number of Orders in the Backlog
+        Medium
+
+        166
+
+        176
+
+        Add to List
+
+        Share
+        You are given a 2D integer array orders, where each orders[i] = [pricei, amounti, orderTypei] denotes that amounti orders have been placed of type orderTypei at the price pricei. The orderTypei is:
+
+        0 if it is a batch of buy orders, or
+        1 if it is a batch of sell orders.
+        Note that orders[i] represents a batch of amounti independent orders with the same price and order type. All orders represented by orders[i] will be placed before all orders represented by orders[i+1] for all valid i.
+
+        There is a backlog that consists of orders that have not been executed. The backlog is initially empty. When an order is placed, the following happens:
+
+        If the order is a buy order, you look at the sell order with the smallest price in the backlog. If that sell order's price is smaller than or equal to the current buy order's price, they will match and be executed, and that sell order will be removed from the backlog. Else, the buy order is added to the backlog.
+        Vice versa, if the order is a sell order, you look at the buy order with the largest price in the backlog. If that buy order's price is larger than or equal to the current sell order's price, they will match and be executed, and that buy order will be removed from the backlog. Else, the sell order is added to the backlog.
+        Return the total amount of orders in the backlog after placing all the orders from the input. Since this number can be large, return it modulo 109 + 7.
+
+        class Solution:
+            def getNumberOfBacklogOrders(self, orders):
+                b, s = [], []
+                
+                for p,a,o in orders:
+                    if o == 0:
+                        heapq.heappush(b, [-p, a])
+                        
+                    elif o == 1:
+                        heapq.heappush(s, [p, a])
+                    
+                    # Check "good" condition
+                    while s and b and s[0][0] <= -b[0][0]:
+                        a1, a2 = b[0][1], s[0][1]
+                        
+                        if a1 > a2:
+                            b[0][1] -= a2
+                            heapq.heappop(s)
+                        elif a1 < a2:
+                            s[0][1] -= a1
+                            heapq.heappop(b)
+                        else:
+                            heapq.heappop(b)
+                            heapq.heappop(s)
+                            
+                count = sum([a for p,a in b]) + sum([a for p,a in s])
+                return count % (10**9 + 7)
+
+
+26.8) Common Prefixes
+
+        The distance between 2 binary strings is the sum of their lengths after removing the common prefix. 
+        For example: the common prefix of 1011000 and 1011110 is 1011 so the distance is len("000") + len("110") = 3 + 3 = 6.
+
+        Given a list of binary strings, pick a pair that gives you maximum distance 
+        among all possible pair and return that distance.
+
+
+
+26.9)  904. Fruit Into Baskets
+        Medium
+
+        897
+
+        67
+
+        Add to List
+
+        Share
+        You are visiting a farm that has a single row of fruit trees arranged from left to right. 
        The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.
+
+        You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:
+
+        You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
+        Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
+        Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
+        Given the integer array fruits, return the maximum number of fruits you can pick.
+
+        
+
+        Example 1:
+
+        Input: fruits = [1,2,1]
+        Output: 3
+        Explanation: We can pick from all 3 trees.
+        Example 2:
+
+        Input: fruits = [0,1,2,2]
+        Output: 3
+        Explanation: We can pick from trees [1,2,2].
+        If we had started at the first tree, we would only pick from trees [0,1].
+        Example 3:
+
+        Input: fruits = [1,2,3,2,2]
+        Output: 4
+        Explanation: We can pick from trees [2,3,2,2].
+        If we had started at the first tree, we would only pick from trees [1,2].
+        
+        Soln:
+            # slide the window!
+            '''  
+            take a fruit, 
+            take another fruit type,
+            extend window untily oucant.
+            
+            throw away fruit until one fruit is there. 
+            
+            then move right pointer to incldue a new fruit.
+            Repeat
+            record max.
+            '''
+
+        Soln 2: O(1) space doing Longest Subarray With 2 Elements
+
+            class Solution {
+                public int totalFruit(int[] tree) {
+                    // track last two fruits seen
+                    int lastFruit = -1;
+                    int secondLastFruit = -1;
+                    int lastFruitCount = 0;
+                    int currMax = 0;
+                    int max = 0;
+                    
+                    for (int fruit : tree) {
+                        if (fruit == lastFruit || fruit == secondLastFruit)
+                            currMax++;
+                        else
+                            currMax = lastFruitCount + 1; // last fruit + new fruit
+                        
+                        if (fruit == lastFruit)
+                            lastFruitCount++;
+                        else
+                            lastFruitCount = 1; 
+                        
+                        if (fruit != lastFruit) {
+                            secondLastFruit = lastFruit;
+                            lastFruit = fruit;
+                        }
+                        
+                        max = Math.max(max, currMax);
+                    }
+                    
+                    return max;
+                }
+            }
+
+26.95) Heap Vs Binary Search Vs Quick Select -> Super important bin search technique
+
+        973. K Closest Points to Origin
+        Medium
+        Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an 
        integer k, return the k closest points to the origin (0, 0).
+
+        The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).
+
+        You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).
+
+        Soln:
+
+        Max heap priority soln is easy (keep track of k furthest points)
+            class Solution:
+                def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
+                    # Since heap is sorted in increasing order,
+                    # negate the distance to simulate max heap
+                    # and fill the heap with the first k elements of points
+                    heap = [(-self.squared_distance(points[i]), i) for i in range(k)]
+                    heapq.heapify(heap)
+                    for i in range(k, len(points)):
+                        dist = -self.squared_distance(points[i])
+                        if dist > heap[0][0]:
+                            # If this point is closer than the kth farthest,
+                            # discard the farthest point and add this one
+                            heapq.heappushpop(heap, (dist, i))
+                    
+                    # Return all points stored in the max heap
+                    return [points[i] for (_, i) in heap]
+                
+                def squared_distance(self, point: List[int]) -> int:
+                    """Calculate and return the squared Euclidean distance."""
+                    return point[0] ** 2 + point[1] ** 2
+
+        Binary Search Soln (Time Complexity O(N) space O(N))
+            
+            It would be NlogN but we elimiate our search space as we iterate the bin search!
+
+            In this case, however, we can improve upon the time complexity of this modified binary search by eliminating 
+            one set of points at the end of each iteration. If the target distance yields fewer than kk closer points, 
+            then we know that each of those points belongs in our answer and can then be ignored in later iterations. 
+            If the target distance yields more than kk closer points, on the other hand, we know that 
+            we can discard the points that fell outside the target distance.
+            
+            
+            Since we're going to be using the midpoint of the range of distances for each iteration of our binary search, we should
+            calculate the actual Euclidean distance for each point, rather than using the squared distance as in the other approaches. 
+            An even distribution of the points in the input array will yield an even distribution of
+            distances, but an uneven distribution of squared distances.
+            Complexity = N + N/2 + N/4 ... = 2N
+            
+            
+            class Solution:
+                def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
+                    # Precompute the Euclidean distance for each point
+                    distances = [self.euclidean_distance(point) for point in points]
+                    # Create a reference list of point indices
+                    remaining = [i for i in range(len(points))]
+                    # Define the initial binary search range
+                    low, high = 0, max(distances)
+                    
+                    # Perform a binary search of the distances
+                    # to find the k closest points
+                    closest = []
+                    while k:
+                        mid = (low + high) / 2
+                        closer, farther = self.split_distances(remaining, distances, mid)
+                        if len(closer) > k:
+                            # If more than k points are in the closer distances
+                            # then discard the farther points and continue
+                            remaining = closer
+                            high = mid
+                        else:
+                            # Add the closer points to the answer array and keep
+                            # searching the farther distances for the remaining points
+                            k -= len(closer)
+                            closest.extend(closer)
+                            remaining = farther
+                            low = mid
+                            
+                    # Return the k closest points using the reference indices
+                    return [points[i] for i in closest]
+
+                def split_distances(self, remaining: List[int], distances: List[float],
+                                    mid: int) -> List[List[int]]:
+                    """Split the distances around the midpoint
+                    and return them in separate lists."""
+                    closer, farther = [], []
+                    for index in remaining:
+                        if distances[index] <= mid:
+                            closer.append(index)
+                        else:
+                            farther.append(index)
+                    return [closer, farther]
+
+                def euclidean_distance(self, point: List[int]) -> float:
+                    """Calculate and return the squared Euclidean distance."""
+                    return point[0] ** 2 + point[1] ** 2
+
+
+
+        Quick Select Soln (with partiail sorting partition function):
+            Lets reduce space to O(1) by modifying in place
+            Try to understand the partition function!!
+
+            1.Return the result of a QuickSelect algorithm on the points array to kk elements.
+            2. In the QuickSelect function:
+                Repeatedly partition a range of elements in the given array while homing in on the k^{th}kth element.
+            3. In the partition function:
+                Choose a pivot element. The pivot value will be squared Euclidean distance from the origin to the pivot element and will be compared to the 
+                    squared Euclidean distance of all other points in the partition.
+                Start with pointers at the left and right ends of the partition, then while the two pointers have not yet met:
+                    If the value of the element at the left pointer is smaller than the pivot value, increment the left pointer.
+                    Otherwise, swap the elements at the two pointers and decrement the right pointer.
+            Make sure the left pointer is past the last element whose value is lower than the pivot value.
+            Return the value of the left pointer as the new pivot index.
+            4. Return the first kk elements of the array.
+
+
+
+            class Solution:
+                def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
+                    return self.quick_select(points, k)
+                
+                def quick_select(self, points: List[List[int]], k: int) -> List[List[int]]:
+                    """Perform the QuickSelect algorithm on the list"""
+                    left, right = 0, len(points) - 1
+                    pivot_index = len(points)
+                    while pivot_index != k:
+                        # Repeatedly partition the list
+                        # while narrowing in on the kth element
+                        pivot_index = self.partition(points, left, right)
+                        if pivot_index < k:
+                            left = pivot_index
+                        else:
+                            right = pivot_index - 1
+                    
+                    # Return the first k elements of the partially sorted list
+                    return points[:k]
+                
+                def partition(self, points: List[List[int]], left: int, right: int) -> int:
+                    """Partition the list around the pivot value"""
+                    pivot = self.choose_pivot(points, left, right)
+                    pivot_dist = self.squared_distance(pivot)
+                    while left < right:
+                        # Iterate through the range and swap elements to make sure
+                        # that all points closer than the pivot are to the left
+                        if self.squared_distance(points[left]) >= pivot_dist:
+                            points[left], points[right] = points[right], points[left]
+                            right -= 1
+                        else:
+                            left += 1
+                    
+                    # Ensure the left pointer is just past the end of
+                    # the left range then return it as the new pivotIndex
+                    if self.squared_distance(points[left]) < pivot_dist:
+                        left += 1
+                    return left
+                
+                def choose_pivot(self, points: List[List[int]], left: int, right: int) -> List[int]:
+                    """Choose a pivot element of the list"""
+                    return points[left + (right - left) // 2]
+                
+                def squared_distance(self, point: List[int]) -> int:
+                    """Calculate and return the squared Euclidean distance."""
+                    return point[0] ** 2 + point[1] ** 2
+
+
+26.97)  1509. Minimum Difference Between Largest and Smallest Value in Three Moves
+
+        You are given an integer array nums. In one move, you can choose one element of nums and change it by any value.
+
+        Return the minimum difference between the largest and smallest value of nums after performing at most three moves.
+
+        
+
+        Example 1:
+
+        Input: nums = [5,3,2,4]
+        Output: 0
+        Explanation: Change the array [5,3,2,4] to [2,2,2,2].
+        The difference between the maximum and minimum is 2-2 = 0.
+        Example 2:
+
+        Input: nums = [1,5,0,10,14]
+        Output: 1
+        Explanation: Change the array [1,5,0,10,14] to [1,1,0,1,1]. 
+        The difference between the maximum and minimum is 1-0 = 1.
+
+        class Solution:
+            def minDifference(self, nums: List[int]) -> int:
+                '''
+                get top 3 mins,
+                get top 3 maxes
+                
+                Ok remove the worst offenders. 
+                compare min and max. 
+                
+                try every possibility with these 3. 
+                i guess!
+                We have 4 plans:
+
+                kill 3 biggest elements
+                kill 2 biggest elements + 1 smallest elements
+                kill 1 biggest elements + 2 smallest elements
+                kill 3 smallest elements
+                '''
+                pass
+            
+        import heapq
+        class Solution:
+            def minDifference(self, nums: List[int]) -> int:
+                
+                if len(nums) <= 3:
+                    return 0
+                
+                top_4 = []
+                for x in nums:
+                    heapq.heappush(top_4,x)
+                    if len(top_4) > 4:
+                        heapq.heappop(top_4)
+                top_4.sort()
+                down_4 = []
+                for x in nums:
+                    heapq.heappush(down_4,-x)
+                    if len(down_4) > 4:
+                        heapq.heappop(down_4)
+                down_4.sort()
+                down_4 = [-x for x in down_4]
+                
+                
+                res = float('inf')
+                for i in range(4):
+                    if abs(top_4[i] - down_4[3-i]) < res:
+                        res = abs(top_4[i] - down_4[3-i])
+                
+                return res
+
+
+26.98)  Do good tracking as you iterate left to right
+
+        1525. Number of Good Ways to Split a String
+        You are given a string s.
+
+        A split is called good if you can split s into two non-empty strings sleft and sright where their concatenation is equal to 
+        s (i.e., sleft + sright = s) and the number of distinct letters in sleft and sright is the same.
+
+        Return the number of good splits you can make in s.
+
+        Have two dicionaries to track the frequency of letters for the left partition and the right partition. Initially, 
+        left partion will be empty. For each loop, update both dictionaries to reflect the frequency on the left and right 
+        partition. If the length of both partitions are equal, we found the good ways, so increment the result.
+
+
+        class Solution:
+            def numSplits(self, s: str) -> int:
+                left_count = collections.Counter()
+                right_count = collections.Counter(s)
+                res = 0
+                for c in s:
+                    left_count[c] += 1
+                    right_count[c] -= 1
+                    if right_count[c] == 0:
+                        del right_count[c]
+                    
+                    if len(left_count) == len(right_count):
+                        res += 1
+                        
+                return res
+
+26.99) DP or something else? 
+
+        A pizza shop offers n pizzas along with m toppings. A customer plans to spend around x coins. 
+        The customer should order exactly one pizza, and may order zero, one or two toppings. Each topping may be order only once.
+
+        Given the lists of prices of available pizzas and toppings, what is the price closest to x of possible orders? 
+        Here, a price said closer to x when the difference from x is the smaller. 
+        Note the customer is allowed to make an order that costs more than x.
+
+        Example 1:
+
+        Input: pizzas = [800, 850, 900], toppings = [100, 150], x = 1000
+        Output: 1000
+        Explanation:
+        The customer can spend exactly 1000 coins (two possible orders).
+        Example 2:
+
+        Input: pizzas = [850, 900], toppings = [200, 250], x = 1000
+        Output: 1050
+        Explanation:
+        The customer may make an order more expensive than 1000 coins.
+        Example 3:
+
+        Input: pizzas = [1100, 900], toppings = [200], x = 1000
+        Output: 900
+        Explanation:
+        The customer should prefer 900 (lower) over 1100 (higher).
+        Example 4:
+
+        Input: pizzas = [800, 800, 800, 800], toppings = [100], x = 1000
+        Output: 900
+        Explanation:
+        The customer may not order 2 same toppings to make it 1000. 
+
+
+        def closestPrice(pizzas, toppings, x):
+            import bisect
+            closest = float('inf')
+            new_toppings = [0]
+            
+            # Generate combinations for 0, 1, and 2 toppings
+            for i in range(len(toppings)):
+                new_toppings.append(toppings[i])
+                for j in range(i+1, len(toppings)):
+                    new_toppings.append(toppings[i] + toppings[j])
+            new_toppings.sort()
+            for pizza in pizzas:
+                idx = bisect.bisect_left(new_toppings, x - pizza)
+                for j in range(idx-1, idx+2):
+                    if 0 <= j < len(new_toppings):
+                        diff = abs(pizza + new_toppings[j] - x)
+                        if diff == abs(closest - x):
+                            closest = min(closest, pizza + new_toppings[j]) # When two are equal, take the lowest one according to example 3
+                        elif diff < abs(closest - x):
+                            closest = pizza + new_toppings[j]
+            return closest
+
+26.999) BFS TOPOLOGICAL SORT TECHNIQUE/MEET IN THE MIDDLE
+    TREE GRAPH PBOELMS AND GRAPH PROBLEMS IN GENERAL! THINK ABOUT STRUCTURE OF GRAPHS AND TREES TO SOLVE AS MUCH AS POSSIBLY
            DONT JUST BLINDLY APPLY DFS/BFS -> BFS/DFS FROM LEAVES, OR OTHER TYPES OF NODES KNOW HOW OT DO THIS BE SMART, THINK CREATIVELY 
            ALL DIRECTIONS IN GRAPHS AND TREES, INORDER, PREORDER, ETC, FROM LEAVE, FROM PARENT, UP DOWN YOU

+        310. Minimum Height Trees
+
+        Share
+        A tree is an undirected graph in which any two vertices are connected by exactly one path. 
+        In other words, any connected graph without simple cycles is a tree.
+
+        Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates 
+        that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. 
+        When you select a node x as the root, the result tree has height h. Among all possible rooted trees, 
         those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).
+
+        Return a list of all MHTs' root labels. You can return the answer in any order.
+
+        The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.
+
+        Soln:
+            We start from every end, by end we mean vertex of degree 1 (aka leaves). We let the pointers move the same speed. 
+            When two pointers meet, we keep only one of them, until the last two pointers 
+            meet or one step away we then find the roots.
+
+            It is easy to see that the last two pointers are from the two ends of the longest path in the graph.
+
+            The actual implementation is similar to the BFS topological sort. Remove the leaves, update the degrees of inner vertexes. 
+            Then remove the new leaves. Doing so level by level until there are 2 or 1 nodes left. What's left is our answer!
+
+            The time complexity and space complexity are both O(n).
+
+            Note that for a tree we always have V = n, E = n-1.
+
+        def findMinHeightTrees(self, n, edges):
+            if n == 1: return [0] 
+            adj = [set() for _ in xrange(n)]
+            for i, j in edges:
+                adj[i].add(j)
+                adj[j].add(i)
+
+            leaves = [i for i in xrange(n) if len(adj[i]) == 1]
+
+            while n > 2:
+                n -= len(leaves)
+                newLeaves = []
+                for i in leaves:
+                    j = adj[i].pop()
+                    adj[j].remove(i)
+                    if len(adj[j]) == 1: newLeaves.append(j)
+                leaves = newLeaves
+            return leaves
+            
+        # Runtime : 104ms
+
+26.99999) Trees and leaf removal
+    You are given a tree-shaped undirected graph consisting of n nodes labeled 1...n and n-1 edges. 
        The i-th edge connects nodes edges[i][0] and edges[i][1] together.
+    For a node x in the tree, let d(x) be the distance (the number of edges) from x to its farthest node. Find the min value of d(x) for the given tree.
+    The tree has the following properties:
+
+    It is connected.
+    It has no cycles.
+    For any pair of distinct nodes x and y in the tree, there's exactly 1 path connecting x and y.
+    Example 1:
+    Input: n = 6, edges = [[1, 4], [2, 3], [3, 4], [4, 5], [5, 6]]
+        1
+        |
+    2-3-4-5-6
+
+    Output: 2
+
+    Input: n = 2, edges = [[1, 2]]
+    Output: 1
+
+    Input: n = 10, edges = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
+    Output: 5
+
+
+    Imagine a spider with many legs, and each leg has multiple sections.
+
+    In each iteration, you're going to pull out the outermost section of each of its legs.
+
+    In the context of a graph, you're removing the leaf nodes (nodes with exactly 1 adjacent edge) at every iteration.
+
+    The number of iterations at the end (when you've finished pulling out all its legs) gives min([d(node) for node in graph])
+
+
+    from collections import defaultdict
+    from typing import List
+
+
+    def min_dist_to_furthest_node(n: int, edges: List[List[int]]) -> int:
+        """
+        Time  : O(N)
+        Space : O(N),
+        """
+
+        # SETUP THE GRAPH
+        g = defaultdict(list)
+
+        # SETUP A SET OF NODES WITH IN-DEGREE = 0
+        id0 = set([i for i in range(1, n + 1)])
+
+        # COUNT THE IN-DEGREE OF EACH NODE
+        id = [0] * (n + 1)
+
+        # TRACK THE DIST
+        dist = 0
+
+        for e in edges:
+            g[e[0]].append(e[1])
+            g[e[1]].append(e[0])
+            id[e[0]] += 1
+            id[e[1]] += 1
+            if id[e[0]] > 1 and e[0] in id0: id0.remove(e[0])
+            if id[e[1]] > 1 and e[1] in id0: id0.remove(e[1])
+
+        # LOOP TILL WE ONLY HAVE 0 - 1 NODE WITH ID = 0
+        while len(id0) > 1:
+
+            # TRACK THE NEW ID0
+            new_id0 = set()
+
+            # REMOVE ALL LEAVES AND THEIR EDGES
+            for leaf in id0:
+                for nb in g.get(leaf):
+                    id[nb] -= 1
+                    if id[nb] == 1: new_id0.add(nb)
+
+            id0 = new_id0
+            dist += 1
+
+        return dist
+
+
+27) Backtracking Example:
+    282. Expression Add Operators
+    Share
+    Given a string num that contains only digits and an integer target, return all possibilities to insert the binary 
+    operators '+', '-', and/or '*' between the digits of num so that the resultant expression evaluates to the target value.
+
+    Note that operands in the returned expressions should not contain leading zeros.
+
+    Example 1:
+
+    Input: num = "123", target = 6
+    Output: ["1*2*3","1+2+3"]
+    Explanation: Both "1*2*3" and "1+2+3" evaluate to 6.
+    Example 2:
+
+    Input: num = "232", target = 8
+    Output: ["2*3+2","2+3*2"]
+    Explanation: Both "2*3+2" and "2+3*2" evaluate to 8.
+    Example 3:
+
+    Input: num = "3456237490", target = 9191
+    Output: []
+    Explanation: There are no expressions that can be created from "3456237490" to evaluate to 9191.
+
+    Soln 1
+
+    class Solution:
+        def addOperators(self, num: 'str', target: 'int') -> 'List[str]':
+
+            def backtracking(idx=0, path='', value=0, prev=None):            
+                if idx == len(num) and value == target:
+                    rtn.append(path)
+                    return
+                
+                for i in range(idx+1, len(num) + 1):
+                    tmp = int(num[idx: i])
+                    if i == idx + 1 or (i > idx + 1 and num[idx] != '0'):
+                        if prev is None :
+                            backtracking(i, num[idx: i], tmp, tmp)
+                        else:
+                            backtracking(i, path+'+'+num[idx: i], value + tmp, tmp)
+                            backtracking(i, path+'-'+num[idx: i], value - tmp, -tmp)
+                            backtracking(i, path+'*'+num[idx: i], value - prev + prev*tmp, prev*tmp)
+            
+            rtn = []
+            backtracking()
+            
+            return rtn    
+
+
+
+    Soln 2 (Official Leetcode)
+
+    class Solution:
+        def addOperators(self, num: 'str', target: 'int') -> 'List[str]':
+
+            N = len(num)
+            answers = []
+            def recurse(index, prev_operand, current_operand, value, string):
+
+                # Done processing all the digits in num
+                if index == N:
+
+                    # If the final value == target expected AND
+                    # no operand is left unprocessed
+                    if value == target and current_operand == 0:
+                        answers.append("".join(string[1:]))
+                    return
+
+                # Extending the current operand by one digit
+                current_operand = current_operand*10 + int(num[index])
+                str_op = str(current_operand)
+
+                # To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
+                # valid operand. Hence this check
+                if current_operand > 0:
+
+                    # NO OP recursion
+                    recurse(index + 1, prev_operand, current_operand, value, string)
+
+                # ADDITION
+                string.append('+'); string.append(str_op)
+                recurse(index + 1, current_operand, 0, value + current_operand, string)
+                string.pop();string.pop()
+
+                # Can subtract or multiply only if there are some previous operands
+                if string:
+
+                    # SUBTRACTION
+                    string.append('-'); string.append(str_op)
+                    recurse(index + 1, -current_operand, 0, value - current_operand, string)
+                    string.pop();string.pop()
+
+                    # MULTIPLICATION
+                    string.append('*'); string.append(str_op)
+                    recurse(index + 1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
+                    string.pop();string.pop()
+            recurse(0, 0, 0, 0, [])    
+            return answers
+
+
+
+27.5) Top K elements
+        -> CAN BE SOLVED IN O(N) WITH BUCKET SORT, AND QUICK SELECT. CHECK IT OUT
+        -> TOP K MOST FREQUENT ELEMENTS QUESTION TO SEE THIS. 
+
+        Any problem that asks us to find the top/smallest/top most frequently occuring ‘K’ 
+        elements among a given set falls under this pattern.
+
+        The best data structure to keep track of ‘K’ elements is Heap. 
+        This pattern will make use of the Heap to solve multiple 
+        problems dealing with ‘K’ elements at a time from 
+        a set of given elements. The pattern looks like this:
+
+        Insert ‘K’ elements into the min-heap or max-heap based on the problem.
+
+        Iterate through the remaining numbers and if you find one that is 
+        larger than what you have in the heap, 
+        then remove that number and insert the larger one.
+
+        There is no need for a sorting algorithm because the heap will keep track of the elements for you.
+        How to identify the Top ‘K’ Elements pattern:
+        If you’re asked to find the top/smallest/frequent ‘K’ elements of a given set
+        If you’re asked to sort an array to find an exact element
+        Problems featuring Top ‘K’ Elements pattern:
+        Top ‘K’ Numbers (easy)
+        Top ‘K’ Frequent Numbers (medium)
+
+        # Top K Frequent Elements
+
+        class Solution(object):
+            def topKFrequent(self, nums, k):
+                """
+                :type nums: List[int]
+                :type k: int
+                :rtype: List[int]
+                """
+
+                num_of_items_to_return = k
+                m = collections.defaultdict(int)
+                
+                for i in nums:
+                    m[i] += 1
+
+                pq = [] # heapq
+                counter = itertools.count()
+                
+                # entry_finder = {} Used for deleting other elements in heapq!
+
+                for k, v in m.items():
+                
+                    if len(pq) < num_of_items_to_return:
+                        count = next(counter)
+                        i = [v, count, k] #[priority, count, task]
+                        heappush(pq, i)
+                    else:
+                        top =  pq[0][0] # get priority
+                        print("TOP IS", top)
+
+                        if v > top:
+                            _ = heappop(pq)
+                            count = next(counter)
+                            i = [v, count, k] #[priority, count, task]
+                            heappush(pq, i)     
+                return map(lambda x: x[-1], pq)
+
+        # BUCKET SOLN:
+        There are solution, using quickselect with O(n) complexity in average, but I think they are 
+        overcomplicated: actually, there is O(n) solution, using bucket sort. The idea, is that frequency 
+        of any element can not be more than n. So, the plan is the following:
+
+        Create list of empty lists for bucktes: for frequencies 1, 2, ..., n.
+        Use Counter to count frequencies of elements in nums
+        Iterate over our Counter and add elements to corresponding buckets.
+
+        buckets is list of lists now, create one big list out of it.
+
+        Finally, take the k last elements from this list, these elements will be top K frequent elements.
+
+        Complexity: time complexity is O(n), because we first iterate over nums once and create buckets, then we 
+        flatten list of lists with total number of elements O(n) and finally 
+        we return last k elements. Space complexity is also O(n).
+
+        class Solution:
+            def topKFrequent(self, nums, k):
+                bucket = [[] for _ in range(len(nums) + 1)]
+                Count = Counter(nums).items()  
+                for num, freq in Count: bucket[freq].append(num) 
+                flat_list = list(chain(*bucket))
+                return flat_list[::-1][:k]
+                

-64) Bomber DP or is the DP just precomputation below? you should check:
    (CAN DO WITH PRECOMPUTATION BUT LETS DO WITH DP!!!)
    
    Each cell in a 2D grid contains either a wall ('W') or an 
    enemy ('E'), or is empty ('0'). Bombs can destroy enemies, 
    but walls are too strong to be destroyed. A bomb placed in 
    an empty cell destroys all enemies in the same row and column, 
    but the destruction stops once it hits a wall.

    Return the maximum number of enemies you can destroy using one bomb.

    Note that your solution should have O(field.length · field[0].length) 
    complexity because this is what you will be asked during an interview.

    Example
    For
    field = [["0", "0", "E", "0"],
            ["W", "0", "W", "E"],
            ["0", "E", "0", "W"],
            ["0", "W", "0", "E"]]
    the output should be
    bomber(field) = 2.

    Sol'n A Easy (Cool Top Down):
        from functools import lru_cache
        def bomber(q):
            if not q or not q[0]:
                return 0
            a , b = len(q),len(q[0])
            @lru_cache(maxsize=None)
            def g(m,n,x,y):
                return 0 if m<0 or n<0 or m>=a or n>=b or q[m][n]=="W" \
                    else g(m + x,n + y,x,y)+(q[m][n]=="E")
            ans = 0
            for i in range(a):
                for j in range(b):
                    if q[i][j] == "0":
                        ans = max(ans,g(i-1,j,-1,0)+g(i,j-1,0,-1)+g(i+1,j,1,0)+g(i,j+1,0,1))
            return ans
    Soln B:
        def bomber(F):
            if not F or not F[0]         :   return 0
            row ,col = len(F) ,len(F[0]) ;   F = numpy.array(F)
            dp = numpy.zeros((row,col))  ;   t = zip(*numpy.where(F == 'E'))
            for x,y in t:
                for i in range(y-1,-1,-1):   
                    if F[x,i] == 'W'  :   break
                    if F[x,i] == '0' :   dp[x,i]+=1 
                for i in range(y+1,col):
                    if F[x,i] == 'W'  :   break
                    if F[x,i] == '0'  :   dp[x,i]+=1 
                for i in range(x-1,-1,-1):
                    if F[i,y] == 'W'  :   break
                    if F[i,y] == '0'  :   dp[i,y]+=1 
                for i in range(x+1,row):
                    if F[i,y] == 'W'  :   break
                    if F[i,y] == '0'  :   dp[i,y]+=1 
            return dp.max()

    Soln C:
        def bomber(A):
            from itertools import groupby
            if not A or not A[0]: return 0
            R, C = len(A), len(A[0])
            dp = [ [0] * C for _ in xrange(R) ]
            for r, row in enumerate(A):
                c = 0
                for k, v in groupby(row, key = lambda x: x != 'W'):
                    w = list(v)
                    if k:
                        enemies = w.count('E')
                        for c2 in xrange(c, c + len(w)):
                            dp[r][c2] += enemies
                    c += len(w)

            for c, col in enumerate(zip(*A)):
                r = 0
                for k, v in groupby(col, key = lambda x: x != 'W'):
                    w = list(v)
                    if k:
                        enemies = w.count('E')
                        for r2 in xrange(r, r + len(w)):
                            dp[r2][c] += enemies
                    r += len(w)
            
            ans = 0
            for r, row in enumerate(A):
                for c, val in enumerate(row):
                    if val == '0':
                        ans = max(ans, dp[r][c])
            return ans

28) K way Merge:

K-way Merge helps you solve problems that involve a set of sorted arrays.

        Whenever you’re given ‘K’ sorted arrays, you can use a
        Heap to efficiently perform a sorted traversal of all 
        the elements of all arrays. You can push the smallest 
        element of each array in a Min Heap to get the overall minimum. 
        After getting the overall minimum, push the next element 
        from the same array to the heap. Then, repeat this process 
        to make a sorted traversal of all elements.

        The pattern looks like this:
        Insert the first element of each array in a Min Heap.
        After this, take out the smallest (top) element from the heap and add it to the merged list.
        After removing the smallest element from the heap, insert the next element of the same list into the heap.
        Repeat steps 2 and 3 to populate the merged list in sorted order.
        How to identify the K-way Merge pattern:
        The problem will feature sorted arrays, lists, or a matrix
        If the problem asks you to merge sorted lists, find the smallest element in a sorted list.
        Problems featuring the K-way Merge pattern:
        Merge K Sorted Lists (medium)
        K Pairs with Largest Sums (Hard)

29) Questions you can solve with XOR can also probably be done with other operators such as +, -, *, /. Make
    sure you check for integer overflow. thats why xor method is always better.

29.5)  You are given an array A of n - 2 integers 
    which are in the range between 1 and n. All numbers 
    appear exactly once, except two numbers, which are 
    missing. Find these two missing numbers.

    Ok so go through range of numbers between 1 and n.
    XOR all those numbers,
    ex:

    def soln(ans):
        val = 0
        for i in range(0, N):
            val ^= (i+1)

        # then xor with ever number in ans. 
        for i in A:
            val ^= i

        # ok now val is   a ^ b where both a and b are different 
        # how to extract a, and b?
        '''

        you could run  a ^ b and try to xor it with 
        a number between 1 and n -> the result would be the other number, 
        b. 

        then check if the other number is between 1 and N and if it is,
        keep it. -> could you also check if the a and b you found is 
        also the same as the sum of the 2 missing numbers, which you can 
        get by subtracting N(n+1)/2 - sum(A). 
        so if it passes both tests then its more likely to be that rite!!
        but could 2 seperate 'a, 'b pairs still pass both sum and xor test?
        
        ABOVE SOLUTION IS HACKY!!
        '''

        BETTER SOLUTION: 
        # ok now val is   a ^ b where both a and b are different 
        # how to extract a, and b?
        '''
        well if the value is 10001. 

        The 0 means they were both either 0 bit, or both 1 bit.
        if its 1, then either the a has a 1 bit and b has 0 bit or 
        vice versa. 

        Partitioning based on inspecting u ^ v
        Luckily, we can figure out what to do by using what we 
        already stated earlier. Let’s think about this:

        If the two bits XOR takes as input are the same, the result is 0, otherwise it is 1.

        If we analyze the individual bits in u ^ v, then every 0 means that the 
        bit had the same value in both u and v. Every 1 means that the bits differed.

        Using this, we find the first 1 in u ^ v, i.e. the first position i where u and v 
        have to differ. Then we partition A as well as the numbers from 1 to n according to that bit.
        We end up with two partitions, each of which contains two sets:

        Partition 0
        The set of all values from 1 to n where the i-th bit is 0
        The set of all values from A where the i-th bit is 0
        Partition 1
        The set of all values from 1 to n where the i-th bit is 1
        The set of all values from A where the i-th bit is 1
        Since u and v differ in position i, we know that they have to be in different partitions.

        Reducing the problem
        Next, we can use another insight described earlier:

        While we worked on integers from 1 to n so far, this is not required. In fact, the 
        previous algorithm works in any situation where there is (1) some set of potential 
        elements and (2) a set of elements actually appearing. The sets may only differ 
        in the one missing (or duplicated) element.

        These two sets correspond exactly to the sets we have in each partition. 
        We can thus search for u by applying this idea to one of the partitions 
        and finding the missing element, and then find v by applying it to the other partition.

        This is actually a pretty nice way of solving it: We effectively 
        reduce this new problem to the more general version of the problem we solved earlier.


29.6)  Use loop invarients when doing 2 pointer solutions, greedy solutions, etc. to think about, and help
    interviewer realize that your solution works!!!

29.7)  Derieive mathematical relationships between numbers in array, and solve for a solution. Since
    there was a mathematical relationship, xor can prolly be used for speedup. 
    For instance: Find the element that appears once

        Given an array where every element occurs three times, except one element which occurs only once. 

        SHITTY Soln: Add each number once and multiply the sum by 3, we will get thrice the sum of each 
        element of the array. Store it as thrice_sum. Subtract the sum of the whole array 
        from the thrice_sum and divide the result by 2. The number we get is the required 
        number (which appears once in the array).
        How do we add each number once though? we cant use a set. 
        XOr? wtf?

        
        CORRECT SOLN:

        Use two variables, ones and twos, to track the bits that appear an odd and 
        even number of times, respectively. In each iteration, XOR the current element 
        with ones to update ones with the bits that appear an odd number of times then 
        use a bitwise AND operation between ones and the current element to find the common 
        bits that appear three times. These common bits are removed from both ones and twos 
        using a bitwise AND operation with the negation of the common bits. 
        Finally, ones contains the element that appears only once.

        # Python3 code to find the element that 
        # appears once

        def getSingle(arr, n):
            ones = 0
            twos = 0
            
            for i in range(n):
                # one & arr[i]" gives the bits that
                # are there in both 'ones' and new
                # element from arr[]. We add these
                # bits to 'twos' using bitwise XOR
                twos = twos ^ (ones & arr[i])
                
                # one & arr[i]" gives the bits that
                # are there in both 'ones' and new
                # element from arr[]. We add these
                # bits to 'twos' using bitwise XOR
                ones = ones ^ arr[i]
                
                # The common bits are those bits 
                # which appear third time. So these
                # bits should not be there in both 
                # 'ones' and 'twos'. common_bit_mask
                # contains all these bits as 0, so
                # that the bits can be removed from
                # 'ones' and 'twos'
                common_bit_mask = ~(ones & twos)
                
                # Remove common bits (the bits that 
                # appear third time) from 'ones'
                ones &= common_bit_mask
                
                # Remove common bits (the bits that
                # appear third time) from 'twos'
                twos &= common_bit_mask
            return ones


29.8)  DP is like traversing a DAG. it can have a parents array, dist, and visited set. SOmetimes you need to backtrack
    the DP updates to retrieve parents so remember how to do that!!!!. 

29.9)  Do bidirectional BFS search if you know S and T and you are finding the path! 
    (i think its good for early termination in case there is no path)

30)  For linked list questions, draw it out. Dont think about it. Then figur eout how you are rearranging the ptrs.
    and how many new variables you need. ALSO USE DUMMY POINTERS to not deal with modifying head pointer case. 


31)  Linear Algorithms:
    Bracket Matching => Use stack
    Postfix Calculator and Conversion
        Prefix calculator => 2 + 6 * 3 => this needs binary tree to do i think! with extra mem
        Prefix: + 2 * 6 3, * + 2 6 3
        Postfix: 2 6 3 * +, 2 6 + 3 *
        We can evaluate postfix in O(n). Push elements in stack. when you see operator, 
        pop 2 elements right, do compute, put back into stack.

    (Static) selection problem
        Given unchanged array of n elements. can we find kth smallest element of A in O(n). yeah prlly
        A = {2, 8, 7, 1, 5, 4, 6, 3} 
        4th smallest is 4. 
        4 solutions: 
        sort and get k-1 element O(nlogn)
        Do heap-select USING MIN HEAP. create min heap of given n element and call extractMin() k times.
            O(n + kLogn) => because heapify is O(n)
        Method 3 (Using Max-Heap)
            We can also use Max Heap for finding the k’th smallest element. Following is algorithm.
            1) Build a Max-Heap MH of the first k elements (arr[0] to arr[k-1]) of the given array. O(k)

            2) For each element, after the k’th element (arr[k] to arr[n-1]), compare it with root of MH.
            ……a) If the element is less than the root then make it root and call heapify for MH
            ……b) Else ignore it.
            // The step 2 is O((n-k)*logk)

            1) Finally, root of the MH is the kth smallest element.

            Time complexity of this solution is O(k + (n-k)*Logk)

        METHOD 4(BEST METHOD QUICK SELECT): -> DO MEDIAN OF MEDIANS TO GET O(N) NOT WORST AVERAGE CASE? worst time!!!
           The idea is, not to do complete quicksort, but stop at the point where pivot itself is k’th         
            smallest element. Also, not to recur for both left and right sides of pivot, 
            but recur for one of them according to the position of pivot. The worst case time          
            complexity of this method is O(n^2), but it works in O(n) on average.

    Sorting in linear time
        Given array, each int between [0, 100] can we sort in O(n). yeah counting sort.
        What if given arry has range of 32 bit unsigned ints [0, 2^32 -1] => radix sort

    Sliding window
        -> Given an array of n elements, can we find a smallest sub-array size so that the sum of the sub-array is greater 
        than or equal to a constant S in O(n)
        
        2 pointers both start at index 0. move end pointer 
        to right until you have S. then keep that as current_min_running_length_of_subarray.
        move start pointer to right to remove elements, then fix by extending end pointer if sum falls below S. 
        get new subarrays and update current_min_running_length_of_subarray. 


31.5)

        Bloomberg - I was asked to design a class which replicated browser history. I tried implementing it with dictionaries, 
        but still could not meet O(1) complexity. This was nothing but a modification of LRU Cache. I 
        was uncomfortable with implementing linked list in python, the whole idea of linked list and its 
        addresses seemed daunting to me. Weakpoints - Linked List, Pointers

        Simon LLC - Design a HashMap. Started by creating buckets, but failed to explain hash collision, 
        and hash function. Intuitively, was near to the correct explanation, but thinking it through and 
        cohesively putting it in words under a time ticking clock became challenging. Weakpoints - 
        Time Management, Hash Collision, Internal Working of HashMap

        Palantir - A forest was given, had to find nodes which had only one parent. Follow up questions, 
        like finding ancestor, and farthest ancestor. Weakpoints - Complexity, I had used a set in the program, 
        and gave complexity as O(1), but actually in the program, from the set I had created a list, and the 
        actual time complexity should’ve been O(n).

        Microsoft - 30 min coding interview, Inorder successor in a binary search tree, this was on 
        the tips with recursion, but when they asked me to do it in iterative manner, 
        I struggled, and realized, I still lacked clarity on few concepts. Weakpoints - Iterative tree traversals

        Doordash - Final 2 hour coding round. Each round had one question each. In the first round 
        was asked to design a in design memory file system. In another round, was asked to design 
        auto complete suggestion system. Both were applications of trie. First round went smooth, 
        second round had a few hiccups, I was unable to handle the edge 
        cases. Weakpoints - Critical thinking to accommodate edge cases

        My other weakpoint that I had discovered was tackling behavorial questions. 
        It wasn't until, I started answering using the STAR method that my answers became better. 
        There were some recruiter rounds where they asked me behavorial questions, 
        and I was unprepared, hadn't researched much about the company, and it costed 
        me opportunities. A great link to a behavioral guide can be 
        found here https://leetcode.com/discuss/interview-question/1729926/a-guide-for-behavioral-round

        All these interviews were making me more aware of my weakness, at the heavy cost of losing 
        out on great opportunities. While my friends had started interviewing for Facebook, 
        I was even skeptical of applying. A recruiter had reached out to me, and I took a leap of faith, and applied.

32)  Heapify is cool. Python heapify implementation that is O(N) implemented below: 
    UNDERSTAND IT.
        # Single slash is simple division in python. 2 slashes is floor division in python
        # only the root of the heap actually has depth log2(len(a)). Down at the nodes one above a leaf - 
        # where half the nodes live - a leaf is hit on the first inner-loop iteration.

        def heapify(A):
            for root in xrange(len(A)//2-1, -1, -1):
                rootVal = A[root]
                child = 2*root + 1
                while child < len(A):
                    # we pick the smaller child to sort?
                    # makes sense because the smaller child is the one
                    # that has to fight with the parent in a min heap.
                    if child+1 < len(A) and A[child] > A[child+1]:
                        child += 1
                    if rootVal <= A[child]:
                        break
                    A[child], A[(child-1)//2] = A[(child-1)//2], A[child]
                    child = child *2 + 1

33)  Understand counting sort, radix sort.
        Counting sort is a linear time sorting algorithm that sort in O(n+k) 
        time when elements are in range from 1 to k.        
        What if the elements are in range from 1 to n2? 
        We can’t use counting sort because counting sort will take O(n2) which is worse 
        than comparison based sorting algorithms. Can we sort such an array in linear time?
        Radix Sort is the answer. The idea of Radix Sort is to do digit by digit 
        sort starting from least significant digit to most significant digit. 
        Radix sort uses counting sort as a subroutine to sort.
        Look at section below for impls.


34)  To do post order traversal or inorder traversal 
    on a binary tree iteratively (or doing any dfs, where you want to vist root node last). 
    you need to USE A FLAG!! (LOOK at morris traversal for cool funs!)

        def postorderTraversal(self, root):
            traversal, stack = [], [(root, False)]
            while stack:
                node, visited = stack.pop()
                if node:
                    if visited:
                        # add to result if visited
                        traversal.append(node.val)
                    else:
                        # post-order
                        stack.append((node, True))
                        stack.append((node.right, False))
                        stack.append((node.left, False))

            return traversal

        def inorderTraversal(self, root):
            result, stack = [], [(root, False)]

            while stack:
                cur, visited = stack.pop()
                if cur:
                    if visited:
                        result.append(cur.val)
                    else:
                        stack.append((cur.right, False))
                        stack.append((cur, True))
                        stack.append((cur.left, False))

            return result
        
        def preorderTraversal(self, root):
            ret = []
            stack = [root]
            while stack:
                node = stack.pop()
                if node:
                    ret.append(node.val)
                    stack.append(node.right)
                    stack.append(node.left)
            return ret


35) When you need to keep a set of running values such as mins, and prev mins, 
    you can keep all the runnign mins in a datastructre and as your algorithm traverses the datastructure, 
    update the datastructure for the running value as well in the same way to maintaing consistency!
    For instance, min stack: 
    Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

    push(x) -- Push element x onto stack.
    pop() -- Removes the element on top of the stack.
    top() -- Get the top element.
    getMin() -- Retrieve the minimum element in the stack.

    have 2 stacks, one for the actual stack, and another that simulates the same operations to 
    maintain the running min.


36) In some questions you can 
    do DFS or BFS from a root node to a specific 
    child node and you end up traversing a tree, 
    either the DFS tree or BFS Tree. 
    HOWEVER, AS AN O(1) space optimization, you might be able 
    to go backward from the child node to the root node,
    and only end up traversing a path rather than a tree!
    Go up the tree for SPEED. 


+37) In head recursion , the recursive call, when it happens, comes 
+      before other processing in the function (think of it happening at the top, 
+      or head, of the function). In tail recursion , it's the 
+      opposite—the processing occurs before the recursive call.


40.1) Can create queue with 2 stacks


40.2) Heap
    -> Given a node k, easy to compute indices for
        parent and children
        - Parent Node: floor(k/2)
        - Children: 2k, 2k+1
    You must refer to the definition of a Binary Heap:

        A Binary heap is by definition a complete binary tree ,that is, all levels of the       
        tree, 
        except possibly the last one (deepest) are fully filled, and, if the last     
        level of the tree is not complete, the nodes of that level are filled from left to        
        right.
        
        It is by definition that it is never unbalanced. The maximum difference in balance      
        of the two subtrees is 1, when the last level is partially filled with nodes only       
        in the left subtree.

    -> To insert node: (running time O(lgn)) (SIFT UP)
    1) make a new node at last level as far left as possible. 
    2) if node breaks heap property, swap with its parent node. 
    3) new node moves up tree. repeat 1) to 3) until all conflicts resolved.

    -> Deleting root node: (SIFT DOWN)
    1) Remove the root, and bring the last node (rightmost node 
    in the last leve) to the root. 
    2) If the root breaks the heap property, look at its children
    and swap it with the larger one. 
    3) Repeat 2 until all conflicts resolved


41) FORD FULKERSON ALGORITHM PYTHON (MAX FLOW MIN CUT):
    class Graph: 
    
        def __init__(self,graph): 
            self.graph = graph # residual graph 
            self. ROW = len(graph)             
    
        '''Returns true if there is a path from source 's' to sink 't' in 
        residual graph. Also fills parent[] to store the path '''
        def BFS(self,s, t, parent): 
    
            # Mark all the vertices as not visited 
            visited =[False]*(self.ROW) 
            queue=[] 
            queue.append(s) 
            visited[s] = True

            while queue: 
    
                #Dequeue a vertex from queue and print it 
                u = queue.pop(0) 
            
                # Get all adjacent vertices of the dequeued vertex u 
                # If a adjacent has not been visited, then mark it 
                # visited and enqueue it 
                for ind, val in enumerate(self.graph[u]): 
                    if visited[ind] == False and val > 0 : 
                        queue.append(ind) 
                        visited[ind] = True
                        parent[ind] = u 

            return True if visited[t] else False
                      
        # Returns tne maximum flow from s to t in the given graph 
        def FordFulkerson(self, source, sink): 
    
            # This array is filled by BFS and to store path 
            parent = [-1]*(self.ROW) 
    
            max_flow = 0 # There is no flow initially 
    
            # Augment the flow while there is path from source to sink 
            while self.BFS(source, sink, parent) : 
    
                # Find minimum residual capacity of the edges along the 
                # path filled by BFS. Or we can say find the maximum flow 
                # through the path found. 
                path_flow = float("Inf") 
                s = sink 

                # Get the limiting flow in the path. Traverse parents array to get path
                while(s !=  source): 
                    path_flow = min (path_flow, self.graph[parent[s]][s]) 
                    s = parent[s] 
    
                # Add path flow to overall flow 
                max_flow +=  path_flow 
    
                # update residual capacities of the edges and reverse edges along the path 
                v = sink 
                while(v !=  source): 
                    u = parent[v] 
                    self.graph[u][v] -= path_flow 
                    self.graph[v][u] += path_flow 
                    v = parent[v] 
    
            return max_flow 

    graph = [[0, 16, 13, 0, 0, 0], 
            [0, 0, 10, 12, 0, 0], 
            [0, 4, 0, 0, 14, 0], 
            [0, 0, 9, 0, 0, 20], 
            [0, 0, 0, 7, 0, 4], 
            [0, 0, 0, 0, 0, 0]]    
    g = Graph(graph) 
    source = 0; sink = 5
    print ("The maximum possible flow is %d " % g.FordFulkerson(source, sink)) 
  
    Output:
    The maximum possible flow is 23


+41) AVOID LINKED LIST LOOPS IN YOUR CODE. ALWAYS 
+       NULLIFY YOUR POINTERS IF YOU ARE REUSING THE 
+       DATASTRUCTURE/ DOING THINGS IN PLACE!!!!!!
+       SUCH AS HERE by saving nxt pointer as tmp
+
+       1.   Odd Even Linked List
+        Given the head of a singly linked list, group all the nodes 
+        with odd indices together followed by the nodes with even 
+        indices, and return the reordered list.
+
+        The first node is considered odd, and the second node is even, and so on.
+
+        Note that the relative order inside both the even and odd
+        3groups should remain as it was in the input.
+
+        You must solve the problem in O(1) extra space complexity and O(n) time complexity.
+   
+       You should try to do it in place. The program should run in O(1)    
+       space complexity and O(nodes) time complexity.
+
+        def oddEvenList(self, head: ListNode) -> ListNode:
+            oddH = ListNode(0)
+            evenH = ListNode(0)
+            
+            odd = oddH
+            even = evenH
+            
+            isOdd = True
+            node = head
+            
+            while node:
+                nxt = node.next
+                node.next = None # IMPORTANT STOP THE LOOPS
+                if isOdd:
+                    odd.next = node
+                    odd = odd.next
+                    isOdd = False
+                else:
+                    even.next = node
+                    even = even.next
+                    isOdd = True
+                node = nxt
+            
+            odd.next = evenH.next
+            return oddH.next


41.1) Flatten binary tree to linked list. 
+     Given a binary tree, flatten it to a linked list in-place.
+     Use right nodes when creating linked list. 
+     CAN DO THIS WITH O(1) SPACE LIKE SO:
+  
+     So what this solution is basically doing is putting the 
+     right subtree next to the rightmost node on the left subtree 
+     and then making the left subtree the right subtree and 
+     then making the left one null. Neat!
+     
+    class Solution:
+        # @param root, a tree node
+        # @return nothing, do it in place
+        def flatten(self, root):
+            if not root:
+                return
+            
+            # using Morris Traversal of BT
+            node=root
+            
+            while node:
+                if node.left:
+                    pre=node.left
+                    while pre.right:
+                        pre=pre.right
+                    pre.right=node.right
+                    node.right=node.left
+                    node.left=None
+                node=node.right
+
+
41.2) Flattening a multilevel doubly linked list using a stack:
+        def flatten(self, head):
+            if not head:
+                return
+            
+            dummy = Node(0,None,head,None)     
+            stack = []
+            stack.append(head)
+            prev = dummy
+            
+            while stack:
+                root = stack.pop()
+
+                root.prev = prev
+                prev.next = root
+                
+                if root.next:
+                    stack.append(root.next)
+                    root.next = None
+                if root.child:
+                    stack.append(root.child)
+                    root.child = None
+                prev = root        
+                
+            # disengage dummy node
+            dummy.next.prev = None
+            return dummy.next


+-68) Counting clouds by removing and growing as an alternative DFS:
+
+    Given a 2D grid skyMap composed of '1's (clouds) and '0's (clear sky), 
+    count the number of clouds. A cloud is surrounded by clear sky, and is 
+    formed by connecting adjacent clouds horizontally or vertically. 
+    You can assume that all four edges of the skyMap are surrounded by clear sky.
+
+    Example
+
+    For
+
+    skyMap = [['0', '1', '1', '0', '1'],
+              ['0', '1', '1', '1', '1'],
+              ['0', '0', '0', '0', '1'],
+              ['1', '0', '0', '1', '1']]
+    the output should be
+    countClouds(skyMap) = 2;
+    
+    
+    def countClouds(skyMap):
+        if not skyMap or not skyMap[0]:
+            return 0
+        m, n = len(skyMap), len(skyMap[0])
+        ones = {(i, j) for i in range(m) for j in range(n) if skyMap[i][j] == '1'}
+        cc = 0
+        while ones:
+            active = {ones.pop()}
+            while active:
+                ones -= active
+                nxt_active = set()
+                for x, y in active:
+                    for dx, dy in ((-1,0), (1,0), (0,-1), (0,1)):
+                        if 0 <= x+dx < m and 0 <= y + dy < n and \
+                            (x+dx, y+dy) in ones:
+                            nxt_active.add((x+dx, y+dy))
+                active = nxt_active
+            cc += 1
+        return cc
+



+2 - Given an array of integers, find the subarray with maximum XOR. 
+
+        Think of cumulatives and starting from beginning simialar to above problem. 
+
+        Similar to previous problem:
+        Cool XOR trick to solve problem:
+        -> F(L, R) is XOR subarray L to R
+        F(L, R) = F(1, R) ^ F(1, L-1)
+    */
+


+-67) Lazy updates to build faster data structures (aka min stack extended ):
+
+        Similar hill finding question from IMC oa: 
+        Techniques: for stack 2.0, where we create a stack
+        but we can also increment alll elements below index i 
+        by a value
+        
+        -> implement push, pop, increment(index i, value v)
+        you use 2 stacks, and we do LAZY updates. Similar to min stack.
+        When we access an element that should have been increment. 
+        add stack value + increment stack value. 
+        When we increment we only save it at index i. not [0...i] with for loop
+        to do O(1) lookup, push, pop, increment. And when we pop that index,
+        assign to index i-1.
+        
+        THE IDEA IS: -> look at the very specific constraints of problem and solve 
+        for only what it is asking. nothing more (which allows you to simplify and 
+        improve solutions).
+        
+        Try to solve by being as LAZY as possible, and keeping track of critical indexes. 
+        Do it similar to how you as a lazy human would solve it IRL. 
+        
+        By waiting to do operations until it is necessary -> and being GREEDY and smart 
+        about how to update the state of the problem for only the next state[and just the next state], 
+        and not all states, we optimized stack 2.0. 
+
+        IMPLEMENTATION OF SUPER STACK:
+    
+        def superStack(operations):
+            stack = []
+            inc = []
+            result = []
+            '''
+            Save and propogate lazy updates using inc[]
+            based on how we access stack 
+            '''
+            for op in operations:
+                
+                items = op.split()
+                cmd = items[0]  
+                if cmd == "push":
+                    stack.append(int(items[1]) )
+                    inc.append(0)
+                elif cmd == "pop":
+                    if len(stack) > 0:
+                        stack.pop()
+                        poppedInc = inc.pop()
+                        if len(inc) > 0:
+                            inc[-1] += poppedInc
+                elif cmd == "inc":
+                    # inc 2 2
+                    pos, val = int(items[1]), int(items[2])
+                    inc[pos-1] += val
+                
+                if len(stack) > 0:
+                    print(stack[-1] + inc[-1])
+                else:
+                    print("EMPTY")

+-66)  Hill finding w/ stacks and queues and lazy updates in data structures: 
+
+        '''
+        Given an array a composed of distinct elements, find 
+        the next larger element for each element of the array, i.e. 
+        the first element to the right that is greater than this element, 
+        in the order in which they appear in the array, and return the 
+        results as a new array of the same length. If an element does 
+        not have a larger element to its right, put -1 in the 
+        appropriate cell of the result array.
+
+        Example
+
+        For a = [6, 7, 3, 8], the output should be
+        nextLarger(a) = [7, 8, 8, -1]
+
+        '''
+        # use queue. 
+        '''
+        HILL FINDING WITH CRITICAL INDEXES + LAZINESS LECTURE.  
+        KEEP TRACK OF KEY POINTS ONLY IN QUEUE/STACK. 
+        NO WASTE IN QUEUE, JUST WHAT WE NEED. 
+        AKA hill finding.         
+        '''
+
+        def nextLarger(a):        
+            st = []
+            res = []
+
+            for i in range(len(a)-1, -1, -1):
+                val = a[i]
+                while len(st) > 0:
+                    if a[i] > st[-1]:
+                        st.pop()
+                    else:
+                        break     
+                if len(st) == 0:
+                    res.append(-1)
+                else:
+                    res.append(st[-1])
+                st.append(val)
+            return res[::-1]



+-65) REGEX REVIEW USAGE:
+
+    You categorize strings into three types: good, bad, or mixed. If a string has 
+    3 consecutive vowels or 5 consecutive consonants, or both, then it is categorized 
+    as bad. Otherwise it is categorized as good. Vowels in the English alphabet are 
+    ["a", "e", "i", "o", "u"] and all other letters are consonants.
+
+    The string can also contain the character ?, which can be replaced by either a 
+    vowel or a consonant. This means that the string "?aa" can be bad if ? is a 
+    vowel or good if it is a consonant. This kind of string is categorized as mixed.
+
+    Implement a function that takes a string s and returns its category: good, bad, or mixed.
+
+    def classifyStrings(s):
+        if re.search(r"[aeiou]{3}|[^aeiou?]{5}", s):
+            return "bad"
+        if "?" not in s:
+            return "good"
+        a = classifyStrings(s.replace("?", "a", 1))
+        b = classifyStrings(s.replace("?", "b", 1))
+        return "mixed" if a != b else a


-78) Maximal Square DP - DP vs cumulative array strategy?
    
    You have a 2D binary matrix that's filled with 0s and 1s. 
    In the matrix, find the largest square that 
    contains only 1s and return its area.

    NOTES:
        When a problem looks like a cumulative array problem try other accumulations,
        rather than sum, such as 2d segment trees, or 2d maximum slice accumations.

        In this probem -> we did our accumulation based on 3 other coordinates in matrix. 
        Up your preprocessing game
        ALWAYS USE THE DIAGONAL SOMEHOW IN 2D ARRAYS + DONT FORGET TOP-LEFT COORDINATE.
    
    SOLUTION:
        def maximalSquare(matrix):
            
            '''
            then do maximal rectangle. 
            Go right and go down. 
            question -> how many 1's below me?
            
            1 1 1 1
            1 2 2 2 
            1 2 3

            Recurrence:
            dp(i,j) = min(dp(i−1, j), dp(i−1, j−1), dp(i, j−1)) + 1

            BASE CASE: 
            matrix[i,j] == '0' THEN return 0        
            '''
            R = len(matrix)
            if R == 0:
                return 0
            C = len(matrix[0])
            prevRow = [0 for j in range(C+1)]
            maxSquare = 0
            for i in range(R):
                # we have to zero pad. 
                currRow = [0]
                
                for j in range(1, C+1):
                    # if current value is 0, put 0.
                    val = matrix[i][j-1]
                    if val == "0":
                        currRow.append(0)
                    else:
                        minOfTopAndLeft = min(currRow[-1], prevRow[j-1], prevRow[j])
                        cellVal = minOfTopAndLeft + 1
                        maxSquare = max(maxSquare, cellVal**2)
                        currRow.append(cellVal)
                        
                prevRow = currRow[::]
            return maxSquare
            





-77) Painted Ladies BACKWARD DP

    In San Francisco, there is a row of several beautiful houses called 
    the Painted Ladies. Each of the Painted Ladies can be painted with 
    one of three colors: red, blue or green. The cost of painting each 
    house with a certain color is different. cost[i][0] for each i is 
    the cost of painting house i red, cost[i][1] is the cost of painting 
    it blue, and cost[i][2] is the cost of painting it green.

    You want to paint all the houses in a way such that no two adjacent 
    Painted Ladies have the same color. Find the minimum cost to achieve this.

    Example

    For cost = [[1, 3, 4], [2, 3, 3], [3, 1, 4]], the output should be
    paintHouses(cost) = 5.

    def paintHouses(cost):
        
        '''
        recurrence 
        OPT[i, color] = minimum cost as a result of choosing a specific color. 
        # compute all three! -> BACKWARD DP. 
        OPT[i, Blue] = min(OPT[i-1, RED], OPT[i-1, GREEN])
        OPT[i, RED] =  min(OPT[i-1, BLUE], OPT[i-1, GREEN])
        OPT[i, GREEN] =  min(OPT[i-1, BLUE], OPT[i-1, RED])
        answer is min(of all colors OPT[i])
        
        recursive
        fn(idx, prev_color)
            we know prev color -> choose other 2 colors. 
            take min of choosing either color!
        
        Space optimize to 3 variables!        
        '''
        opt_b, opt_r, opt_g = cost[0][0], cost[0][1], cost[0][2]
        IDX_b, IDX_r, IDX_g = 0, 1, 2
        
        for i in range(1, len(cost)):
            blue_cost = cost[i][IDX_b]
            red_cost = cost[i][IDX_r]
            green_cost = cost[i][IDX_g]
            
            opt_b, opt_g, opt_r = \
                min(opt_r, opt_g) + blue_cost, min(opt_r, opt_b) + green_cost, min(opt_b, opt_g) + red_cost  
            
        return min(opt_b, opt_g, opt_r)




-76) Linked Lists, 2 Pointers and simplifying problems by  respecting   
     OPEN-CLOSE 2 pointers which satisfy a <= b < c aka [X, Y) for start and end. 

    Given a singly linked list of integers l and a non-negative integer n, 
    move the last n list nodes to the beginning of the linked list.

    Example

    For l = [1, 2, 3, 4, 5] and n = 3, the output should be
    rearrangeLastN(l, n) = [3, 4, 5, 1, 2];
    For l = [1, 2, 3, 4, 5, 6, 7] and n = 1, the output should be
    rearrangeLastN(l, n) = [7, 1, 2, 3, 4, 5, 6].

    HARMAN SOLUTION WHICH USES 2POINTERS that refer to [start, end]
    problem is both pointers can point to same node so this case 
    has to be handled seperately!! + other edge cases.
    
        def rearrangeLastN(l, n):     
            # use 2 pointers that occupy n space. 
            # go to the  second last element. do you know why? 
            # because we have to set None to the element we are 
            # splitting from. 
            i = l 
            j = l
            
            if l is None:
                return None
            if n == 0:
                return l
                
            # n-1 spaces between n nodes
            for _ in range(n-1):
                j = j.next
            
            # the whole list was chosen as n. 
            if j.next == None:
                return l
            
            # second last.
            while j and j.next and j.next.next:
                i = i.next
                j = j.next
            
            # get last node. 
            j.next.next = l
            
            # end
            newStart = i.next            
            # SET THE NULLS AT THE END BECAUSE WE CAN 
            # BREAK LINKED LIST FUNCTIONALITY
            # IF BOTH POINTERS POINT AT SAME NODE!
            i.next = None
            return newStart

    OPEN CLOSE NOTATION SOLUTION CLEANNN:

        def rearrangeLastN(l, n):
            if n == 0:
                return l
            front, back = l, l
            for _ in range(n):
                front = front.next
            if not front:
                return l
            while front.next:
                front = front.next
                back = back.next
            out = back.next
            back.next = None
            front.next = l
            return out



+-87)Optimizing binary tree questions with bottom up DP: 
+    One way to optimize these questions is to use post-order traversal.
+    Compute the value for the children then compute for parent sorta like DP:
+
+    1.   Count Univalue Subtrees
+    中文English
+    Given a binary tree, count the number of uni-value subtrees.
+    
+    A Uni-value subtree means all nodes of the subtree have the same value.
+    
+    Example
+    Example1
+    
+    Input:  root = {5,1,5,5,5,#,5}
+    Output: 4
+    Explanation:
+                  5
+                 / \
+                1   5
+               / \   \
+              5   5   5
+    Example2
+    
+    Input:  root = {1,3,2,4,5,#,6}
+    Output: 3
+    Explanation:
+                  1
+                 / \
+                3   2
+               / \   \
+              4   5   6
+
+    Solution:
+    def countUnivalSubtrees(self, root):
+        count = 0
+        def helper(node):
+            nonlocal count 
+            if node is None:
+                return None
+            left_result = helper(node.left)
+            right_result = helper(node.right)
+            if left_result == False:
+                return False
+            if right_result == False:
+                return False
+            if left_result and left_result != node.val:
+                return False
+            if right_result and right_result != node.val:
+                return False
+            count += 1
+            return node.val
+        helper(root)
+        return count



+
+-86.5)  HRT: 1775. Equal Sum Arrays With Minimum Number of Operations
+
+        You are given two arrays of integers nums1 and nums2, possibly of different lengths. 
+        The values in the arrays are between 1 and 6, inclusive.
+
+        In one operation, you can change any integer's value in any of the arrays 
+        to any value between 1 and 6, inclusive.
+
+        Return the minimum number of operations required to make the sum of values in nums1 
+        equal to the sum of values in nums2. Return -1​​​​​ if it is not possible 
+        to make the sum of the two arrays equal.
+
+            static const auto speedup = []() { std::ios::sync_with_stdio(false); std::cin.tie(nullptr); return 0; }();
+
+            class Solution {
+            public:
+                int minOperations(vector<int>& nums1, vector<int>& nums2) {
+                    // calculate the frequency and find the sum
+                    vector<int> cnt1(7,0), cnt2(7,0);
+                    for(auto &a:nums1) cnt1[a]++;
+                    for(auto &a:nums2) cnt2[a]++;
+                    int sum1 = accumulate(nums1.begin(), nums1.end(), 0);
+                    int sum2 = accumulate(nums2.begin(), nums2.end(), 0);
+                    
+                    // already equal
+                    if(sum1 == sum2) return 0;
+                    
+                    // reduce sum1 < sum2 to sum1 > sum2....so only one problem we have to deal wiith
+                    if(sum1 < sum2){
+                        swap(sum1, sum2);
+                        cnt1.swap(cnt2);
+                    }
+                    
+                    // since sum1 > sum2 ... we have two options 
+                            // 1) decrease val in first array 
+                            // 2) increase val in second array
+                    
+                    // if we have 6 in first array, maximum decrease of sum1 can be of 5 So, sum1-sum2 decreases by 5 
+                    // Similarly if we have 1 in second array, maximum incrase of sum2 can be 5. And hence sum1-sum2 decreases by 5. 
+                    // So, decreasing 6 in first array and increasing 1 in second lead to maximum deduction of 5 in the difference.
+                    
+                    // Similary, If we consider 5 from first array and 2 from the second array, 
+                    // it can lead to maximum deduction of 4 in the difference.
+                    
+                    // Now, cnt1[i] (i=2,3,4,5,6) will have the number of times it can decrese the difference by (i-1);
+                    for(int i=1; i<=6; i++)
+                        cnt1[i] += cnt2[7-i];
+                    
+                    int diff = sum1-sum2;
+                    int curr = 6;      // start by i=6 and go upto i=2;
+                    int ops = 0;        // to store the # of operations
+                    
+                    while(diff && (curr>1)){
+                        // count the # of substraction needs to be done where the substraction can be of (curr-1) 
+                        int needed = ceil(1.0*diff/(curr-1));
+                        
+                        // maximum operations is bounded by count of the (curr-1) substraction
+                        // As stated earlier cnt1[curr] have the count of how many (curr-1) deduction can be done
+                        // So, we can only do min(needed, cnt1[curr-1]) ops
+                        ops += min(needed, cnt1[curr]);
+                        
+                        // deacrese the difference accordingly
+                        diff -= min(needed, cnt1[curr])*(curr-1);
+                        
+                        // for last deduction diff can be -ve. E.g. diff was 3 and we deduct 5. So, we can assume that we deducted 3 only and make diff = 0
+                        if(diff < 0) diff = 0;
+                        
+                        // go for next smaller value
+                        curr--;
+                    }
+                    
+                    // if diff is non-zero, then return -1. Otherwise return the # of operations
+                    return (diff ? -1 : ops);
+                }
+            };
+
+
+
+-40) CIRCULAR BUFFERS: Implementation Stack and Queue 
+     Declare an array with enough space!
+     
+    7.1: Push / pop function — O(1).
+    1   stack = [0] * N
+    2   size = 0
+    3   def push(x):
+    4       global size
+    5       stack[size] = x
+    6       size += 1
+    7   def pop():
+    8       global size
+    9       size -= 1
+    10      return stack[size]
+
+    7.2: Push / pop / size / empty function — O(1).
+    1   queue = [0] * N
+    2   head, tail = 0, 0
+    3   def push(x):
+    4       global tail
+    5       tail = (tail + 1) % N
+    6       queue[tail] = x
+    7   def pop():
+    8       global head
+    9       head = (head + 1) % N
+    10      return queue[head]
+    11  def size():
+    12      return (tail - head + N) % N
+    13  def empty():
+    14      return head == tail
+
+-86) monotonic stack vs monotonic queue and how to build a monotonic structure
+    HRT PROBLEM: MINIMIZE THE AMPLITUDE!:
+    Given an array of N elements, remove K elements to minimize the amplitude(A_max - A_min) of the remaining array.
+
+    remove k consecutive elements from A such that amplitude is minimal which is the 
+
+
+        #include <vector>
+        #include <deque>
+        #include <bits/stdc++.h>
+        #include <iostream> 
+
+        using namespace std;
+
+
+        struct Item {
+            int idx;
+            int val;
+        };
+
+
+        int solution(vector<int> &A, int K) {
+
+            /*
+            min amplitude.
+
+            remove k consecutive elements from A, 
+            such that amplitude of remaining elements will be minimal.
+
+            Aka keep track of max and min outside of k size interval
+            use both minheap and maxheap -> too painful to do. need to use map plus internal siftup siftdown
+
+            slide window left to right.
+            + monotonic deque for max + monotonic deque for min. 
+            add values to end of queue when you process,
+
+            pop from left side which contains best max, min, and pop when the index gets crossed!
+
+            */
+
+            // montonic queue after K elements. 
+            int idx = K; 
+            int N = A.size();
+
+            deque<Item> mono_max;
+            deque<Item> mono_min; 
+
+            // initialize monotonic struture with stack operations
+            for(int idx = K; idx < N; ++idx) {
+                // pop smaller elements and keep larger elements. 
+                Item i;
+                i.idx = idx;
+                i.val = A[idx];
+                while(!mono_max.empty() && mono_max.back().val <= i.val) {
+                    mono_max.pop_back();
+                }
+                mono_max.push_back(i);
+            }
+
+            for(int idx = K; idx < N; ++idx) {
+                // pop smaller elements and keep larger elements. 
+                Item i;
+                i.idx = idx;
+                i.val = A[idx];
+                while(!mono_min.empty() && mono_min.back().val >= i.val) {
+                    mono_min.pop_back();
+                }
+                mono_min.push_back(i);
+            }
+
+            // use monotone as queue
+            // 2 pointer + runnign min + update mono queues
+            int i = 0;
+            int j = K-1;
+
+            int amp = INT_MAX;
+
+            while(j != N) {
+                
+                int temp = mono_max.front().val - mono_min.front().val;
+                std::cout << "max and min is" << mono_max.front().val << ", " << mono_min.front().val << endl;
+
+                amp = min(amp, temp);
+                // add index i max_mono and min_mono 
+                // burn through the stack from the back garabage points
+                // set index to infinite cause these points we add when we move
+                // sliding window to right cannot be invalidated. 
+
+                // add in ith item in sliding window.         
+                Item newItem;
+                newItem.val = A[i];
+                newItem.idx = INT_MAX;
+
+                while(!mono_max.empty() && mono_max.back().val <= newItem.val) {
+                    mono_max.pop_back();
+                }
+                mono_max.push_back(newItem);
+                
+                while(!mono_min.empty() && mono_min.back().val >= newItem.val) {
+                    mono_min.pop_back();
+                }
+                mono_min.push_back(newItem);
+
+                // move sliding window.
+                i += 1;
+                j += 1;
+
+                // remove j+1th item from sliding window. 
+                if(j != N) {
+                    if(mono_max.front().idx == j) {
+                        mono_max.pop_front();
+                    }   
+
+                    if(mono_min.front().idx == j) {
+                        mono_min.pop_front();
+                    }
+                }
+            }
+
+            return amp; 
+        }
+
+        int main(){
+            vector<int> a = {1,2,3,4,5,6};
+            vector<int> b = {5,3,6,1,3};
+            vector<int> c = {8,8,4,3};
+            vector<int> d = {3,5,1,6,9,8,2,5,6};
+
+            cout << solution(a,  2) << endl;
+            cout << solution(b,  2) << endl;
+            cout << solution(c,  2) << endl;
+            cout << solution(d,  4) << endl;
+        }
+
+
+-85) think of the algo to do citadel problem -> round robin ALGORITHM!!!
+    -> Also take a look at the problem consecutive numbers sum
+
+
+-84) Using cumulative array for sums in 1D and 2D case tricks:
+    1D) sum between i and j inclsuive:
+        sum(j) - sum(i-1)
+        REMEMBER TO DO I-1 to make it INCLUSIVE!
+
+    2D)
+    Have a 2D cumulative array,
+    of size N+1, M+1, for NxM array
+    top row is all 0s.
+    left column is all 0s.
+    similar to cumualtive array. 
+    
+    2 coordinates is top left and bottom right. 
+    
+    (from snap interview)
+    SUM OF LARGE RECTANGE - SUM OF TOP RIGHT - SUM OF BOTTOM LEFT + SUM OF SMALL RECTANGLE. 
+    
+
+
+    topleft -> tlx, tly
+    bottomright -> brx, bry
+    
+    # because inclusive, not sure though, do lc to check below part.
+    tlx -= 1
+    tly -= 1
+
+    arr[brx][bry] - arr[brx][tly] - arr[tlx][bry]  + arr[tlx][tly]
+
+Snap DEADLOCK QUESTION
+
+    We obtained a log file containing runtime information about all threads and mutex locks of a user program. 
+    The log file contains N lines of triplets about threads acquiring or releasing mutex locks. The format of the file 
+    is: The first line contains an integer N indicating how many more lines are in the file. Each of the following N lines 
+    contains 3 numbers separated by space. The first number is an integer representing thread_id (starting from 1). 
+    The second number is either 0 (acquiring) or 1 (releasing). The third number is an integer representing mutex_id 
+    (starting from 1). Now we want you to write a detector program that reads the logs line by line and output the line 
+    number of the trace where a deadlock happens. If there is no deadlock after reading all log traces, output 0.
+
+    Example:
+    4
+    1 0 1
+    2 0 2
+    2 0 1
+    1 0 2
+
+    Output:
+    4
+
+    Ok so create graph and check for cycle?
+    
+    t1 wants a [a taken]
+    t2 wants b  [b taken]
+
+    t1 -> a -> t2 -> b -> t1
+    Cycle right!
+
+    or lets do it like
+
+    remove edges when that line comes from 
+    t2 wants a [a wanted not released -> t2 falls asleep with b]
+    t1 wants b [b wanted not released -> t1 falls asleep with a]
+    Soooo
+    both threads are asleep
+    because
+
+    Do cycle detection on resource allocation graph sir!
+
+
+
+
+
+-37) MAKING SURE YOUR DFS IS CORRECT! And the DP is being resolved 
+     in the DFS tree properly. 
+
+    For a given array A of N integers and a sequence S of N integers 
+    from the set {−1, 1}, we define val(A, S) as follows:
+
+    val(A, S) = |sum{ A[i]*S[i] for i = 0..N−1 }|
+
+    (Assume that the sum of zero elements equals zero.)
+    For a given array A, we are looking for such a sequence S that minimizes val(A,S).
+
+    Write a function:
+    def solution(A)
+
+    that, given an array A of N integers, computes the minimum value of val(A,S) 
+    from all possible values of val(A,S) for all 
+    possible sequences S of N integers from the set {−1, 1}.
+
+    For example, given array:
+
+    A[0] =  1
+    A[1] =  5
+    A[2] =  2
+    A[3] = -2
+    
+    your function should return 0, since for S = [−1, 1, −1, 1], 
+    val(A, S) = 0, which is the minimum possible value.
+
+    def solution(A):
+        # THIS FAILS DUE TO MAX RECURSION DEPTH REACHED!
+        # BUT IT IS 100% CORRECT
+        @lru_cache(None)
+        def recurseB(i,s):
+            
+            if len(A) == i:
+                return s
+                
+            add = recurseB(i+1, s + A[i])
+            sub = recurseB(i+1, s - A[i])
+            print("CORRECT ADD AND SUB FOR I IS", i, add, sub)
+
+            # print("ADD and sub are", add, sub)
+            if abs(add) < abs(sub):
+                return add
+            else:
+                return sub
+        
+        correct_val = abs(recurseB(0, 0))
+        print("CORRECT VALU IS", correct_val)
+        
+        # BELOW WAY IS WRONG!
+        # DO YOU KNOW WHY?
+        # IT GENERATES DIFF ANSWERS FROM ABOVE. 
+        # BECAUSE IN THE RECURSIVE CALLS CLOSE TO THE 
+        # BASE CASE, WE ARENT ABLE TO FINE TUNE THE SOLUTION
+        # TO THE INCOMING SUM, BECAUSE YOU NEVER SEE THE INCOMING
+        # SUM LIKE ABOVE. 
+        # SO INSTEAD, YOU GREEDILY CHOOSE 
+        # IN THE ABOVE RECURSION, HELPER SEES INCOMING SUM, 
+        # AND THEN RETURNS AN OPTIMIZED SUM BASED ON THE INCOMING SUM!
+        # THERE IS COMMUNICATION!
+        def recurseA(i):
+            if len(A) == i:
+                return 0
+                
+            add = A[i] + recurseA(i+1)
+            sub = -A[i] + recurseA(i+1)
+            print("INC ADD AND SUB FOR I IS", i, add, sub)
+            # print("ADD and sub are", add, sub)
+            if abs(add) < abs(sub):
+                return add
+            else:
+                return sub
+
+        incorrect_val = abs(recurseA(0))
+        return correct_val
+

+55.5) K stack pops (Finish it up https://binarysearch.com/problems/K-Stack-Pops):
+    
+        K Stack Pops
+        Medium
+        You are given two-dimensional list of integers stacks and an integer k. Assuming each list in stacks represents a stack, return 
+        the maximum possible sum that can be achieved from popping off exactly k elements from any combination of the stacks.
+        Constraints
+        n ≤ 500 where n is the number of rows in stacks.
+        m ≤ 200 where m is the maximum number of elements in a stack.
+        k ≤ 100
+        Youll realize only way is DP:
+        attempt top down, then do bottom up. Watch out for following failures:: 
+        class Solution:
+            def solveSlow(self, stacks, k):
+                
+                '''
+                This solution didnt pass, we didnt optimize the DP states enuff.
+                FAILURE
+                '''
+                @cache
+                def helper(i,j, remaining):
+                    # i is the list we are processing so far. 
+                    # remaining is amt of elements left.
+                    if remaining == 0:
+                        return 0
+                    if i == len(stacks):
+                        return float("-inf")
+                    take = float("-inf") 
+                    dont = float("-inf")
+                    # print("stacks i is", stacks[i])
+                    if len(stacks[i]) - j >= 0:
+                        element = stacks[i][len(stacks[i])-j]
+                    
+                        take = helper(i, j+1, remaining - 1) + element
+                        # stacks[i].append(element)
+                    dont = helper(i+1,1, remaining)
+                    return max(take, dont)
+                return helper(0,1, k)
+            def solve(self, stacks, k):
+                '''
+                A[i+1,l]=max{A[i,l−t]+(t pops from stack i+1),0≤t≤l}
+                We can compute A[m,k] in time O(k^2m).
+                
+                '''   
+                pass
+            # Forward dp? someone elses solution
+            def solve(self, stacks, k):
+                NINF = float("-inf")
+                dp = [NINF] * (k + 1)
+                dp[0] = 0
+                for stack in stacks:
+                    P = [0]
+                    for x in reversed(stack):
+                        P.append(P[-1] + x)
+                    for j in range(k, 0, -1):
+                        for i in range(1, min(j + 1, len(P))):
+                            dp[j] = max(dp[j], dp[j - i] + P[i])
+                return dp[k]
+                
+                return A[len(stacks) - 1][k]


+-53) Recursive Multiply:
+    Write a recursie function to multiply 2 positive integers without using the 
+    * operator. Only addition, subtraction and bit shifting but minimize ops. 
+
+    Answer:
+
+    int minProduct(int a, int b) {
+        int bigger = a< b ? b : a;
+        int smaller = a < b ? a: b;
+        return minProductHelper(a, b);
+    }
+
+    int minProdHelper(smaller, bigger) {s
+        if smaller == 0 return 0
+        elif smaller == 1 return 1
+
+        int s = smaller >> 1; //divide by 2
+        int halfPrd = minProductHelper(s, bigger);
+        if smaller % 2 == 0:
+            return halfProd + halfProd
+        else:
            # adding extra bigger because the bit is on?
+            return halfProd + halfProd + bigger
+    }
+    Runtime O(log s)

42) Bipartite matching problem: (Max flow 1)
    n students. d dorms. Each student wants to live in one of 
    the dorms of his choice. 
    Each dorm can accomodate at most one student. 

    Problem: Find an assignment that maximizes the number of 
    students who get a housing.
    
    Add source and sink. 
    make edges between students and dorms. 
    all edges weight are 1
    S-> all students -> all dorms -> T

    Find the max-flow. Then find the optimal assignment from the chosen
    edges. 

    If dorm j can accomodate cj students -> make edge with capacity
    cj from dorm j to the sink.
    
43) Decomposing a DAG into nonintersecting paths:
    -> Split each vertex v into vleft and vright
    -> For each edge u->v in the DAG, make an edge from uleft to vright

44) Min Cost Max Flow:
    A varient of max-flow problem. Each edge has capacity c(e) and
    cost cost(e). You have to pay cost(e) amount of money per unit 
    flow per unit flow flowing through e
    -> Problem: Find the max flow that has the minimum total cost.
    -> Simple algo (Slow):
        Repeat following:
            Take the residual graph
            Find a negative cost cycle using Bellman Ford
                -> If there is none, finish. 
            Circulate flow through the cycle to decrease the total cost,
            until one of the edges is saturated.
                -> Total amount of flow doesnt change .

47) Union Find Structure
    -> Used to store disjoint sets
    -> Can support 2 types of operations efficienty:
    - Find(x) returns the "representative" of the set that x belongs. 
    - Union(x, y): merges 2 sets that contain x and y

    Both operations can be done in (essentially) constant time
    Main idea: represent each set by a rooted tree
        -> Every node maintains a link to its parent
        -> A root node is "representative" of the corresponding set.
    
    Find(x) => follow the links from x until a node points itself. 
        -> This is O(N). DO PATH COMPRESSION.
        -> Makes tree shallower every time Find() is called. 
        -> After Find(x) returns the root, backtrack to x and reroute
            all the links to the root. 

    Union(x, y) => run Find(x) and Find(y) to find the 
            corresponding root nodes and direct one to the other

    Union By rank:  
        always attaches the shorter tree to the root of the 
        taller tree. Thus, the resulting tree 
        is no taller than the originals unless they were of equal height, 
        in which case the resulting tree is taller by one node.

        To implement union by rank, each element is associated with a rank. 
        Initially a set has one element and a rank of zero. If two sets are 
        unioned and have the same rank, the resulting set's rank is one larger; 
        otherwise, if two sets are unioned and have different ranks, the resulting
        set's rank is the larger of the two. Ranks are used instead of height or 
        depth because path compression will change the trees' heights over time.

    PSEUDOCODE:
    function MakeSet(x)
        if x is not already present:
            add x to the disjoint-set tree
            x.parent := x
            x.rank   := 0
            x.size   := 1

    function Find(x)
        if x.parent != x
            x.parent := Find(x.parent)
        return x.parent

    function Union(x, y)
        xRoot := Find(x)
        yRoot := Find(y)
    
        // x and y are already in the same set
        if xRoot == yRoot            
            return
    
        // x and y are not in same set, so we merge them
        if xRoot.rank < yRoot.rank
            xRoot, yRoot := yRoot, xRoot // swap xRoot and yRoot
    
        // merge yRoot into xRoot
        yRoot.parent := xRoot
        if xRoot.rank == yRoot.rank:
            xRoot.rank := xRoot.rank + 1
    


48) Applications of Union Find:
    keep track of the connected components of an undirected graph. 
    This model can then be used to determine whether 
    two vertices belong to the same component, 
    or whether adding an edge between them would result in a cycle. 
    DETECT CYCLE IN UNDIRECTED GRAPH:
    
        # A utility function to find the subset of an element i 
        def find_parent(self, parent,i): 
            if parent[i] == -1: 
                return i 
            if parent[i]!= -1: 
                return self.find_parent(parent,parent[i]) 
    
        # A utility function to do union of two subsets 
        def union(self,parent,x,y): 
            x_set = self.find_parent(parent, x) 
            y_set = self.find_parent(parent, y) 
            parent[x_set] = y_set 

        # The main function to check whether a given graph 
        # contains cycle or not 
        def isCyclic(self): 
            
            # Allocate memory for creating V subsets and 
            # Initialize all subsets as single element sets 
            parent = [-1]*(self.V) 
    
            # Iterate through all edges of graph, find subset of both 
            # vertices of every edge, if both subsets are same, then 
            # there is cycle in graph. 
            for i in self.graph: 
                for j in self.graph[i]: 
                    x = self.find_parent(parent, i)  
                    y = self.find_parent(parent, j) 
                    if x == y: 
                        return True
                    self.union(parent,x,y) 


49) Detect negative cycles with Bellman Ford:

    1) Initialize distances from source to all vertices as infinite and distance to source itself as 0. 
    Create an array dist[] of size |V| with all values as infinite except dist[src] where src is source vertex.

    2) This step calculates shortest distances. Do following |V|-1 times where |V| is the number of vertices in given graph.
    …..a) Do following for each edge u-v
    ………………If dist[v] > dist[u] + weight of edge uv, then update dist[v]
    ………………….dist[v] = dist[u] + weight of edge uv

    3) This step reports if there is a negative weight cycle in graph. Do following for each edge u-v
    ……If dist[v] > dist[u] + weight of edge uv, then “Graph contains negative weight cycle”

    The idea of step 3 is, step 2 guarantees shortest distances if graph doesn’t 
    contain negative weight cycle. If we iterate through all edges one more 
    time and get a shorter path for any vertex, then 
    there is a negative weight cycle.

52) Flattening a multilevel doubly linked list using a stack:
        def flatten(self, head):
            if not head:
                return
            
            dummy = Node(0,None,head,None)     
            stack = []
            stack.append(head)
            prev = dummy
            
            while stack:
                root = stack.pop()

                root.prev = prev
                prev.next = root
                
                if root.next:
                    stack.append(root.next)
                    root.next = None
                if root.child:
                    stack.append(root.child)
                    root.child = None
                prev = root        
                
            # disengage dummy node
            dummy.next.prev = None
            return dummy.next


53) The art of segment trees and monoqueues:
    PLEASE REVIEW SEGMENT TREES IN DATA STRUCTURES NOTES!!

    Previously we saw segment trees.
    That data structure was able to answer the question

    reduce(lambda x,y: operator(x,y), arr[i:j], default)
    and we were able to answer it in O(\log(j - i)) time. 
    Moreover, construction of this data structure only took O(n) time. 
    Moreover, it was very generic as operator and 
    default could assume be any values.

    This obviously had a lot of power, but we can use something a lot 
    simpler if we want to answer easier problems. 
    Suppose instead, we wanted to ask the question

    reduce(lambda x,y: operator(x,y), arr[i:i + L])
    with the caveat that operator(x,y) will return 
    either x or y and that L remains fixed.

    Some examples of this is to find the minimum of some 
    fixed length range, or the maximum in some fixed length range.

    Introducing, the Monotonic Queue, with the handy name Monoqueue.

    The code for this guy is pretty self explanatory, but the basic idea 
    is to maintain a sliding window that sweeps across. 
    Moreover, the contents of the Monoqueue are sorted 
    with respect to the comparator Operator, so to 
    find the “best” in a range one only need look at the front.

    from collections import deque
    class Monoqueue:
        def __init__(self, operator):
            self.q = deque()
            self.op = operator
        
        def get_best(self):
            if not self.q:
                return None
            return self.q[0][0]
        
        '''

        if q has elements, look at back of queue. 
        pop the elements off that cannot be best possible (i think? WTF??)
        Ok sure. What the count does is it shows the range for which 
        it is the best possible value. once you exit the range, remove the front
        of the queue, and go to the next best, which will be the best for a certain sized window
        before another thing is the best. 
        So we keep track of the bests at each step only, and keep the best even if we "pop" because
        its located a lil to the right, of the thing you popped on the main stream. 
        '''
        def push(self, val):
            count = 0
            while self.q and self.op(val, self.q[-1][0]):
                count += 1 + self.q[-1][1]
                self.q.pop()
            self.q.append([val, count])
        '''
        Pop front. only if count is == 0.
        Otherwise, decrement the count. Why is the count important?
        I guess the count keeps track of the range you are sliding across
        so if you pop it, and slide to the right, its still THE BEST POSSIBLE VALUE 
        (because the best possible value is located somewhere a lil to the right), 
        because it was not actually popped. just some garbage element on the main list 
        was popped. 
        '''
        def pop(self):
            if not self.q:
                return None
            if self.q[0][1] > 0:
                self.q[0][1] -= 1
            else:
                self.q.popleft()

54) Boyer moore voting algortihm:
    given an array of n elements, tell me if there is an 
    element that appears more than n/2 times.

    Obviously this has a huge amount of uses.

    The trivial solution would be to sort the array in O(n * log(n)) time and then

    Boyer Moore Algorithm
    This will require two passes, the first to find a 
    possible candidate, and the second to verify that 
    it is a candidate. The second part is 
    trivial so we will not focus on it. The first sweep
    however is a little bit more tricky.

    We initialize a count to 0. Then as we proceed in the list, 
    if the count is 0, we make the current value being
    looked at the candidate. If the value we’re looking at 
    is the candidate, then we increment the count by 1, 
    otherwise we decrement the count.

    Distributed Boyer-Moore
    Determine how many cores/processors you have, and call it t. 
    Now, split your array into t positions and run Boyer-Moore 
    on each of these sections. You will get a candidate 
    and a count from each of these sections. Now, 
    recursively run Boyer-Moore on the list of 
    [candidate1 * count1] + [candidate2 * count2] … . Pretty cool right!

    Generalization
    The generalized problem is given a list, 
    find all elements of the list that occur more 
    than n/k times. We simply have to carry 
    the proof from the last case over and it applies directly.

    Have a set of possible candidates, and have an associated count with them.

    Iterate through the list, and if the item is in 
    the dictionary, increment the count. If not and 
    the number of elements in the dictionary is less 
    than k, then just add it to the dictionary with count 1. 
    Otherwise, decrement all the counters by 1, and delete 
    any candidates with a counter 1.

55)    Common problems solved using DP on broken profile include:

        finding number of ways to fully fill an area 
        (e.g. chessboard/grid) with some figures (e.g. dominoes)
        finding a way to fill an area with minimum number of figures
        finding a partial fill with minimum number of unfilled space (or cells, in case of grid)
        finding a partial fill with the minimum number of figures, such that no more figures can be added
        Problem "Parquet"
        
        Problem description. Given a grid of size N×M. 
        Find number of ways to 
        fill the grid with figures of size 2×1 (no cell 
        should be left unfilled, 
        and figures should not overlap each other).

        Let the DP state be: dp[i,mask], where i=1,…N and mask=0,…2^(M)−1.

        i respresents number of rows in the current grid, and mask is the state of 
        last row of current grid. If j-th bit of mask is 0 then the corresponding 
        cell is filled, otherwise it is unfilled.

        Clearly, the answer to the problem will be dp[N,0].

        We will be building the DP state by iterating over each i=1,⋯N 
        and each mask=0,…2^(M)−1, and for each mask we will be only transitioning forward, 
        that is, we will be adding figures to the current grid.
        
        int n, m;
        vector < vector<long long> > dp;
        void calc (int x = 0, int y = 0, int mask = 0, int next_mask = 0)
        {
            if (x == n)
                return;
            if (y >= m)
                dp[x+1][next_mask] += d[x][mask];
            else
            {
                int my_mask = 1 << y;
                if (mask & my_mask)
                    calc (x, y+1, mask, next_mask);
                else
                {
                    calc (x, y+1, mask, next_mask | my_mask);
                    if (y+1 < m && ! (mask & my_mask) && ! (mask & (my_mask << 1)))
                        calc (x, y+2, mask, next_mask);
                }
            }
        }


        int main()
        {
            cin >> n >> m;

            dp.resize (n+1, vector<long long> (1<<m));
            dp[0][0] = 1;
            for (int x=0; x<n; ++x)
                for (int mask=0; mask<(1<<m); ++mask)
                    calc (x, 0, mask, 0);

            cout << dp[n][0];
        }

55.5) FOR THE PROBLEM ABOVE:

        790. Domino and Tromino Tiling
        Medium
        Topics
        Companies
        You have two types of tiles: a 2 x 1 domino shape and a tromino shape. 
        You may rotate these shapes. -> TROMINO looks like 
        domino look like -> aa

        tromino look lik -> aa
                             a 

        Given an integer n, return the number of ways to tile an 2 x n board. 
        Since the answer may be very large, return it modulo 109 + 7.

        In a tiling, every square must be covered by a tile. Two tilings are different if and 
        only if there are two 4-directionally adjacent cells on the board such that exactly 
        one of the tilings has both squares occupied by a tile.

        Example 1:

        Input: n = 3
        Output: 5
        Explanation: The five different ways are show above.
        Example 2:

        Input: n = 1
        Output: 1
        
        Constraints:
        1 <= n <= 1000

        dp[i] denotes the number of ways to tile an 2 * (i + 1) board, note that dp is 0-indexed.
        Intuitively, dp[0] = 1 and dp[1] = 2
        dpa[i] denotes the number of ways to tile an 2 * i board and 1 more square in that row (partially filled)
        Intuitively, dpa[0] = 0 and dpa[1] = 1
        I just explained the case where in i-th column, 2nd row is filled. But it should be noted that the two cases(the other is in i-th column, 1st row is filled) are symmetric and the numbers are both dpa[i], you may imagine dpb[i] = dpa[i] for the second case where i-th column 1st row is filled.

        class Solution(object):
            def numTilings(self, n):
                dp, dpa = [1, 2] + [0] * n, [1] * n
                for i in range(2, n):
                    dp[i] = (dp[i - 1] + dp[i - 2] + dpa[i - 1] * 2) % 1000000007
                    dpa[i] = (dp[i - 2] + dpa[i - 1]) % 1000000007
                return dp[n - 1]

        SIMPLIFIES TO:


        class Solution(object):
            def numTilings(self, n):
                dp = [1, 2, 5] + [0] * n
                for i in range(3, n):
                    dp[i] = (dp[i - 1] * 2 + dp[i - 3]) % 1000000007
                return dp[n - 1]





56) POW(X, N) and binary exponentiation

    Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
    Example 1:

    Input: x = 2.00000, n = 10
    Output: 1024.00000
    Example 2:

    Input: x = 2.10000, n = 3
    Output: 9.26100
    Example 3:

    Input: x = 2.00000, n = -2
    Output: 0.25000
    Explanation: 2-2 = 1/22 = 1/4 = 0.25

                """
                Lets implement binary exponentiation...
                instead of 7*7*7*7*7.. 14 times how about log(14) performance!!
                x = 7^14
                14 is    1110
                ok so we can do 
                7^2 * 7^4 * 7^8
                    7^2^2.  (7^2^2)^2  
                a ^ (b+c+d) = a^b a^c a^d
                we want to acheive log(n) using bin exp
                7^2 = 7 * 7 
                7^4 = 7^2
                7^8 = 7^4 * 7^4 

                Then just multiply the terms you want essentially right!

                If power is negative..

                7^-4 = 7^-2 * 7^-2
                actually just do the postiveis and set 7 to 1/7 bruh to start.         
                """

                class Solution:
                    def myPow(self, x: float, n: int) -> float:
                        power = x

                        if n < 0: 
                            n = -n
                            power = 1/power
                        
                        res = 1 

                        while n:
                            is_set = n & 1
                            if is_set:
                                res = res * power 
                            power = power * power 
                            n = n >> 1
                        return res 


+32.5) DFS LOW LINK!
+        ARTICULATION POINTS :
+        These are also called cut-vertices . When they are removed(including the edges connected to them) , 
+        the remaining graph is broken down into two or more connected components.
+        BRIDGES:
+        These are also called cut-edges . When they are removed, the graph 
+        is broken down into two or more connected components.
+        FORWARD EDGES:
+        They are the edges taken during the dfs (or bfs) . 
+        More specifically, they are the edges present in the dfs tree.
+        BACK EDGES:
+        They are the edges that connect vertices to some of its ancestors..
+        NOTE:  In a dfs tree there exits no cross-edge. Suppose there are 2 vertices u and v connected 
+        to each other and some same ancestor in dfs-tree say w. So u(or v) is visited before v(or u) 
+        during dfs and the edge u-v then becomes a forward edge making the edge v-w(or u-w) a back edge.
+        PROPERTY I :  In a tree , all edges are cut-edges. (By defination of a tree)
+        PROPERTY II: For a vertex u to be a cut-vertex, there should be some vertex v in its subtree 
+        in dfs tree which has no back edge to any ancestor of u or none of its ancestors which lie in 
+        path from u to to v has a backedge to an ancestor of u.  Why? Suppose we claim vertex w is a 
+        cut vertex and u be any vertex in it’s subtree. If u has a back-edge to any ancestor of w, 
+        cutting w doesn’t break it into two components as u is still reachable from ancestor’s of w.
+        PROPERTY III: For an edge e connecting u and v  ( par[v]=u ) to be a cut-edge , none of the 
+        vertices in the subtree of v should have a backedge to u or any of its ancestors. (Same reasoning as above)
+        Note: From now on we will be focussing on finding bridges of a graph. Finding Articulation Points would just need some trivial changes from that.
+        So how to find bridges in a graph ?
+        BRUTE FORCE
+        We can loop through all the edges, remove them and check if it’s a cut edge or not by running a dfs and 
+        checking the number of connected components formed. COMPLEXITY: O(E*(V+E))        
+        OPTIMIZED APPROACH
+        Let’s start by defining a new term:
+        LOWLINKS:
+        Lowlink of a vertex v is the maximum ancestor in dfs tree to 
+        which v or any node in its subtree in dfs tree has a backedge to.
+                 ---> 1 
+                /    /\
+               /   2   5               
+               |  /    /\   
+               | 3    6  7
+               |/
+               4
+        In the above example, node 1 has lowlink 1, node 2 has lowlink 1, node 3 has lowlink 1, node 4 has lowlink 1, node 5 has lowlink 5,
+        node 6 has lowlink 6, node 7, has lowlink 7
+        The above figure shows the dfs tree of a graph with the lowlink values written in red. Note that node 6 
+        has a lowlink value 6 instead of being connected to 5 because the edge 5-6 is not a backedge.
+        Now how to assign the lowlink values? An easy way is by assigning minimum height of node to which it has a
+        backedge. But we will do it similar to Euler Tour (Inorder traversal). As we know in euler tour,  ancestors get 
+        lesser value assigned . So here our lowlink  value will be the least ancestor value to which any node in the subtree 
+        has a backedge.  So now, what will be the condition for an edge to be bridge? An edge u-v (u=par[v]) becomes a 
+        bridge iff the lowlink value of node v is greater than value assigned to node u. Why? Because u will have 
+        lower value than any node in subtree of v. So a higher lowlink means there is no backedge 
+        in subtree of v to u or any ancestor of u.
+        Let’s look into some code now:
+        vectorg[N];
+        int timestamp[N];
+        int best[N];
+        bool visited[N];
+        int par[N];
+        bool iscut[N];
+        int T=0;
+        void dfs(int s,int p)
+        {
+            par[s]=p;
+            timestamp[s]=T++;
+            best[s]=timestamp[s];
+            visited[s]=true;
+            for(auto v:g[s])
+            {
+                if(!visited[v.first])
+                {
+                    dfs(v.first,s);
+                    best[s]=min(best[s],best[v.first]);
+                    if(best[v.first]>timestamp[s])
+                    {
+                        iscut[v.second]=true;
+                    }
+                }
+                else if(v.first!=p)
+                {
+                    best[s]=min(best[s],best[v.first]);
+                }
+        Let’s try understanding what’s happening:
+        timestamp[s]=T++ : This assigns values to nodes same as in euler tour. 
+        Note that T is incremented before visiting subtree of s. 
+        So all nodes in subtree of s has higher timestamp value than s.
+        best[s] : This is the lowlink value of s. Initially it is assigned to same value as that of s.
+        best[s]=min(best[s],best[v.first]) : We update our current lowlink value 
+        if any node in subtree of v.first has a backedge to some ancestor of s.
+        best[v.first]>timestamp[s]: This is the same condition we discussed above for an edge to be cut-edge. 
+        (Note: Here we are labelling v.first as a cut-edge though it is a vertex. We can always represent 
+        any edge u-v (u=par[v]) as v as every vertex has atmax one parent in a tree.
+

+36.5) Eulerean tour:
+    Necessary and sufficient conditions
+        An undirected graph has a closed Euler tour iff it is connected and 
+        each vertex has an even degree.
+        An undirected graph has an open Euler tour (Euler path) if it is connected, and each vertex, 
+        except for exactly two vertices, has an even degree. The two vertices of odd degree have to be the endpoints of the tour.
+        A directed graph has a closed Euler tour iff it is strongly connected and the in-degree of each vertex is equal to its out-degree.
+        Similarly, a directed graph has an open Euler tour (Euler path) iff for each vertex the difference 
+        between its in-degree and out-degree is 0, except for two vertices, where one has difference +1 (the start of the tour) 
+        and the other has difference -1 (the end of the tour) and, if you add an edge from the
+        end to the start, the graph is strongly connected.
+    Fleury's algorithm (Not the best one)
+        Fleury's algorithm is a straightforward algorithm for finding Eulerian paths/tours. It proceeds by repeatedly 
+        removing edges from the graph in such way, that the graph remains Eulerian. A version of the algorithm, 
+        which finds Euler tour in undirected graphs follows.
+        Start with any vertex of non-zero degree. Choose any edge leaving this vertex, which is not a bridge 
+        (i.e. its removal will not disconnect the graph into two or more disjoint connected components). If there is no 
+        such edge, stop. Otherwise, append the edge to the Euler tour, remove it from the graph, and 
+        repeat the process starting with the other endpoint of this edge.
+        Though the algorithm is quite simple, it is not often used, because it needs to identify bridges 
+        in the graph (which is not a trivial thing to code.) Slightly more sophisticated, but easily implementable algorithm is presented below.
+    Cycle finding algorithm (Better)
+        This algorithm is based on the following observation: if C is any cycle in a Eulerian graph, 
+        then after removing the edges of C, the remaining connected components will also be Eulerian graphs.
+        The algorithm consists in finding a cycle in the graph, removing its edges and
+        repeating this steps with each remaining connected component. It has a very compact code with recursion:
+    PSUEDOCODE:
+        find_tour(u):
+            for each edge e=(u,v) in E:
+                remove e from E
+                find_tour(v)
+            prepend u to tour
+        where u is any vertex with a non-zero degree.  
+37) Lets go over that again! Cut edges, cut vertices, Eulerian tours   
+    Definitions An Euler tour (or Eulerian tour) in an undirected graph is a tour that
+    traverses each edge of the graph exactly once. Graphs that have an Euler tour 
+    are called Eulerian.
+    Finding cut edges -------------------
+        The code below works properly because the lemma above (first lemma):
+        h is the height of the vertex. v is the parent. u is the child.
+
+        We need compute for each subtree, the lowest node in the DFS tree that a back edge can reach. 
+        This value can either be the depth of the other end point, or the discovery time. 
+        Cut edges can, also, be seen as edges that needs to be removed 
+        to end up with strongly connected components.
+        h[root] = 0
+        par[v] = -1
+        dfs (v):
+                d[v] = h[v]
+                color[v] = gray
+                for u in adj[v]:
+                        if color[u] == white
+                                then par[u] = v and dfs(u) and d[v] = min(d[v], d[u])
+                                if d[u] > h[v]
+                                        then the edge v-u is a cut edge
+                        else if u != par[v])
+                                then d[v] = min(d[v], h[u])
+                color[v] = black
+        In this code, h[v] =  height of vertex v in the DFS tree and d[v] = min(h[w] where 
+                                            there is at least vertex u in subtree of v in 
+                                      the DFS tree where there is an edge between u and w).

+    Finding cut vertices -----------------
+        The code below works properly because the lemma above (first lemma):
+        h[root] = 0
+        par[v] = -1
+        dfs (v):
+                d[v] = h[v]
+                color[v] = gray
+                for u in adj[v]:
+                        if color[u] == white
+                                then par[u] = v and dfs(u) and d[v] = min(d[v], d[u])
+                                if d[u] >= h[v] and (v != root or number_of_children(v) > 1)
+                                        then the edge v is a cut vertex
+                        else if u != par[v])
+                                then d[v] = min(d[v], h[u])
+                color[v] = black
+        In this code, h[v] =  height of vertex v in the DFS tree and d[v] = min(h[w] where 
+        there is at least vertex u in subtree of v in the DFS tree where there is an edge between u and w).

+    Finding Eulerian tours ----------------
+        It is quite like DFS, with a little change :
+        vector E
+        dfs (v):
+                color[v] = gray
+                for u in adj[v]:
+                        erase the edge v-u and dfs(u)
+                color[v] = black
+                push v at the end of e
+        e is the answer.
+    Python implementation
+    This is a short implementation of the Euler tour in python.
+        # g is a 2D adjacency matrix
+        circuit = []
+        def visit(current):
+            for x in range(MAX_N):
+                if g[current][x] > 0:
+                    g[current][x] -= 1
+                    g[x][current] -= 1
+                    visit(x)
+            circuit.append(current)
+        # circuit now has the circuit in reverse, if ordering matters
+        # you can start with any node for a closed tour, and an odd degree node for a open tour


57) Articulation points and Biconnected graphs:
    In graph theory, a biconnected component (sometimes known as a 2-connected component) 
    is a maximal biconnected subgraph. Any connected graph decomposes into a 
    tree of biconnected components called the block-cut tree of the graph. 
    The blocks are attached to each other at shared vertices called 
    cut vertices or articulation points. Specifically, a 
    cut vertex is any vertex whose removal 
    increases the number of connected components.

    biconnected graph -> if any one vertex were to be removed, 
    the graph will remain connected. 
    Therefore a biconnected graph has no articulation vertices.
    
    We obviously need two passes. In the first pass, we want to figure out which 
    vertex we can see from each vertex through back edges, if any. 
    In the second pass we want to visit vertices in the opposite 
    direction and collect the minimum bi-component ID 
    (i.e. earliest ancestor accessible from any descendants).

    AP Pseudocode:

    time = 0
    visited[i] = false for all i
    GetArticulationPoints(u)
        visited[u] = true
        u.st = time++
        u.low = u.st    //keeps track of highest ancestor reachable from any descendants
        dfsChild = 0    //needed because if no child then removing this node doesn't decompose graph
        for each ni in adj[i]
            if not visited[ni]
                GetArticulationPoints(ni)
                ++dfsChild
                parents[ni] = u
                u.low = Min(u.low, ni.low)  //while coming back up, get the lowest reachable ancestor from descendants
            else if ni <> parent[u] //while going down, note down the back edges
                u.low = Min(u.low, ni.st)

        //For dfs root node, we can't mark it as articulation point because 
        //disconnecting it may not decompose graph. So we have extra check just for root node.
        if (u.low = u.st and dfsChild > 0 and parent[u] != null) or (parent[u] = null and dfsChild > 1)
            Output u as articulation point
            Output edges of u with v.low >= u.low as bridges
            Output u.low as bicomponent ID


58) K closest points to origin:

    h=[]

    for x,y in points:
        dist=math.sqrt(x**2+y**2)
        if len(h)<k:
            heapq.heappush(h,(-dist,[x,y]))
        else:
            heapq.heappushpop(h,(-dist,[x,y]))
    return [h[i][1] for i  in range(k)]


    You must also know Quick selection soln as well!

    Not the best partition algo find a better one. 

    class Solution:
        def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
            # QUICK SELECT?
            # kth CLOSEST POINT + points greater than k right..
            #hmm not a sorted list can i do quick seleecT? yes
            lst = []
            for idx, [x, y] in enumerate(points):
                lst.append( [x**2 + y**2, idx] )

            # quick select.
            # we want smallest values...
            
            left_elements = []
            right_elements = []
            soln = []
            
            while k != 0:
                pivot = random.randint(0, len(lst) - 1)
                pivot_val = lst[pivot]

                for i in lst:
                    if i[0] <= pivot_val[0]:
                        left_elements.append(i)
                    else:
                        right_elements.append(i)            

                if len(left_elements) <= k:
                    lst = right_elements
                    k = k - len(left_elements)
                    soln += left_elements
                    left_elements = []
                    right_elements = []
                else:
                    lst = left_elements
                    left_elements = []
            
            return list(map(lambda x: points[x[1]], soln + left_elements))


59)
        Given an integer array sorted in non-decreasing order, there is exactly one integer 
        in the array that occurs more than 25% of the time, return that integer.

        

        Example 1:

        Input: arr = [1,2,2,6,6,6,6,7,10]
        Output: 6
        Example 2:

        Input: arr = [1,1]
        Output: 1

        I think you can solve this with boyer moore voting algo.. 


        class Solution:
            def findSpecialInteger(self, arr: List[int]) -> int:
                n = len(arr)
                candidates = [arr[n // 4], arr[n // 2], arr[3 * n // 4]]
                target = n / 4
                
                for candidate in candidates:
                    left = bisect_left(arr, candidate)
                    right = bisect_right(arr, candidate) - 1
                    if right - left + 1 > target:
                        return candidate
                    
                return -1

        BOYER MOOR VOTING ALGO SOLN:

        class Solution {
                bool valid(int candidate, vector<int> &nums, int req)
                {
                    int count = 0;
                    for(int num : nums)
                        count += candidate == num;
                    return count > req;
                }
                
            public:
                int findSpecialInteger(vector<int>& arr) {
                    int first = 0;
                    int second = 0;
                    int third = 0;
                    
                    int c1 = 0, c2 = 0, c3 = 0;
                    
                    
                    for(int num : arr)
                    {
                        if(num == first)
                            ++c1;
                        else if(num == second)
                            ++c2;
                        else if(num == third)
                            ++c3;
                        else if(c1 == 0)
                        {
                            first = num;
                            c1 = 1;
                        }
                        else if(c2 == 0)
                        {
                            second = num;
                            c2 = 1;
                        }
                        else if(c3 == 0)
                        {
                            third = num;
                            c3 = 1;
                        }
                        else
                        {
                            --c1, --c2, --c3;
                        }
                    }
                    
                    int req = arr.size() / 4;
                    if(valid(first, arr, req)) return first;
                    if(valid(second, arr, req)) return second;
                    if(valid(third, arr, req)) return third;
                    return -1;
                }
            };

60) 2D PREFIX GRID FOR PREFIX SUMS GOOD TO MEMORIZE THIS ALGO:

        First understand 1D prefix sum array..

        The prefix sum array starts at 0 and has N+1 elements if 
        original array has N elements. 
        int sum = 0;
        for(int i=0;i<n;i++){
            sum+=arr[i];
            prefix[i]=sum;
        }

        Then you can compute the sum between i and j as sum[j] - sum[i-1] Right?
        yes. 
        So the total sum is actually   sum[N] (or sum[N] - sum[0]) as such!

        For this array
        1 6 4 2 5 3
        0 1 2 3 4 5

        Prefix sum:
        0, 1, 7, 11, 13, 18, 21
        0  1  2  3    4   5  6
        to get sum from index 1 to index 4 in original array do (6+4+2+5):
        Prefix[5] - Prefix[2 - 1] (Point to include - stuff we dont want in sum)
        == 18 - 1 = 17
        but the indices add 1 to go from original to prefix array for htis shit. 

        NOW LETS THINK IN 2D:

        3 0 1 4 2
        5 6 3 2 1
        1 2 0 1 5
        4 1 0 1 7
        1 0 3 0 5


        vector<vector<int>> prefix(n+1,vector<int>(m+1,0));
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
            prefix[i][j]=prefix[i][j-1]+prefix[i-1][j]-prefix[i-1][j-1]+grid[i-1][j-1];
            }
        }

        0 0  0  0  0  0 
        0 3  3  4  8  10 
        0 8  14 18 24 27 
        0 9  17 21 28 36 
        0 13 22 26 34 49 
        0 14 23 30 38 58

        We can now query any rectangle here and get a sum!
        
        Have to look at a 3x3 SQUARE in prefix sum to get a 2x2 SQUARE sum
        reason being 
        We have to deal with coordinate representing r2c2 and also the coordinate that is outside the square 
        we want which is r1-1 and c1-1 
        look at below algo also look at the last binary search problem in template more prefix sums. 

        while(query--){
            int r1,c1; cin>>r1>>c1;
            int r2,c2; cin>>r2>>c2;
            // 0 based indexing so need increase
            r1++; c1++; r2++; c2++;

            cout<<prefix[r2][c2]-prefix[r1-1][c2]-prefix[r2][c1-1]+prefix[r1-1][c1-1];
            cout<<endl;
        }


61) Union find practice:

        200. Number of Islands
        Solved
        Medium
        Topics
        Companies
        Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
        return the number of islands.

        An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
        You may assume all four edges of the grid are all surrounded by water.

        

        Example 1:

        Input: grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
        ]
        Output: 1
        Example 2:

        Input: grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
        ]
        Output: 3



        YOU CAN DO THIS WITH DFS OR BFS.

        LETS TRY UNION FIND:


        class Solution:
            def numIslands(self, grid: List[List[str]]) -> int:
                
                rank = {}
                parent = {}
                components = 0 

                for i, row in enumerate(grid):
                    for j, val in enumerate(grid[i]):
                        # print(i, row, j, val)
                        if val == "1":
                            rank[(i, j) ] = 1
                            parent[(i, j)] = (i, j)
                            components += 1

                def union(a, b):
                    nonlocal components
                    pa = find(a)
                    pb = find(b)
                    if pa == pb :
                        return 
                    if rank[pb] > rank[pa]:
                        pb, pa, = pa, pb

                    parent[pb] = pa
                    if rank[pb] == rank[pa]:
                        rank[pa] += 1  

                    components -= 1         
                
                def find(a):
                    if parent[a] != a:
                        parent[a] = find(parent[a]) 
                        return parent[a]
                    return a 
                    
                dirs = [(+1, 0), (-1, 0), (0, +1), (0, -1)]

                visited = set()

                for i, row in enumerate(grid):
                    for j, val in enumerate(row):
                        if val == "1":
                            for direction in dirs:
                                x, y = direction
                                if i + x >= 0 and i + x < len(grid) and j + y >= 0 and j+y < len(grid[0]) :
                                    if grid[i+x][j+y] == "1":
                                        union((i,j), (i + x, j + y) )
                
                #component_set = set()
                #for key, value in parent.items():
                #    component_set.add(find(key))
                #return len(component_set)
                # faster way is tokeep track of number of components as well!

                return components



62) DFAs to sovle problems:

    Approach 2: Deterministic Finite Automaton (DFA)
    Intuition

    Let's now view Approach 1 from a different angle. There were 3 boolean variables, and they can be 
    either true or false. Each time we read a character in the string, we either stayed in the current 
    state (boolean variables stayed the same) or we transitioned into a new state (boolean variables changed).
    What we've described above is a lot like a deterministic finite automaton. A DFA is a finite number of 
    states with transition rules to move between them. However, keep in mind that the state depends on 
    more than just those 3 boolean variables; it also depends on whether we have encountered a + or -. We will discuss this in more detail later.

    Never heard of a DFA before?

    DFA's are useful for solving many problems, including advanced dynamic programming problems such as 1411. 
    Number of Ways to Paint N X 3 Grid. So if you're not yet familiar with them, we recommend that you read up on them. It will be worth it!

    DFAs share a lot of similarities with the trie data structure. Recall that a trie is used to 
    represent a dictionary of words, in a space-efficient manner. To check whether or not a word is 
    in the dictionary, we would simultaneously traverse through the word and the trie. If we end at a 
    node that is marked as a valid end-point, then we would return true. Otherwise, if we get "stuck", 
    or end at a node that is not an end-point, we would return false. It's the same for a DFA: we start 
    at a "root" node, and then check each character one by one, checking whether or not there is a valid transition we can make.

    There are a few key differences between DFA's and tries, so keep these in mind while reading through the remainder of this section.

    While a trie can only represent a finite number of strings (the given dictionary), a DFA can represent an infinite number of different strings.
    While a trie can only move down the implicit tree, a DFA can essentially "loopback" to a higher level, or stay on the same level, or even the same node.
    A trie is a type of tree, and a DFA is a type of directed graph.
    Other than that, you can lean on your existing knowledge of tries to wrap your head around this new data structure.

    Algorithm

    The first step is to design our DFA. Picture the DFA as a directed graph, where each node is a state, 
    and each edge is a transition labeled with a character group (digit, exponent, sign, or dot). There are two key steps to designing it.

    Identify all valid combinations that the aforementioned boolean variables can be in. 
    Each combination is a state. Draw a circle for each state, and label what it means.

    For each state, consider what a character from each group would mean in the context of that state. 
    Each group will either cause a transition into another state, or it will signify that the string is invalid. 
    For each valid transition, draw a directed arrow between the two states and write the group next to the arrow.
    Try this on your own before reading any further! Take a few minutes and try to design the DFA. Keep in mind 
    that a state can point to itself. For example, with the input "12345", if we were to use the first approach, after the first character, none of the boolean variables change, so the state should not change. Therefore, the node should have an edge that points to itself, labeled by "digit". Another hint: the start node should have 3 outgoing edges labeled "digit", "sign", and "dot".


        65. Valid Number
        Attempted
        Hard
        Topics
        Companies
        Given a string s, return whether s is a valid number.

        For example, all the following are valid numbers: "2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789", while the following are not valid numbers: "abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53".

        Formally, a valid number is defined using one of the following definitions:

        An integer number followed by an optional exponent.
        A decimal number followed by an optional exponent.
        An integer number is defined with an optional sign '-' or '+' followed by digits.

        A decimal number is defined with an optional sign '-' or '+' followed by one of the following definitions:

        Digits followed by a dot '.'.
        Digits followed by a dot '.' followed by digits.
        A dot '.' followed by digits.
        An exponent is defined with an exponent notation 'e' or 'E' followed by an integer number.

        The digits are defined as one or more digits.


        With our constructed DFA, our algorithm will be:

        Initialize the DFA as an array of hash tables. Each hash table's keys will be a 
        character group, and the values will be the state it should transition to. We can use 
        the indexes of the array to handle state transitions. Set the currentState = 0.

        Iterate through the input. For each character, first determine what group it belongs to. 
        Then, check if that group exists in the current state's hash table. If it does,
         transition to the next state. Otherwise, return false.

        At the end, check if we are currently in a valid end state: 1, 4, or 7.

        If you're having trouble with your implementation, try to go through your DFA with a 
        complicated case such as -123.456E+789. Follow along with your designed DFA, and if 
        there is a bug, check which edge case went wrong and adjust the graph accordingly. 
        Once your DFA is correctly designed, the coding part will be less challenging.

        # DFA
        https://leetcode.com/problems/valid-number/editorial/
    
        class Solution(object):
            def isNumber(self, s):
                # This is the DFA we have designed above
                dfa = [
                    {"digit": 1, "sign": 2, "dot": 3},
                    {"digit": 1, "dot": 4, "exponent": 5},
                    {"digit": 1, "dot": 3},
                    {"digit": 4},
                    {"digit": 4, "exponent": 5},
                    {"sign": 6, "digit": 7},
                    {"digit": 7},
                    {"digit": 7},
                ]

                current_state = 0
                for c in s:
                    if c.isdigit():
                        group = "digit"
                    elif c in ["+", "-"]:
                        group = "sign"
                    elif c in ["e", "E"]:
                        group = "exponent"
                    elif c == ".":
                        group = "dot"
                    else:
                        return False

                    if group not in dfa[current_state]:
                        return False

                    current_state = dfa[current_state][group]

                return current_state in [1, 4, 7]


63) DFA part 2:

    1411. Number of Ways to Paint N × 3 Grid
        Hard
        Topics
        Companies
        Hint
        You have a grid of size n x 3 and you want to paint each cell of the grid with exactly
        one of the three colors: Red, Yellow, or Green while making sure that no two adjacent 
        cells have the same color (i.e., no two cells that share vertical or horizontal sides have the same color).

        Given n the number of rows of the grid, return the number of ways you can paint this grid. 
        As the answer may grow large, the answer must be computed modulo 109 + 7.

        
        Example 1:


        Input: n = 1
        Output: 12
        Explanation: There are 12 possible way to paint the grid as shown.
        this is because:

        RYG RYR GRY... GYG 

        3 choices for first one.. then 2 choices for next one, then 2 choices again right? yes.  == 12 counting theroy!
        _ _ _
        2 
        - - -
        To count.. -> fill first row -> then -> fill next row right based on above row?   So a state is RYG or whatever 12 states in each row!  

        any of these 12 states -> forces a different state an allowed set of states  -> and the same for next row.. 



        
        Example 2:

        


        Input: n = 5000
        Output: 30228214


        Explanation
        Two pattern for each row, 121 and 123.
        121, the first color same as the third in a row.
        123, one row has three different colors.

        We consider the state of the first row,
        pattern 121: 121, 131, 212, 232, 313, 323.
        pattern 123: 123, 132, 213, 231, 312, 321.
        So we initialize a121 = 6, a123 = 6.

        We consider the next possible for each pattern:
        Patter 121 can be followed by: 212, 213, 232, 312, 313
        Patter 123 can be followed by: 212, 231, 312, 232

        121 => three 121, two 123
        123 => two 121, two 123

        So we can write this dynamic programming transform equation:
        b121 = a121 * 3 + a123 * 2
        b123 = a121 * 2 + a123 * 2

        We calculate the result iteratively
        and finally return the sum of both pattern a121 + a123

        def numOfWays(self, n):
            a121, a123, mod = 6, 6, 10**9 + 7
            for i in xrange(n - 1):
                a121, a123 = a121 * 3 + a123 * 2, a121 * 2 + a123 * 2
            return (a121 + a123) % mod



64) Boyer moor voting problem II

    Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

        Example 1:

        Input: nums = [3,2,3]
        Output: [3]
        Example 2:

        Input: nums = [1]
        Output: [1]
        Example 3:

        Input: nums = [1,2]
        Output: [1,2]

    class Solution:

        def majorityElement(self, nums):
            if not nums:
                return []
            
            # 1st pass
            count1, count2, candidate1, candidate2 = 0, 0, None, None
            for n in nums:
                if candidate1 == n:
                    count1 += 1
                elif candidate2 == n:
                    count2 += 1
                elif count1 == 0:
                    candidate1 = n
                    count1 += 1
                elif count2 == 0:
                    candidate2 = n
                    count2 += 1
                else:
                    count1 -= 1
                    count2 -= 1
            
            # 2nd pass
            result = []
            for c in [candidate1, candidate2]:
                if nums.count(c) > len(nums)//3:
                    result.append(c)

            return result



65) 1539. Kth Missing Positive Number (Binary Search)
        Attempted
        Easy
        Topics
        Companies
        Hint
        Given an array arr of positive integers sorted in a strictly increasing order, and an integer k.

        Return the kth positive integer that is missing from this array.



        Example 1:

        Input: arr = [2,3,4,7,11], k = 5
        Output: 9
        Explanation: The missing positive integers are [1,5,6,8,9,10,12,13,...]. The 5th missing positive integer is 9.
        Example 2:

        Input: arr = [1,2,3,4], k = 2
        Output: 6
        Explanation: The missing positive integers are [5,6,7,...]. The 2nd missing positive integer is 6.

        class Solution:
            def findKthPositive(self, arr: List[int], k: int) -> int:
                """
                To find the kth missing one..
                just start counting from 1. 
                and keep track. 
                You can also I guess.. binary search for it .
                Check the last element.. and see how off it is.. how many misisng numbers should there be at this value?
                if theres more than K?you gutta keep going left. 
                check mid point see how many missing should be from here? 
                once you find the left and right of the binary search then...
                go from left and keep adding 1 until you find it? ig?
                """ 
                l = 0 
                r = len(arr)
                def missing_amt(idx, number):
                    # at array idx how many are missing ? are 0 missing? 
                    return number - (idx + 1)

                while l < r:
                    
                    mid = l + (r-l) // 2    
                    # check a condition
                    val = arr[mid]
                    missing = missing_amt(mid, val)
                    # how many are missing at this point
                    if missing < k:
                        # shift to right side 
                        l = mid + 1
                    elif missing > k:
                        r = mid 
                    else:
                        break

                return l + k





Cool Notes Part 0.4 ##################################
 ################################## ##################################

SORTED CONTAINERS USAGE PYTHON.

        READ DOCUMTATION FOR SORTED DICT:
        https://grantjenks.com/docs/sortedcontainers/sorteddict.html#sorteddict

        demonstrated in below prblem.

        Binary Tree Map:

        651 · Binary Tree Vertical Order Traversal

        Description
        Given a binary tree, return the vertical order traversal of its 
        nodes' values. (ie, from top to bottom, column by column).

        If two nodes are in the same row and column, the order should be from left to right.

        For each node at position (row, col), its left and right children will be at positions 
        (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).

        Inpurt:  {3,9,20,#,#,15,7}
        Output: [[9],[3,15],[20],[7]]
        Explanation:
         3
        /\
        /  \
        9  20
            /\
        /  \
        15   7
        Example2

        Input: {3,9,8,4,0,1,7}
        Output: [[4],[9],[3,0,1],[8],[7]]
        Explanation:
            3
            /\
        /  \
        9   8
        /\  /\
        /  \/  \
        4  01   7

        from collections import defaultdict, deque
        from sortedcontainers import SortedDict

        class Solution:
            """
            @param root: the root of tree
            @return: the vertical order traversal
            """
            def vertical_order_okay(self, root: TreeNode) -> List[List[int]]:
                """
                Keep track of what level youre on!
                going left makes it negative.. going right is positive. 

                -3,-2,-1,0,1,2

                The above solution uses preorder traversal but that is wrong..
                it works for a lot of cases, but just messes up the ordering within 
                the vertical.
                """
                res = defaultdict(list)
                
                # if you do preorder traversal, it should keep the verticals ordered. 
                # ^ THIS IS A VERY INCORRECT ASSUMPTION
                
                start = float("inf")  #10000000
                end = float("-inf") # -1000000
                # ^ ALSO USING FLOAT HERE CAN FUCK STUFF UP UNLESS YOU GUARD FOR IT!
                # OTHERWISE SHOULD USE NONE HERE SO THAT RANGE FUNCTION ERRORS BELOW. 
                def helper(node, col):
                    nonlocal start
                    nonlocal end 

                    if node is None:
                        return 
                    
                    start = min(start, col)
                    end = max(end, col)

                    res[col].append(node.val)
                    helper(node.left, col-1)
                    helper(node.right, col+1)

                if root is None: 
                    # otherwise it will try to use inf in the range function so guard!
                    return []
                    
                helper(root, 0)

                # convert res to dictionary 
                ans = []
                for i in range(int(start), int(end+1) ):
                    ans.append(res[i])
                return ans


            def vertical_order(self, root: TreeNode) -> List[List[int]]:
                # use bfs for proper ordering of the vertical leveling. 
                d = deque([(root, 0)])
                res = SortedDict()


                while len(d) > 0:
                    
                    n, col = d.popleft()

                    if n:
                        res.setdefault(col, [])
                            
                        res[col].append(n.val)
                        d.append((n.left, col-1))
                        d.append((n.right, col+1))
                # print(list(res.values()) )
                # sorted containers will do the sorting of the keys for you!

                
                # if you dont do list() it will return a SortedDictValuesView -> need to type cast it.
                return list(res.values())

+-70) Example of extracting dynamic programming traversal paths 
+     after doing DP problem.
+        '''
+        CombinationSum:
+        Given an array of integers a and an integer sum, find all of the 
+        unique combinations in a that add up to sum.
+        The same number from a can be used an unlimited number of times in 
+        a combination.
+        Elements in a combination (a1 a2 … ak) must be sorted in non-descending order, 
+        while the combinations themselves must be sorted in ascending order.
+        If there are no possible combinations that add up to sum, 
+        the output should be the string "Empty".
+
+        Example
+
+        For a = [2, 3, 5, 9] and sum = 9, the output should be
+        combinationSum(a, sum) = "(2 2 2 3)(2 2 5)(3 3 3)(9)".
+
+
+        The DP problem is simple, done previously before HARMAN!!
+
+        Here we try to return the paths themselves, that were traversed in the DP
+        2 ways to do so:
+        A parents map OR as we save our results in the DP array, we also save our paths in a DP paths array.
+        Look at both methods and learn!!
+
+        '''
+        from collections import defaultdict, deque
+        def combinationSum(a, sum):
+            # parents map? 
+            g = defaultdict(list)
+            
+            # sort a and deduplicate. 
+            
+            a = sorted(list(set(a)))
+            
+            # we could also space optimize and just use One D array, because new 
+            # index doesnt use just previous index, but all previous indexes.
+            # so include all of em. 
+            OPT = [[0 for i in range(sum+1)]]
+            OPT[0][0] = 1
+            
+            
+            dp_paths = [[] for i in range(sum+1)]
+            dp_paths[0].append([])
+            
+            for idx, coinVal in enumerate(a):
+                # to compute for current index, 
+                # first copy previous, then operate on current. 
+                curr = OPT[-1][:]
+                '''
+                idx, coin?
+                '''
+                for i in range(sum+1):
+                    if i >= coinVal:
+                        # do we specify the coin type we used??
+                        # depends if we built from previous index, or 
+                        # coins from this index.  -> cant you use difference in amts
+                        # to determine coins -> YESS.
+                        # you dont need to save coinVal
+                        curr[i] += curr[i-coinVal]
+                        # can we save it, as we build the dp?
+                        
+                        parent_paths = dp_paths[i-coinVal]
+                        for p in parent_paths:
+                            cp = p[::]
+                            cp.append(coinVal)
+                            dp_paths[i].append(cp)
+
+                        if(curr[i-coinVal] > 0):
+                            g[i].append(i-coinVal)
+                                
+                OPT.append(curr)
+            
+            # DP PATHS WORKS HOW YOU EXPECT. IF OPT[sum] = 6, then in DP paths there is 6 paths.
+            print("DP_PATHS", dp_paths)
+            print("OPT", OPT)
+            
+            '''
+            Problem with getting all paths: we end up with all permutations instead of 
+            combinations: 
+            
+            Output: "(2 2 2 2)(2 2 4)(2 4 2)(2 6)(4 2 2)(4 4)(6 2)(8)"
+            Expected Output: "(2 2 2 2)(2 2 4)(2 6)(4 4)(8)"
+            SO WE NEED LIMIT ARGUMENT.
+            '''
+            
+            results = []
+            
+            def get_all_paths(node, path, limit):
+                kids = g[node]
+                if len(kids) == 0:
+                    # nonlocal results
+                    results.append(path)
+                
+                # USING A LIMIT ALLOWS YOU TO TURN 
+                # PERMUTATONS INTO COMBINATIONS IF ITS SORTED.
+                # BY TRAVERSING COINS FROM LARGEST TO SMALLEST ONLY. 
+                
+                for k in kids:
+                    coinVal = node-k
+                    if coinVal <= limit:
+                        cp = path.copy()
+                        cp.appendleft(coinVal)
+                        get_all_paths(k, cp, min(limit, coinVal))
+                        
+            get_all_paths(sum, deque([]), float("inf"))
+            final=[]
+            
+            # Uncomment this line and code still creates correct output!
+            # results = dp_paths[sum]
+
+            for r in results:
+                if len(r) == 0:
+                    continue
+                s = str(r[0])
+                for idx in range(1, len(r)):
+                    s += " " + str(r[idx])
+                final.append(s)
+            
+            final.sort()
+            
+            if len(final) == 0:
+                return "Empty"
+                
+            last = ")(".join(final)
+            return "(" + last + ")" 


#############################################################################
###################################################33

Cool Notes Part 0.5: Sliding Window with a deque
        -> In this question you learn about the difference between useless elements, 
           and useful elements.
           How to use a deque to maintain the useful elemenets as you run through the arry
           Operating on the indexes of an array instead of the actual elements (pointers!)
           Always try to realized how to discern useless and useful elements when sliding your window,
           and how to keep track of them and maintain them
           (In this case, every iteration we cull bad elements!)

        -> Sliding Window Maximum (Maximum of all subarrays of size k)
           Given an array and an integer k, find the maximum for each 
           and every contiguous subarray of size k. O(N) algorithm

           We create a Deque, Qi of capacity k, that stores only useful elements 
           of current window of k elements. An element is useful if it is in current window 
           and is greater than all other elements on left side of it in current window. 
           We process all array elements one by one and maintain Qi to contain 
           useful elements of current window and these useful elements are 
           maintained in sorted order. The element at front of the Qi is 
           the largest and element at rear of Qi is the smallest of current window. 
           Time Complexity: O(n). It seems more than O(n) at first look. 
           If we take a closer look, we can observe that every element of 
           array is added and removed at most once. 
           So there are total 2n operations.
          
            def printMax(arr, n, k): 
                
                """ Create a Double Ended Queue, Qi that  
                will store indexes of array elements.  
                The queue will store indexes of useful  
                elements in every window and it will 
                maintain decreasing order of values from 
                front to rear in Qi, i.e., arr[Qi.front[]] 
                to arr[Qi.rear()] are sorted in decreasing 
                order"""
                Qi = deque() 
                
                # Process first k (or first window)  
                # elements of array 
                for i in range(k): 
                    
                    # For every element, the previous  
                    # smaller elements are useless 
                    # so remove them from Qi 
                    while Qi and arr[i] >= arr[Qi[-1]] : 
                        Qi.pop() 
                    
                    # Add new element at rear of queue 
                    Qi.append(i); 
                    # We are storing the indexes for the biggest elements!
                
                # Qi contains the biggest elements in the first k so:
                # k -> [1, 4, 6, 3, 5] (5 elements)
                # Qi -> [2, 5] (useful elements -> 
                #              the indexes for elements that can become a maximum)
                # element at front of queue is maximum, in this case, it is 6 (index is 2)

                # Process rest of the elements, i.e.  
                # from arr[k] to arr[n-1] 
                for i in range(k, n): 
                    
                    # The element at the front of the 
                    # queue is the largest element of 
                    # previous window, so print it 
                    print(str(arr[Qi[0]]) + " ", end = "") 
                    
                    # Remove our really good candidates which are  
                    # out of window now. SO SAD!! 
                    # out of window elements are in the front of the queue. 
                    # indexes are increasing like above -> 2 -> 5
                    # its i-k because we want to shift our removal range to start at index 0 
                    # and go up to n-k for the last window.
                    while Qi and Qi[0] <= i-k: 
                        
                        # remove from front of deque 
                        Qi.popleft()  
                    
                    # Remove all elements smaller than 
                    # the currently being added element  
                    # (Remove useless elements) 
                    # we can do this because the element we are adding will always
                    # be a candidate for the sliding window in the next iteration, 
                    # and a better candidate, than the  "useless" elements in the sliding
                    # window
                    while Qi and arr[i] >= arr[Qi[-1]] : 
                        Qi.pop() 
                    
                    # Add current element at the rear of Qi 
                    Qi.append(i) 
                
                # Print the maximum element of last window 
                print(str(arr[Qi[0]])) 
        
        -> YOU CAN ALSO ANSWER THIS QUESTION WITH A SEGMENT TREE. 

        OTHER TAKEAWAYS:
        This queue is a monoqueue
        What does Monoqueue do here:

        It has three basic options:
        push: push an element into the queue; O (1) (amortized)
        pop: pop an element out of the queue; O(1) (pop = remove, it can't report this element)
        max: report the max element in queue;O(1)

        It takes only O(n) time to process a N-size sliding window minimum/maximum problem.
        Note: different from a priority queue (which takes O(nlogk) to solve this problem), 
        it doesn't pop the max element: It pops the first element (in original order) in queue.

###################################################################################
##################################################################################

[Python] Powerful Ultimate Binary Search Template. Solved many problems


    Binary Search helps us reduce the search time from linear O(n) to logarithmic O(log n). 

    >> Most Generalized Binary Search
    Suppose we have a search space. It could be an array, a range, etc. 
    
    Usually it's sorted in ascending order. For most tasks, 
    we can transform the requirement into the following generalized form:

    Minimize k , s.t. condition(k) is True

    The following code is the most generalized binary search template:

    def binary_search(array) -> int:
        def condition(value) -> bool:
            pass

        left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        return left

    What's really nice of this template is that, for most of the binary search problems, 
    we only need to modify three parts after copy-pasting this template, 
    and never need to worry about corner cases and bugs in code any more:

    Correctly initialize the boundary variables left and right to specify search space. 
    Only one rule: set up the boundary to include all possible elements;
    
    Decide return value. Is it return left or return left - 1? 

    Remember this: after exiting the while loop, left is the minimal k​ satisfying the condition function;
    Design the condition function. This is the most difficult and most beautiful part. Needs lots of practice.
    Below I'll show you guys how to apply this powerful template to many LeetCode problems.

    "Remember this: after exiting the while loop, left is the minimal k satisfying the condition function" 
    This is not 100% accurate. In the case that left == array size after exiting the loop, 
    no element in the array satisfy the condition function.


    Excellent work! Many people think sorted array is a must to apply 
    binary search, which is not 100% correct. In some cases, there is no 
    such array, or the array is not sorted, or the element are not even 
    comparable! What makes binary search work is that there exists a function 
    that can map elements in left half to True, and the other half to False, 
    or vice versa. If we can find such a function, we can apply bs to find 
    the boundary (lower_bound for example). For the interval notation, 
    Professor E.W. Dijkstra favors left closed right open interval 
    notation and explained why we benefit from this notation in his 
    post which was published in 1982.

    In short, loop invariants help us design a loop and ensure the 
    correctness of a loop in a formal way. ref1, ref2. For the binary search code, 
    if we imagine the array expanded to infinity for both sides, then the loop 
    invariants can phrased as:1) l<=r 2) elements in (-inf, l) are mapped to 
    False by condition 3) elements in [r, +inf) are mapped to True by condition. 
    These invariants are true before the loop, in each iteration of 
    the loop, and after the loop. After the loop breaks, we know l ==r, 
    from these invariants, we can conclude that elements in (-inf, l) are False, 
    and those in [l, +inf) are true. Besides, to ensure a loop is 
    correct, we also need to prove the loop break eventually. 
    The proof is straight forward: at each iteration, the search 
    range [l, r) shrinks by at least 1.


    1.   First Bad Version [Easy]
       You are a product manager and currently leading a team to develop a new 
       product. Since each version is developed based on the previous version, 
       all the versions after a bad version are also bad. Suppose you have n 
       versions [1, 2, ..., n] and you want to find out the first bad one, which 
       causes all the following ones to be bad. You are given an API bool 
       isBadVersion(version) which will return whether version is bad.

       Example:

       Given n = 5, and version = 4 is the first bad version.

       call isBadVersion(3) -> false
       call isBadVersion(5) -> true
       call isBadVersion(4) -> true

       Then 4 is the first bad version. 
       First, we initialize left = 1 and right = n to include all possible values. 
       Then we notice that we don't even need to design the condition function. 
       It's already given by the isBadVersion API. Finding the first bad version is 
       equivalent to finding the minimal k satisfying isBadVersion(k) is True. 
       Our template can fit in very nicely:

       class Solution:
           def firstBadVersion(self, n) -> int:
               left, right = 1, n
               while left < right:
                   mid = left + (right - left) // 2
                   if isBadVersion(mid):
                       right = mid
                   else:
                       left = mid + 1
               return left


    2.  Sqrt(x) [Easy]
       Implement int sqrt(int x). Compute and return the square root of x, 
       where x is guaranteed to be a non-negative integer. Since the return type 
       is an integer, the decimal digits are truncated and only the 
       integer part of the result is returned.

       Example:

       Input: 4
       Output: 2
       Input: 8
       Output: 2
       Easy one. First we need to search for minimal k satisfying 
       condition k^2 > x, then k - 1 is the answer to the question. 
       We can easily come up with the solution. Notice that I set 
       right = x + 1 instead of right = x to deal with special input 
       cases like x = 0 and x = 1.

       def mySqrt(x: int) -> int:
           left, right = 0, x + 1
           while left < right:
               mid = left + (right - left) // 2
               if mid * mid > x:
                   right = mid
               else:
                   left = mid + 1
           return left - 1  # `left` is the minimum k value, `k - 1` is the answer


    3.  Search Insert Position [Easy]
       Given a sorted array and a target value, return the index if the 
       target is found. If not, return the index where it would be if it were 
       inserted in order. You may assume no duplicates in the array.

       Example:

       Input: [1,3,5,6], 5
       Output: 2
       Input: [1,3,5,6], 2
       Output: 1

       Very classic application of binary search. We are looking for the 
       minimal k value satisfying nums[k] >= target, and we can just 
       copy-paste our template. Notice that our solution is correct regardless 
       of whether the input array nums has duplicates. Also notice that the input 
       target might be larger than all elements in nums and therefore 
       needs to placed at the end of the array. 
       That's why we should initialize right = len(nums) instead of right = len(nums) - 1.

       class Solution:
           def searchInsert(self, nums: List[int], target: int) -> int:
               left, right = 0, len(nums)
               while left < right:
                   mid = left + (right - left) // 2
                   if nums[mid] >= target:
                       right = mid
                   else:
                       left = mid + 1
               return left

    >> Advanced Application
    The above problems are quite easy to solve, because they 
    already give us the array to be searched. We'd know that we should 
    use binary search to solve them at first glance. However, more often 
    are the situations where the search space and search target are not 
    so readily available. Sometimes we won't even realize that the problem 
    should be solved with binary search -- we might just turn to dynamic 
    programming or DFS and get stuck for a very long time.

    As for the question "When can we use binary search?", my answer 
    is that, If we can discover some kind of monotonicity, for example, 
    if condition(k) is True then condition(k + 1) is True, 
    then we can consider binary search.


    1.    Capacity To Ship Packages Within D Days [Medium]
        A conveyor belt has packages that must be shipped from 
        one port to another within D days. The i-th package on the 
        conveyor belt has a weight of weights[i]. Each day, 
        we load the ship with packages on the conveyor belt 
        (in the order given by weights). We may not load more 
        weight than the maximum weight capacity of the ship.

        Return the least weight capacity of the ship that 
        will result in all the packages on the conveyor 
        belt being shipped within D days.

        Example :

        Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5
        Output: 15
        Explanation: 
        A ship capacity of 15 is the minimum to ship 
            all the packages in 5 days like this:
        1st day: 1, 2, 3, 4, 5
        2nd day: 6, 7
        3rd day: 8
        4th day: 9
        5th day: 10

        Note that the cargo must be shipped in the order given, 
        so using a ship of capacity 14 and splitting 
        the packages into parts like (2, 3, 4, 5), 
        (1, 6, 7), (8), (9), (10) is not allowed.
        
        SOLN:
        Start with a high capacity that works -> then keep reducing it until 
        we are at an integer that causes D to go from viable -> not viable
        Since thats the linear search solution -> Try binary
        
        Try total sum -> then half sum -> then quarter sum -> and keep going 
        until D == 5
        But how do verify a particular weight -> In O(N) by running it through array?
        We should be able to do faster by doing a type of DP rite? idk.


        Binary search probably would not come to our mind when we first meet 
        this problem. We might automatically treat weights as search space and 
        then realize we've entered a dead end after wasting lots of time. 
        In fact, we are looking for the minimal one among all feasible capacities. 
        We dig out the monotonicity of this problem: if we can successfully 
        ship all packages within D days with capacity m, then we can definitely 
        ship them all with any capacity larger than m. Now we can design a 
        condition function, let's call it feasible, given an input capacity, 
        it returns whether it's possible to ship all packages within D days. 
        This can run in a greedy way: if there's still room for the current 
        package, we put this package onto the conveyor belt, otherwise we 
        wait for the next day to place this package. If the total days 
        needed exceeds D, we return False, otherwise we return True.

        Next, we need to initialize our boundary correctly. Obviously 
        capacity should be at least max(weights), otherwise the conveyor 
        belt couldn't ship the heaviest package. On the other hand, capacity 
        need not be more thansum(weights), because then we can 
        ship all packages in just one day.

        Now we've got all we need to apply our binary search template:

        def shipWithinDays(weights: List[int], D: int) -> int:
            def feasible(capacity) -> bool:
                days = 1
                total = 0
                for weight in weights:
                    total += weight
                    if total > capacity:  # too heavy, wait for the next day
                        total = weight
                        days += 1
                        if days > D:  # cannot ship within D days
                            return False
                return True

            left, right = max(weights), sum(weights)
            while left < right:
                mid = left + (right - left) // 2
                if feasible(mid):
                    right = mid
                else:
                    left = mid + 1
            return left


    2.   Split Array Largest Sum [Hard]
        Given an array which consists of non-negative integers and an 
        integer m, you can split the array into m non-empty continuous 
        subarrays. Write an algorithm to minimize the largest sum 
        among these m subarrays.

        Input:
        nums = [7,2,5,10,8]
        m = 2

        Output:
        18

        Explanation:
        There are four ways to split nums into two subarrays. 
        The best way is to split it into [7,2,5] and [10,8], 
        where the largest sum among the two subarrays is only 18.
        
        Are these problems like rotated DP? or binsearch on DP?
        How are we avoiding doing DP?

        def answer(nums, m):

            def viable(nums, allowedSum):
                # there is always 1 section
                sections = 1
                curr = 0

                for i in nums:
                    # curr += i
                    if curr + i  > allowedSum:
                        sections += 1
                        curr = i 
                        if sections > m:
                            return False
                

                return True
            
            # do binsearch 
            i, j = max(nums), sum(nums)
            while i < j:
                mid = i  + (j-i)//2

                if viable(nums, mid):
                    j = mid
                else:
                    i = mid + 1

            return i

        If you take a close look, you would probably see how similar 
        this problem is with LC 1011 above. Similarly, we can design a 
        feasible function: given an input threshold, then decide if we can 
        split the array into several subarrays such that every subarray-sum 
        is less than or equal to threshold. In this way, we discover the 
        monotonicity of the problem: if feasible(m) is True, then all inputs 
        larger than m can satisfy feasible function. You can see that the 
        solution code is exactly the same as LC 1011.
        Their soln:
        
        def splitArray(nums: List[int], m: int) -> int:        
            def feasible(threshold) -> bool:
                count = 1
                total = 0
                for num in nums:
                    total += num
                    if total > threshold:
                        total = num
                        count += 1
                        if count > m:
                            return False
                return True

            left, right = max(nums), sum(nums)
            while left < right:
                mid = left + (right - left) // 2
                if feasible(mid):
                    right = mid     
                else:
                    left = mid + 1
            return left
    

    3.   Koko Eating Bananas [Medium]
        Koko loves to eat bananas. There are N piles of bananas, 
        the i-th pile has piles[i] bananas. The guards have gone and 
        will come back in H hours. Koko can decide her bananas-per-hour 
        eating speed of K. Each hour, she chooses some pile of bananas, 
        and eats K bananas from that pile. If the pile has less than K 
        bananas, she eats all of them instead, and won't eat any more 
        bananas during this hour.

        Koko likes to eat slowly, but still wants to finish eating 
        all the bananas before the guards come back. Return the minimum 
        integer K such that she can eat all the bananas within H hours.

        Example :

        Input: piles = [3,6,7,11], H = 8
        Output: 4
        Input: piles = [30,11,23,4,20], H = 5
        Output: 30
        Input: piles = [30,11,23,4,20], H = 6
        Output: 23

        Very similar to LC 1011 and LC 410 mentioned above. 
        Let's design a feasible function, given an input speed, 
        determine whether Koko can finish all bananas within H hours 
        with hourly eating speed speed. Obviously, the lower bound 
        of the search space is 1, and upper bound is max(piles), 
        because Koko can only choose one pile of bananas to eat every hour.

        def minEatingSpeed(piles: List[int], H: int) -> int:
            def feasible(speed) -> bool:
                # return sum(math.ceil(pile / speed) for pile in piles) <= H  # slower        
                return sum((pile - 1) // speed + 1 for pile in piles) <= H  # faster

            left, right = 1, max(piles)
            while left < right:
                mid = left  + (right - left) // 2
                if feasible(mid):
                    right = mid
                else:
                    left = mid + 1
            return left
    
    4.    Minimum Number of Days to Make m Bouquets [Medium]
          
        Given an integer array bloomDay, an integer m and an integer k. 
        We need to make m bouquets. To make a bouquet, you need to use 
        k adjacent flowers from the garden. The garden consists of n flowers, 
        the ith flower will bloom in the bloomDay[i] and then can be 
        used in exactly one bouquet. Return the minimum number of 
        days you need to wait to be able to make m bouquets from the garden. 
        If it is impossible to make m bouquets return -1.
        Examples:
        Input: bloomDay = [1,10,3,10,2], m = 3, k = 1
        Output: 3
        Explanation: Let's see what happened in the first three days. 
        x means flower bloomed and _ means flower didn't bloom in the garden.
        We need 3 bouquets each should contain 1 flower.
        After day 1: [x, _, _, _, _]   // we can only make one bouquet.
        After day 2: [x, _, _, _, x]   // we can only make two bouquets.
        After day 3: [x, _, x, _, x]   // we can make 3 bouquets. The answer is 3.
        
        Input: bloomDay = [1,10,3,10,2], m = 3, k = 2
        Output: -1
        Explanation: We need 3 bouquets each has 2 flowers, 
        that means we need 6 flowers. We only have 5 flowers 
        so it is impossible to get the needed bouquets and we return -1.
        
        You can either enumerate all k length intervals, and m of them. 
        Which may be hard?
        Or we create a viable function and feed in an argument that is binary searched. 
        ...
        Algo:
        Check if its viable first, if theres enough flowers. 
        Viable function -> specify max time to wait. 
        Which is between min(Array) and max(Array) inclusive. 
        Then create interval -> it cant accept values that are bigger, 
        otherwise create new interval -> if you werent able to create atleast m, then 
        not viable. 
        Binary search to find first viable?
        Now that we've solved three advanced problems above, 
        this one should be pretty easy to do. The monotonicity 
        of this problem is very clear: if we can make m bouquets 
        after waiting for d days, then we can definitely finish 
        that as well if we wait for more than d days.

        def minDays(bloomDay: List[int], m: int, k: int) -> int:
            def feasible(days) -> bool:
                bonquets, flowers = 0, 0
                for bloom in bloomDay:
                    if bloom > days:
                        flowers = 0
                    else:
                        bonquets += (flowers + 1) // k
                        flowers = (flowers + 1) % k
                return bonquets >= m

            if len(bloomDay) < m * k:
                return -1
            left, right = 1, max(bloomDay)
            while left < right:
                mid = left + (right - left) // 2
                if feasible(mid):
                    right = mid
                else:
                    left = mid + 1
            return left
                    


    5.   Kth Smallest Number in Multiplication Table [Hard]
        Nearly every one have used the Multiplication Table. 
        But could you find out the k-th smallest number quickly 
        from the multiplication table? Given the height m and the 
        length n of a m * n Multiplication Table, and a positive 
        integer k, you need to return the k-th smallest number in this table.

        Example :

        Input: m = 3, n = 3, k = 5
        Output: 3
        Explanation: 
        The Multiplication Table:
        1	2	3
        2	4	6
        3	6	9

        The 5-th smallest number is 3 (1, 2, 2, 3, 3).


            5th smallest. Can we do a quick select? 
            so partition, then look for value in left or right partion 
            -> worst case O(n^2) -> best case O(n)
            
        
        For Kth-Smallest problems like this, what comes to our mind 
        first is Heap. Usually we can maintain a Min-Heap and just 
        pop the top of the Heap for k times. However, that doesn't 
        work out in this problem. We don't have every single number in 
        the entire Multiplication Table, instead, we only have the height 
        and the length of the table. If we are to apply Heap method, 
        we need to explicitly calculate these m * n values and save 
        them to a heap. The time complexity and space complexity of this 
        process are both O(mn), which is quite inefficient. This is 
        when binary search comes in. Remember we say that designing condition 
        function is the most difficult part? In order to find the k-th smallest 
        value in the table, we can design an enough function, given an input num, 
        determine whether there're at least k values less than or 
        equal to num. The minimal num satisfying enough function is the 
        answer we're looking for. Recall that the key to binary search 
        is discovering monotonicity. In this problem, if num satisfies 
        enough, then of course any value larger than num can satisfy. 
        This monotonicity is the fundament of our binary search algorithm.

        Let's consider search space. Obviously the lower bound should be 1, 
        and the upper bound should be the largest value in the Multiplication 
        Table, which is m * n, then we have search space [1, m * n]. The 
        overwhelming advantage of binary search solution to heap solution 
        is that it doesn't need to explicitly calculate all numbers in that 
        table, all it needs is just picking up one value out of the 
        search space and apply enough function to this value, to determine 
        should we keep the left half or the right half of the search 
        space. In this way, binary search solution only requires constant 
        space complexity, much better than heap solution.

        Next let's consider how to implement enough function. 
        It can be observed that every row in the Multiplication Table 
        is just multiples of its index. For example, all numbers in 
        3rd row [3,6,9,12,15...] are multiples of 3. Therefore, 
        we can just go row by row to count the total number of entries 
        less than or equal to input num. Following is the complete solution.
        (Could probably binary search it form each row? 
        nah cause we still have to check
        each row, so speed benefit is trumped by that.)

        def findKthNumber(m: int, n: int, k: int) -> int:
            def enough(num) -> bool:
                count = 0
                for val in range(1, m + 1):  # count row by row
                    add = min(num // val, n)
                    if add == 0:  # early exit
                        break
                    count += add
                return count >= k                

            left, right = 1, n * m
            while left < right:
                mid = left + (right - left) // 2
                if enough(mid):
                    right = mid
                else:
                    left = mid + 1
            return left 
                
    6.   Find K-th Smallest Pair Distance [Hard]
       Given an integer array, return the k-th smallest distance 
       among all the pairs. The distance of a pair (A, B) is 
       defined as the absolute difference between A and B.

       Example :

       Input:
       nums = [1,3,1]
       k = 1
       Output: 0 
       Explanation:
       Following are all the pairs. The 1st smallest 
        distance pair is (1,1), and its distance is 0.
       (1,3) -> 2
       (1,1) -> 0
       (3,1) -> 2

       Ok so sort it, [1,1,3]
       -> now 1,1 1,3 1,3 -> 0, 2, 2
       Number of pairs is 3 * 2  / 2!  = 6/2 = 3   3*2  permutations / 2! to remove perms.  
       Lets binary search the space. 
       min -> difference between 1st and 2nd pair.
           -> largest diff -> 1st and last pair


    how many pairs between index 0 and index 4?
    [1,1,3,5,7]

        Enough function 
        
        -> Use 2 pointers and check when the difference 
        is too much 
        keep first ptr on left most, then increment right
        when difference gets too big, then start reducing first ptr?

        Very similar to LC 668 above, both are about finding Kth-Smallest. Just like LC 668, 
        We can design an enough function, given an input distance, determine whether 
        there're at least k pairs whose distances are less than or 
        equal to distance. We can sort the input array and use two pointers 
        (fast pointer and slow pointer, pointed at a pair) to scan it. 
        Both pointers go from leftmost end. If the current pair pointed 
        at has a distance less than or equal to distance, all pairs between 
        these pointers are valid (since the array is already sorted), we move 
        forward the fast pointer. Otherwise, we move forward the slow pointer. 
        By the time both pointers reach the rightmost end, we finish our 
        scan and see if total counts exceed k. Here is the implementation:

        ENOUGH FUNCTION IS O(N) DO YOU KNOW WHY!
        MEMORIZE IT!!! SLIDING WINDOW, 
        It is O(N). the possible function is a classic sliding windowing solution, 
        left and right would always increment in each outer loop iteration. 
        So time complexity is O(2N) = O(N).

        [1,3,6,10,15, END]
         ^         ^
        Lets say dist is 11 so ptrs end up like above. 
        Count is 4-0-1 = 3 pairs = 2 + 1
        
        [1,3,6,10,15,END]
           ^       ^
        Lets say dist is 11 so ptrs end up like above. 
        Count is 4-1-1 = 2 pairs          

        [1,3,6,10,15,END]
             ^       ^

        Lets say dist is 11 so ptrs end up like above. 
        Count is 5-2-1 = 2 pairs    

        def enough(distance) -> bool:  # two pointers
            count, i, j = 0, 0, 0
            while i < n or j < n:
                while j < n and nums[j] - nums[i] <= distance:  # move fast pointer
                    j += 1
                count += j - i - 1  # count pairs  # this counts pairs in sliding window like 1+2+3+4+5 == n*(n-1)/2 pairs
                                    # j is on the first element that fails the abve condition for the reason for (-1)
                i += 1  # move slow pointer # we finally moved i so we can start counting more pairs from here!.
            return count >= k
        
        Obviously, our search space should be [0, max(nums) - min(nums)]. 
        Now we are ready to copy-paste our template:

        def smallestDistancePair(nums: List[int], k: int) -> int:
            nums.sort()
            n = len(nums)
            left, right = 0, nums[-1] - nums[0]
            while left < right:
                mid = left + (right - left) // 2
                if enough(mid):
                    right = mid
                else:
                    left = mid + 1
            return left
                
        ANOTHER IMPLEMENTATION:

        class Solution(object):
            def smallestDistancePair(self, nums, k):
                def possible(guess):
                    #Is there k or more pairs with distance <= guess?
                    count = left = 0
                    for right, x in enumerate(nums):
                        while x - nums[left] > guess:
                            left += 1
                        count += right - left
                    return count >= k

                nums.sort()
                lo = 0
                hi = nums[-1] - nums[0]
                while lo < hi:
                    mi = (lo + hi) / 2
                    if possible(mi):
                        hi = mi
                    else:
                        lo = mi + 1
                return lo
        
    7.    Ugly Number III [Medium]
       Write a program to find the n-th ugly number. 
       Ugly numbers are positive integers which are 
       divisible by a or b or c.

       Example :

       Input: n = 3, a = 2, b = 3, c = 5
       Output: 4
       Explanation: The ugly numbers are 2, 3, 4, 5, 6, 8, 9, 10... The 3rd is 4.
       Input: n = 4, a = 2, b = 3, c = 4
       Output: 6
       Explanation: The ugly numbers are 2, 3, 4, 6, 8, 9, 10, 12... The 4th is 6.

        Nothing special. Still finding the Kth-Smallest. 
        We need to design an enough function, given an input num, 
        determine whether there are at least n ugly numbers less than or equal to num. Since a might be a multiple of b or c, or the other way round, 
        we need the help of greatest common divisor to avoid counting duplicate numbers.

        def nthUglyNumber(n: int, a: int, b: int, c: int) -> int:
            def enough(num) -> bool:
                // Triple set addition!
                total = mid//a + mid//b + mid//c - mid//ab - mid//ac - mid//bc + mid//abc
                return total >= n

            ab = a * b // math.gcd(a, b) # LCM
            ac = a * c // math.gcd(a, c) # LCM
            bc = b * c // math.gcd(b, c) # LCM
            abc = a * bc // math.gcd(a, bc)
            left, right = 1, 10 ** 10
            while left < right:
                mid = left + (right - left) // 2
                if enough(mid):
                    right = mid
                else:
                    left = mid + 1
            return left


    8.    Find the Smallest Divisor Given a Threshold [Medium]
       Given an array of integers nums and an integer threshold, 
       we will choose a positive integer divisor and divide all 
       the array by it and sum the result of the division. Find 
       the smallest divisor such that the result mentioned above 
       is less than or equal to threshold.

       Each result of division is rounded to the nearest integer 
       greater than or equal to that element. (For example: 7/3 = 3 
       and 10/2 = 5). It is guaranteed that there will be an answer.

       Example :

       Input: nums = [1,2,5,9], threshold = 6
       Output: 5
       Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1. 
       If the divisor is 4 we can get a sum to 7 (1+1+2+3) and 
       if the divisor is 5 the sum will be 5 (1+1+1+2). 


       After so many problems introduced above, this one 
       should be a piece of cake. We don't even need to bother 
       to design a condition function, because the problem has 
       already told us explicitly what condition we need to satisfy.

       def smallestDivisor(nums: List[int], threshold: int) -> int:
           def condition(divisor) -> bool:
               return sum((num - 1) // divisor + 1 for num in nums) <= threshold

           left, right = 1, max(nums)
           while left < right:
               mid = left + (right - left) // 2
               if condition(mid):
                   right = mid
               else:
                   left = mid + 1
           return left



    YOU CAN ALSO USE THE TEMPLATE TO MAXIMIZE THINGS!
    HERE IS AN EXAMPLE FROM THE COMMENTS IN THE POST:
    1231: Divide chocolate:

        You have one chocolate bar that consists of some chunks. 
        Each chunk has its own sweetness given by the array sweetness.

        You want to share the chocolate with your k friends so you start cutting the 
        chocolate bar into k + 1 pieces using k cuts, each piece consists of some consecutive chunks.

        Being generous, you will eat the piece with the minimum
        total sweetness and give the other pieces to your friends.

        Find the maximum total sweetness of the piece you can get 
        by cutting the chocolate bar optimally.

        Example 1:

        Input: sweetness = [1,2,3,4,5,6,7,8,9], k = 5
        Output: 6
        Explanation: You can divide the chocolate to [1,2,3], [4,5], [6], [7], [8], [9]
        Example 2:

        Input: sweetness = [5,6,7,8,9,1,2,3,4], k = 8
        Output: 1
        Explanation: There is only one way to cut the bar into 9 pieces.
        Example 3:

        Input: sweetness = [1,2,2,1,2,2,1,2,2], k = 2
        Output: 5
        Explanation: You can divide the chocolate to [1,2,2], [1,2,2], [1,2,2]

        THIS SOLUTION WORKS!
        class Solution:
            def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
                def possible(sweetness_limit: int) -> bool:
                    total_sweetness = 0
                    shared_with = 0
                    
                    for sweet in sweetness:
                        total_sweetness += sweet
                        
                        if total_sweetness > sweetness_limit:
                            total_sweetness = 0
                            shared_with += 1

                    # if at the current sweetness_limit we are choosing
                    # we haven't been successful in cutting the chocolate
                    # into k+1 pieces then that means we should try a lower sweetness to divide it by,
                    # because that would allow more cuts to be made as less subarray sums(cut chocolate pieces)
                    # will be able to fit in the current `sweetness_limit`
                    return shared_with > k
                
                low = min(sweetness) # the lowest is by taking the least sweetest chunk and dividing by that
                high = sum(sweetness) # the max we could cut the chocolate is by 1 piece that accounts for all the sweetness
                
                while low < high:
                    # mid is potential optimal sweetness
                    mid = low + (high - low) // 2
                    
                    if not possible(mid):
                        high = mid
                    else:
                        low = mid + 1
                return low


        THIS SOLUTION WORKS TOO, ONLY DIFFERRENCE IS HIGH IS SUM() + 1 THIS SOLUTION FROM ORIGINAL AUTHOR

        class Solution:
            def maximizeSweetness(self, nums: List[int], K: int) -> int:        
                def feasible(threshold) -> bool:
                    count = 0
                    total = 0
                    for num in nums:
                        total += num
                        if total >= threshold:
                            total = 0
                            count += 1
                            if count > K:
                                return True
                    return False

                left, right = 0, sum(nums) + 1
                while left < right:
                    mid = left + (right - left) // 2
                    if not feasible(mid):
                        right = mid
                    else:
                        left = mid + 1
                return left - 1

        REASON FOR SUM() + 1

        I think it's because your approach/template is assuming the search space to be left closed right open, [l, r). 
        If you specify l to be the min and r to be the max of a potential answer, there is a chance you might 
        miss the max potential answer. While searching for the min k such that it meets some condition as you 
        mentioned in all the questions above, your template won't miss it since as long as we meets condition, 
        we squeeze the search space to left, the new search space will become [l, m) so it won't get to 
        the max potential answer, which is the initialized r. But if you want to search for max k such condition is met, 
        you are squeezing the search space to right to [m, r) and this can possibly lead to your answer off by one. 
        Why we are returning left - 1 at the end as I mentioned in last reply. It's because when condition meets, 
        we will have left = mid + 1 (think it as mid = left - 1), 
        nums[left] will definitely not equal to the target when the loop ends (left == right), so we need to return left -1.

        If I alter your solution in one part which is change right to sum(nums)+1, it passes all test cases.
        Let me know if I have anything stated wrong. I am happy to further discuss with you.


    1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

        Given a m x n matrix mat and an integer threshold, return the maximum side-length 
        of a square with a sum less than or equal to threshold or return 0 if there is no such square.

        Example 1:


        Input: mat = [[1,1,3,2,4,3,2],
                      [1,1,3,2,4,3,2],
                      [1,1,3,2,4,3,2]], threshold = 4
        Output: 2
        Explanation: The maximum side length of square with sum less than 4 is 2 as shown.
        Example 2:

        Input: mat = [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]], threshold = 1
        Output: 0

        
        Algorithm:
        Build a (m+1)x(n+1) matrix of prefix sums of given matrix where prefix[i][j] = sum(mat[:i][:j]). 
        And search for maximum k such that sum(mat[i:i+k][j:j+k]) not exceeding threshold. Notice:

        prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j] + mat[i][j].
        sum(mat[i:i+k][j:j+k]) = prefix[i+k][j+k] - prefix[i][j+k] - prefix[i+k][j] + prefix[i][j].

            def maxSideLength(self, mat, threshold: int) -> int:
                m, n = len(mat), len(mat[0])
                prefix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        prefix[i][j] = mat[i - 1][j - 1] + prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1]

                def candidate(k) -> bool:
                    for i in range(k, m + 1):
                        for j in range(k, n + 1):
                            if prefix[i][j] + prefix[i - k][j - k] - prefix[i - k][j] - prefix[i][j - k] <= threshold:
                                return True
                    return False

                left, right = 0, min(m, n) + 1      # its min(m, n) + 1 because we want open right interval for this to work!
                                                    # also cause we return left -1 in the end?
                while left < right:
                    mid = left + (right - left) // 2
                    if not candidate(mid):
                        right = mid
                    else:
                        left = mid + 1
                return left - 1



###################################################################################
###################################################################################

BINARY SEARCH DIFFERENT TEMPLATES AND THEIR USE CASES!!!!!!

Tip
Personally,
If I want find the index, I always use while (left < right)
If I may return the index during the search, I'll use while (left <= right)


I like you tip, summary of 2 most frequently used binary search templates.
one is return index during the search:

while lo <= hi:
  mid = (lo+hi)/2
  if nums[mid] == target:
    return mid
  if nums[mid] > target:
    hi = mid-1
  else:
    lo = mid+1
return -1

Another more frequently used binary search template is for searching lowest 
element satisfy function(i) == True (the array should satisfy function(x) == False 
for 0 to i-1, and function(x) == True for i to n-1, and it is up to the question to 
define the function, like in the find peak element problem, function(x) can be nums[x] < nums[x+1] ), 
there are 2 ways to write it:

while lo <= hi:
  mid = (lo+hi)/2
  if function(mid):
    hi = mid-1
  else:
    lo = mid+1
return lo

or

while lo <  hi:
  mid = (lo+hi)/2
  if function(mid):
    hi = mid
  else:
    lo = mid+1
return lo

No matter which one you use, just be careful about updating the hi and lo, which 
could easily lead to infinite loop. Some binary question is searching a floating 
number and normally the question will give you a precision, in which case you 
don't need to worry too much about the infinite loop but your while 
condition will become something like "while lo+1e-7<hi"


For people who are wondering about the section of the post,

"If I want find the index, I always use while (left < right)
If I may return the index during the search, I'll use while (left <= right)"

Let's use the array a = [1,2,3] as an example, where we're searching for key = 3

If we know the index definitely exists, then we use while l < r

We first search 2, notice that 2 < key. So we set l = res = mid+1. 
We break the loop since l == r and return res. Because res is the 
only possible answer left, and since we know the index exists, we just return that.

Now if we don't know if the index exists, then we set l = mid+1 and 
only set res if a[mid] == key. We still have to check the final 
possibility, because we don't know whether or not that index contains the key.
######################################################
#############################################################################


COOL NOTES PART 0.8999: DYNAMIC PROGRAMMING PATTERNS, ILLUTRASTRAIONS AND EXAMPLES PART 1

    Dynamic Programming (DP) is a powerful algorithmic technique used to solve a wide range of optimization problems. It's a cornerstone of technical interviews, and LeetCode provides an extensive collection of DP problems to help you master this crucial skill. In this comprehensive guide, we'll explore LeetCode's top DP problems, dissecting their statements, approaches, solution strategies (Top-Down and Bottom-Up), and identifying similar problems across various DP patterns.

    Minimum Cost Climbing Stairs (Problem #746)
    Problem Statement: You are given an array cost where cost[i] represents the cost of climbing the i-th stair. 
    You can start climbing from either the 0-th or 1-st stair. Each time
     you can either climb one or two steps. Return the minimum cost to reach the top of the floor.

    Approach:
    This problem exhibits a classic Dynamic Programming pattern where the minimum cost to reach a stair 
    is the minimum of the costs of reaching the previous two stairs plus the cost of the current stair.

    Top-Down Approach (Recursive with Memoization):
    We define a recursive function minCostClimbingStairsTopDown, which takes the current index n and 
    the array cost as inputs.
    Within this function, we recursively call itself for the previous two stairs (if they exist) 
    and memoize the results to avoid redundant calculations.

    Finally, we return the minimum cost to reach the n-th stair.
    
    def minCostClimbingStairsTopDown(cost, n, memo):
        if n <= 1: # 2 BASE cases encoded here!
            return cost[n] 
        if memo[n] != -1:
            return memo[n]
        memo[n] = min(minCostClimbingStairsTopDown(cost, n-1, memo), minCostClimbingStairsTopDown(cost, n-2, memo)) + cost[n]
        return memo[n]

    def minCostClimbingStairs(cost):
        n = len(cost)
        memo = [-1] * n
        return min(minCostClimbingStairsTopDown(cost, n-1, memo), minCostClimbingStairsTopDown(cost, n-2, memo))
    

    Bottom-Up Approach (Iterative):
    We initialize a DP array dp of size n+1 to store the minimum cost to reach each stair.
    We start by assigning the costs of the first two stairs to the first two elements of the DP array.
    Then, we iterate through the remaining stairs and compute the minimum cost to reach each stair by considering the minimum of the costs of the previous two stairs plus the cost of the current stair.
    Finally, we return the minimum cost to reach either the last or second last stair, as these are the possible starting points.
    
    def minCostClimbingStairsBottomUp(cost):
        n = len(cost)
        dp = [0] * (n+1)
        dp[0], dp[1] = cost[0], cost[1]
        for i in range(2, n):
            dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
        return min(dp[n-1], dp[n-2])
    
    Example Usage:
    cost = [10, 15, 20]
    print(minCostClimbingStairsBottomUp(cost))  # Output: 15


    Similar Problems:

    Problem 70: Climbing Stairs - Given n stairs, you can climb 1 or 2 steps at a time. Determine the number of distinct ways to reach the top.
    Top-Down Approach (Recursive with Memoization):
    def climbStairsTopDown(n, memo):
        if n <= 1:
            return 1
        if memo[n] != -1:
            return memo[n]
        memo[n] = climbStairsTopDown(n-1, memo) + climbStairsTopDown(n-2, memo)
        return memo[n]

    def climbStairs(n):
        memo = [-1] * (n + 1)
        return climbStairsTopDown(n, memo)


    Bottom-Up Approach (Iterative):
    def climbStairsBottomUp(n):
        if n <= 1:
            return 1
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
        
    Problem 322: Coin Change - Given an amount n and a list of coin denominations, 
    determine the minimum number of coins needed to make up that amount.
    
    Top-Down Approach (Recursive with Memoization):
    def coinChangeTopDown(coins, amount, memo):
        if amount < 0:
            return -1
        if amount == 0:
            return 0
        if memo[amount] != float('inf'):
            return memo[amount]
        for coin in coins:
            subproblem = coinChangeTopDown(coins, amount - coin, memo)
            if subproblem >= 0:
                memo[amount] = min(memo[amount], 1 + subproblem)
        return memo[amount] if memo[amount] != float('inf') else -1

    def coinChange(coins, amount):
        memo = [float('inf')] * (amount + 1)
        return coinChangeTopDown(coins, amount, memo)

    Analysis:

    Both the Top-Down and Bottom-Up approaches have a time complexity of O(n) and a
    space complexity of O(n), where n is the number of stairs.
    These approaches efficiently compute the minimum cost to reach the top of the stairs by 
    considering the optimal substructure property of the problem.
   
   
    Merging Intervals
    Problem Statement: Given a collection of intervals, merge all overlapping intervals.

    Approach:
    The key to solving merging interval problems lies in identifying the optimal solution 
    for each interval by considering the best from the left and right sides, then merging them as necessary.

    Top-Down Approach (Recursive with Memoization):
    We define a recursive function mergeIntervalsTopDown, which takes the collection of 
    intervals intervals, the start index start, the end index end, and a memoization table memo.
    Within this function, we recursively call itself for different partitions of the intervals and
     memoize the results to avoid redundant calculations.
    Finally, we return the merged intervals for the given range.
    
    def mergeIntervalsTopDown(intervals, start, end, memo):
        if start == end:
            return intervals[start][0], intervals[start][1]
        if memo[start][end]:
            return memo[start][end]
        for i in range(start, end):
            left_start, left_end = mergeIntervalsTopDown(intervals, start, i, memo)
            right_start, right_end = mergeIntervalsTopDown(intervals, i+1, end, memo)
            if left_end >= right_start:
                memo[start][end] = left_start, right_end
                return memo[start][end]
        memo[start][end] = intervals[start][0], intervals[end][1]
        return memo[start][end]

    def merge(intervals):
        n = len(intervals)
        if n == 0:
            return []
        memo = [[None]*n for _ in range(n)]
        return mergeIntervalsTopDown(intervals, 0, n-1, memo)

    Bottom-Up Approach (Iterative):
    We start by sorting the intervals based on the start times.
    Then, we initialize an empty list merged to store the merged intervals.
    We iterate through the sorted intervals and merge overlapping intervals by comparing the start and end times.
    Finally, we return the list of merged intervals.
    
    def mergeIntervalsBottomUp(intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or interval[0] > merged[-1][1]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
    
    Example Usage:
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(merge(intervals))  # Output: [[1,6],[8,10],[15,18]]
    
    Similar Problems:

    Problem 57: Insert Interval - Given a set of non-overlapping intervals sorted by their start times, 
    insert a new interval into the intervals (merge if necessary).

    Top-Down Approach:
    def insertInterval(intervals, newInterval):
        merged = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            merged.append(intervals[i])
            i += 1
        while i < len(intervals) and intervals[i][0] <= newInterval[1]:
            newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
            i += 1
        merged.append(newInterval)
        merged.extend(intervals[i:])
        return merged
    
    Bottom-Up Approach:
    def insertInterval(intervals, newInterval):
        merged = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            merged.append(intervals[i])
            i += 1
        while i < len(intervals) and intervals[i][0] <= newInterval[1]:
            newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
            i += 1
        merged.append(newInterval)
        merged.extend(intervals[i:])
        return merged
    
    
    Problem 253: Meeting Rooms II - Given an array of meeting time intervals, determine the minimum number of conference rooms required.
    Top-Down Approach:
    import heapq

    Sort the given meetings by their start time.
    Initialize a new min-heap and add the first meeting's ending time to the heap. We simply need to keep track of the ending times as that tells us when a meeting room will get free.
    For every meeting room check if the minimum element of the heap i.e. the room at the top of the heap is free or not.
    If the room is free, then we extract the topmost element and add it back with the ending time of the current meeting we are processing.
    If not, then we allocate a new room and add it to the heap.
    After processing all the meetings, the size of the heap will tell us the number of rooms allocated. This will be the minimum number of rooms needed to accommodate all the meetings.


    def minMeetingRooms(intervals):
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        rooms = []
        heapq.heappush(rooms, intervals[0][1])
        for interval in intervals[1:]:
            if interval[0] >= rooms[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, interval[1])
        return len(rooms)
    
    Bottom-Up Approach:
    import heapq

    def minMeetingRooms(intervals):
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        rooms = []
        heapq.heappush(rooms, intervals[0][1])
        for interval in intervals[1:]:
            if interval[0] >= rooms[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, interval[1])
        return len(rooms)

    Analysis:

    The Top-Down approach has a time complexity of O(n^2) and a space complexity of O(n^2) due to the memoization table.
    The Bottom-Up approach has a time complexity of O(n log n) and a space complexity of O(n) due to the sorting of intervals.

    Both approaches efficiently merge overlapping intervals while considering the optimal substructure property of the problem.
    Dynamic Programming on Strings

    Problem Statement: Dynamic Programming on strings involves solving problems that require processing or comparing strings optimally.

    Approach:
    Most problems in this category involve building a 2D DP table to store intermediate results, where the value at dp[i][j] represents the optimal solution considering substrings s1[:i] and s2[:j].

    Example Problem: Longest Common Subsequence (Problem #1143)
    Problem Statement: Given two strings text1 and text2, return the length of their longest common subsequence.

    Approach:
    We build a 2D DP table dp where dp[i][j] represents the length of the longest common subsequence of text1[:i] and text2[:j].

    If text1[i-1] == text2[j-1], we extend the common subsequence by 1: dp[i][j] = dp[i-1][j-1] + 1.
    Otherwise, we take the maximum length of common subsequences from either deleting a character from text1 or text2: dp[i][j] = max(dp[i-1][j], dp[i][j-1]).
    Similar Problems:

    Problem 516: Longest Palindromic Subsequence - Given a string s, find the length of the longest palindromic subsequence in s.
    Problem 72: Edit Distance - Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.
    Code Implementation:

    def longestCommonSubsequence(text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    Example Usage:
    text1 = "abcde"
    text2 = "ace"

    print(longestCommonSubsequence(text1, text2))  # Output: 3 (LCS: "ace")
    
    Analysis:

    The time complexity of the DP solution is O(m * n), where m and n are the lengths of 
    text1 and text2, respectively.
    The space complexity is also O(m * n) due to the DP table.
    This approach efficiently finds the longest common subsequence between
    two strings by leveraging dynamic programming techniques.
    


    Decision Making
    Problem Statement: Decision-making problems involve determining whether to include 
    or exclude the current state to optimize a given objective.

    Approach:
    These problems usually involve constructing a DP table where dp[i][j] represents the maximum 
    value or minimum cost achievable up to the i-th state by considering j options.

    Example Problem: House Robber (Problem #198)
    Problem Statement: You are a professional robber planning to rob houses along a street. Each house 
    has a certain amount of money stashed, and the only constraint stopping you from robbing each of 
    them is that adjacent houses have security systems connected, and it will automatically contact 
    the police if two adjacent houses were broken into on the same night. Determine the maximum 
    amount of money you can rob tonight without alerting the police.

    Approach:
    We use a DP approach to calculate the maximum amount of money that can be robbed up to the 
    i-th house while considering two options: either robbing the current house or skipping it.

    If we choose to rob the i-th house, we add its value to the maximum loot from the previous non-adjacent house.
    If we choose to skip the i-th house, we take the maximum loot up to the i-1-th house.
    Similar Problems:

    Problem 213: House Robber II - Similar to Problem #198, but the houses are arranged in a circular manner.
    Problem 121: Best Time to Buy and Sell Stock - Given an array of stock prices, 
                 find the maximum profit that can be obtained by buying and selling at most once.
    
    
    House Robber II (Problem #213)
    Problem Statement: Similar to Problem #198, but the houses are arranged in a circular manner.

    Approach:
    To solve this problem, we can use dynamic programming to find the 
    maximum loot that can be obtained while considering two cases:

    Robbing the houses from the first to the second-to-last house.
    Robbing the houses from the second to the last house.
    
    We then return the maximum loot obtained from these two cases.

    Top-Down Approach:
    def rob(nums):
        if len(nums) == 1:
            return nums[0]
        return max(robHelper(nums[:-1]), robHelper(nums[1:]))

    def robHelper(nums):
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]
    
    Bottom-Up Approach:
    
    def rob(nums):
        if len(nums) == 1:
            return nums[0]
        return max(robHelper(nums[:-1]), robHelper(nums[1:]))

    def robHelper(nums):
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]
   
   
    Similar Problems:

    Problem 198: House Robber - You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed.
    Problem 337: House Robber III - The third problem in the house robber series, where houses are arranged in a binary tree.
    
    Best Time to Buy and Sell Stock (Problem #121)
    Problem Statement: Given an array of stock prices, find the maximum profit that can be obtained by buying and selling at most once.

    Approach:
    To solve this problem, we can use dynamic programming to keep track of the maximum 
    profit that can be obtained by buying and selling the stock at each day.

    Top-Down Approach:
    def maxProfit(prices):
        def helper(index, bought):
            if index >= len(prices):
                return 0
            if (index, bought) in memo:
                return memo[(index, bought)]

            if bought:
                # If currently holding a stock, we can either sell it or skip this day
                sell_today = prices[index] + helper(index + 1, False)
                skip_today = helper(index + 1, True)
                memo[(index, bought)] = max(sell_today, skip_today)
            else:
                # If not holding a stock, we can either buy a stock or skip this day
                buy_today = -prices[index] + helper(index + 1, True)
                skip_today = helper(index + 1, False)
                memo[(index, bought)] = max(buy_today, skip_today)
            
            return memo[(index, bought)]

    if not prices:
        return 0

    memo = {}
    return helper(0, False)


    Example usage:
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(prices)) # Output should be 5

    Bottom-Up Approach:
    def maxProfit(prices):
        if not prices:
            return 0
        max_profit = 0
        min_price = prices[0]
        for price in prices[1:]:
            max_profit = max(max_profit, price - min_price)
            min_price = min(min_price, price)
        return max_profit

    Similar Problems:

    Problem 122: Best Time to Buy and Sell Stock II - You may complete as many transactions as you 
    like (i.e., buy one and sell one share of the stock multiple times).

    Problem 123: Best Time to Buy and Sell Stock III - You may complete at most two transactions.
    Code Implementation:

    def rob(nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        
        return dp[-1]


    Example Usage:
    nums = [2,7,9,3,1]
    print(rob(nums))  # Output: 12 (Rob houses 1, 3, and 5 for a total of 12)
    Analysis:

    The time complexity of the DP solution is O(n), where n is the number of houses.
    The space complexity is O(n) due to the DP array.
    This approach efficiently determines the maximum loot that can be obtained without alerting 
    the police by employing dynamic programming principles.
#########################
############################################################
✅Most Frequently Asked Dynamic Programming Questions | Practice Well

    Here is the list of 10 dynamic programming questions that are commonly asked by interviewers (I made this list while preparing for interviews):

    Dice Throw Problem: Given n dice each with m faces, numbered from 1 to m, find the number of ways to get sum X. 
    X is the summation of values on each face when all the dice are thrown.

    Coin Change: You are given n types of coin denominations of values v(1) < v(2) < ... < v(n) (all integers). Assume v(1) = 1, 
    so you can always make change for any amount of money C. Give an algorithm 
    which makes change for an amount of money C with as few coins as possible.

    Counting Boolean Parenthesizations: You are given a boolean expression consisting of a string of the symbols 
    'true', 'false', 'and', 'or', and 'xor'. Count the number of ways to parenthesize the expression 
    such that it will evaluate to true. For example, there is only 1 way to parenthesize 'true and false xor true' 
    such that it evaluates to true.

    Subset Sum Problem: Given a set of non-negative integers, and a value sum, determine if 
    there is a subset of the given set with sum equal to given sum.

    Minimum Number of Jumps: Given an array of integers where each element represents the maximum number of 
    steps that can be made forward from that element, find the minimum number of jumps to reach the 
    end of the array (starting from the first element).


    Two-Person Traversal of a Sequence of Cities: You are given an ordered sequence of n cities, 
    and the distances between every pair of cities. You must partition the cities into two subsequences 
    (not necessarily contiguous) such that person A visits all cities in the first subsequence (in order), 
    person B visits all cities in the second subsequence (in order), and such that the sum of the 
    total distances travelled by A and B is minimized. Assume that person A and person B start 
    initially at the first city in their respective subsequences.

    Balanced Partition: You have a set of n integers each in the range 0 ... K. 
    Partition these integers into two subsets such that you minimize |S1 - S2|, where S1 and S2 denote the 
    sums of the elements in each of the two subsets.


    Optimal Strategy for a Game: Consider a row of n coins of values v(1) ... v(n), where n is even. 
    We play a game against an opponent by alternating turns. In each turn, a player selects either 
    the first or last coin from the row, removes it from the row permanently, and
    receives the value of the coin. Determine the maximum possible amount 
    of money we can definitely win if we move first.


    Maximum Value Contiguous Subsequence: Given a sequence of n real numbers A(1) ... A(n), 
    determine a contiguous subsequence A(i) ... A(j) for which the sum of elements in the subsequence is maximized.

    Edit Distance: Given two text strings A of length n and B of length m, you 
    want to transform A into B with a minimum number of operations of the following types: 
    delete a character from A, insert a character into A, or change some character in A into a 
    new character. The minimal number of such operations required to transform A into B 
    is called the edit distance between A and B.

    You can also include the type which requires traversing through the array and then solving 
    subproblems in left side and right side. Eg - Matrix Chain Multiplication, Egg Dropping problems etc.
    Some examples from leetcode:

    https://leetcode.com/problems/burst-balloons/
    https://leetcode.com/problems/scramble-string/
    https://leetcode.com/problems/parsing-a-boolean-expression/


    Wish u Good Luck Mates !!

###################################################################################
###################################################################################

###################################################################################
+##################################################################################
+Interesting/Google Problems Fun:

+    1) Given a zero-inexed array H of height of buildings, number of bricks b and number of ropes r. You start your journey from buiding 0 and 
+        move to adjacent building either using rope or bricks. You have limited number of bricks and ropes.
+        While moving from ith building to (i+1)th building,
+        if next building's height is less than or equal to the current buiding's height, you do not need rope or bricks.
+        if next building's height is greater than current buiding's height, you can either use one rope or (h[i+1] - h[i]) bricks.
+        So, question is How far can you reach from 0th buiding if you use bricks and ropes optimally? return index of building till which you can move.
+        Example 1:
+        Input : H = [4,2,7,6,9,11,14,12,8], b = 5, r = 2
+        Output: 8
+        Explanation: use rope to move from index 1 to index 2. 
+        use 3 bricks to move from index 3 to index 4. 
+        use 2 bricks to move from index 4 to index 5. 
+        use rope to move from index 5 to index 6. 
+        so we can reach at the end of the array using 2 ropes and 5 bricks. 
+        Example 2:
+        Input : H = [4,2,7,6,9,11,14,12,8], b = 5, r = 1
+        Output: 5
+        Explanation: use rope to move from index 1 to index 2. 
+        use 3 bricks to move from index 3 to index 4. 
+        use 2 bricks to move from index 4 to index 5. 
+        so we can reach at index 5 using 1 ropes and 5 bricks. 

        You can just be greedy tbh right? hmm... 
        use bricks for the smallest ones.. 



+        ✔️ Solution - I (Using Min-Heap)
+        Ladders can be used anywhere even for an infinite jump to next building. We need to realise that bricks limit our jumping capacity if 
+        used at wrong jumps. So, bricks should be used only on the smallest jumps in the path and ladders should be used on the larger ones.
+        We could have sorted the jumps and used ladders on largest L jumps(where, L is number of ladders) and bricks elsewhere. 
+        But, we don't know how many jumps will be performed or what's the order of jump sequence.
+        For this, we will assume that the first L jumps are the largest ones and store the jump heights in ascending order. 
+        We can use priority_queue / min-heap for this purpose (since we would be needing to insert and delete elements from it...explained further).
+        Now, for any further jumps, we need to use bricks since the first L jumps have used up all the ladders. 
+        Let's call the jump height requried from now on as curJumpHeight. Now,
+        If curJumpHeight > min-heap top : We have the choice to use bricks on the previous jump which had less jump height. 
+        So, we will use that many bricks on previous (smaller) jump and use ladder for current (larger) jump.
+        If curJumpHeight <= min-heap top : There's no way to minimize usage of bricks for current jump. 
+        We need to spend atleast curJumpHeight number of bricks
+        So, using the above strategy, we can find the furthest building we can reach. As soon as the available 
+        bricks are all used up, we return the current building index.


+        Time Complexity : O(NlogL)
+        Space Complexity : O(L)
+        def furthestBuilding(self, H: List[int], bricks: int, ladders: int) -> int:
+            jumps_pq = []
+            for i in range(len(H) - 1):
+                jump_height = H[i + 1] - H[i]
+                if jump_height <= 0: continue
+                heappush(jumps_pq, jump_height)
+                if len(jumps_pq) > ladders:
+                    bricks -= heappop(jumps_pq)
+                if(bricks < 0) : return i
+            return len(H) - 1

+        Dumber soln:
+        Time Complexity : O(NlogL)
+        Space Complexity : O(L)
+        from sortedcontainers import SortedDict
+        def furthestBuilding(self, H: List[int], bricks: int, ladders: int) -> int:
+            jumps = SortedDict()
+            for i in range(len(H) - 1):
+                jump_height = H[i + 1] - H[i]
+                if jump_height <= 0: continue
+                jumps[jump_height], ladders = jumps.get(jump_height, 0) + 1, ladders - 1
+                if ladders < 0:
+                    top = jumps.peekitem(0)[0]
+                    bricks -= top
+                    jumps[top] -= 1
+                    if jumps[top] == 0 : jumps.popitem(0)
+                if(bricks < 0) : return i
+            return len(H) - 1


+    2) Topological ordering
+        Given task dependencies (x, y) denoting that to complete y one must first complete x. Find the least
+        amount of time needed to finish all the tasks given that the task that can be done in parallel can be 
+        processed together and it takes 1 unit of time to process a task. 
+        Inorder -> reduce by 1? Process TOPOLOGICAL ORDER BFS'ing?
+        figure it out!
+        (https://docs.google.com/document/d/1S4osCeZjZa20sWywMq4EneHJALcUfPsPm5fXwLbwiAM/edit#heading=h.p5po8tu4jedu)
+    3) Write something about huffman codes here:
+        https://courses.csail.mit.edu/6.897/spring03/scribe_notes/L13/lecture13.pdf
+        and arithmetic codes, etc
+        Given a list of character and the frequencies they appear. Construct a Huffman Tree to encode all character.
+        For more details of building the Huffman Tree and how to use it to encode and decode, please take a look at this article.
+        In terms of writing the code to build the Huffman Tree, one solution I have is use a min heap to store Node objects, each Node 
+        object contains a character and its frequency, then every time poll the two lowest frequency Node objects from the min heap and 
+        create the sum Node, offer the sum Node back to the min heap, repeat this process till we only have one Node in the heap, and that 
+        will be the root node of the Huffman Tree. Eventually all the character nodes will become the leaves.
+        Encoding and decoding are pretty straightforward, just like how we traverse a tree to find a leaf with certain value, the only thing
+        is we use 0 and 1 to indicate whether we are traversing left or right direction.
+    4) Convert  some problems to multiplication and turn them into logs for djikstras such as 2sigmas currency exchange?


+    5) LeetCode 711 - Number of Distinct Islands II
+    https://protegejj.gitbooks.io/algorithm-practice/content/711-number-of-distinct-islands-ii.html
+    Given a non-empty 2D arraygridof 0's and 1's, an island is a group of1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
+
+    Count the number of distinct islands. An island is considered to be the same as another if they have the same shape, or have the same shape after 
+    rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).
+
+
+    -> dfs to find all islands
+    -> ok now we need to encode the island!
+
+    
+    -> Find top left coordinate of shape, then subtract all coordiantes by top left coordinate to get it in a picture frame
+    -> width and height of picture frame??
+        -> keep track of top most coordinate, leftmost, rightmost, bottom most. 
+        -> leftmost - rightmost + 1= width
+        -> bttom - top + 1 = height
+         012 
+         111 0
+           1 1
+           1 2 
+
+        3 by 3!
+
+
+    -> get its width and height, and put it in an array
+    
+    -> then rotate array and insert into map, and check
+    -> (Memorize rotation code again)
    -> TO ROTATE BY 270 DEGREES DO TRANSPOSE OF MATRIX ADN THEN COLUMNS REVERSED FOR THE TRANSPOSE MATRIX 

    -> TO DO 90 DEGREE ROTATION DO TRANSPOSE OF MATRIX? DO TRANSPOSE AND ROW REVERSE OF THE TRANSPOSE RIGHT? MAYUBE... 
    
    
    function transpose(matrix) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = i; j < matrix[0].length; j++) {
            const temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
            }
        }
        return matrix;
    }


+SPIRAL MATRIX GOOD SOLN:
+    Let us use coordinate (x, y) and direction of movement (dx, dy). Each time when we reach point outside 
+    matrix we rotate. How we can rotate? We can either create array of rotations in advance or we can use the trick 
+    dx, dy = -dy, dx. Also how we understand it is time to rotate? We will write already visited elements with *, so 
+    when we see * or we go outside the grid, it is time to rotate.
+    Complexity
+    It is O(mn) both for time and space.
+    class Solution:
+        def spiralOrder(self, matrix):
+            n, m = len(matrix[0]), len(matrix)
+            x, y, dx, dy = 0, 0, 1, 0
+            ans = []
+            for _ in range(m*n):
+                if not 0 <= x+dx < n or not 0 <= y+dy < m or matrix[y+dy][x+dx] == "*":
+                    dx, dy = -dy, dx
+                    
+                ans.append(matrix[y][x])
+                matrix[y][x] = "*"
+                x, y = x + dx, y + dy
+            
+            return ans


+MEMORY ALLOCATOR
+    I've been recently asked this question in the onsite interview:
+    You are required to design a memory allocation class for an bigArray, which will have below methods:
+
+    int bigArray[10000] -> Not necessarily int[], can be anything(Should not modify array)
+
+    int allocateSlice(int size){
+    // allocate a contiguous slice of length size in the array and return the starting index from where data can be stored
+    }
+
+    int freeSlice(int startIndex){
+    // de-allocate a contiguous slice starting at startIndex, with size it has been allocated earlier
+    }
+
+    I was able to come up with an approach but don't think it was scaleable and would pass all test cases!
+    Can you guys tell me your thoughts on possible solutions?
+
+    #Google #Design
+
+        Maintain a set of 'free' ranges in a set sorted by their length. The set will initially have only one value - [0, MAXSIZE - 1]. When you have to
+        allocate for size n, binary search for first range, then split it and store this operation in a hashmap. To free a slice, get the size from the hashmap 
+        and insert that range back into the set. Both operations take O(logn).
+
+        Write so we use a treemap to contain sizes pointing to ranges
+        
+        when we free, we will have to find if that range toouches another range, merge it, and then put it into tree map right?
+
+
+    Top k most frequenet element with TREE MAP/BST
+
+        The total time complexity is O(n + klogn)
+
+        class Solution {
+        public:
+            vector<int> topKFrequent(vector<int>& nums, int k) {
+                unordered_map<int, int> valToCnt;
+                for (auto it: nums) 
+                    valToCnt[it]++;
+                
+                map<int, vector<int>> cntToVal;
+                for (auto it: valToCnt) 
+                    cntToVal[-it.second].push_back(it.first);
+                
+                vector<int> ans; 
+                for (auto it: cntToVal) {
+                    for (auto itB: it.second) {
+                        ans.push_back(itB);
+                        k--;
+                        if (!k) return ans;
+                    }
+                }
+                return ans;
+            }
+        };
+       
        SO we have ranges in treemap?
        Allocate size 5 
                -> find a size bigger than five and take it from there. 
                -> we store big size 100000 in treemap -> points to full range 

                -> then allocate -> store   200 in treempa -> point to starte of range 0 - 200 and rest 201 - 100000 is also in treemap that points to a bigger size in hashmap. 

                -> keep doing this.
                -> when you free.. -> you ahave to find the start index -> we can store these start indexes in hashmap -> then point to the treemap... 
                -> BECAUSE YOU AHVE the size with the start index index treemap to get all ranges allocated for that size, then find the range with that start index. 
                -> then remove the range. -> then you have to do merge intervals right! -> so you have to see if start index + n + 1 was allocated or somethignr ight? 
                -> same to check if start index -1 is allocated.. 
                -> hmmm or ... 
                -> wait this soln is kinda wrong..



+
+
+
+
+    Snapchat Phone Screen
+
+        Question - Design a class/method AddAndGetTopK to add a number and get top K frequent elements.
+        Eg - k = 3
+        4
+        4,5
+        4,5,4 => 4,5
+        4,5,4,6 => 4,5,6
+        4,5,4,6,5 => 4,5,6
+        4,5,4,6,5,7 => 4,5,6
+        4,5,4,6,5,7,7 => 4,5,7
+
+        I suggessted a BST solution with O(logn) insertion time and O(logn) time for getting top K elements, by maintaining the count of nodes below every node.
+
+        Any other solutions?
+
+
+        freq -> val BST. 
+        
+        OK add a number
+        val -> freq hashmap
+        then also update in freq->val map 
+
+        Ok but how do we do top k?
+
+        can also use pq, and get rid of dirty elements. 
+
+
+    ENUMERATING ALL TOPOLOGICAL SORTS (DATAVISOR)
+        Enumerate all possible topological sorts,
+        dont just give me one topological sort!!
+
+    Word finder:
+
+        We are given a list of words that have both 'simple' and 'compound' words in them. Write an algorithm that 
+        prints out a list of words without the compound words that are made up of the simple words.
+
+        Input: chat, ever, snapchat, snap, salesperson, per, person, sales, son, whatsoever, what so.
+        Output should be: chat, ever, snap, per, sales, son, what, so
+ 
+        The idea . First we have to sort words by length. Then generate trie and put the words in it. Check in the trie if some word is compound.
+       -> actulally wouldnt we also need the prefixes here -> like for snapchat -> we should be searching the prefix chat.. 
        -> waiot if you store smaller words in trie first. 
        -> then search snapchat.... then you already know snap and chat in trie, so once you find snap, make sure chat exists and if both exist and string is complete, add both as 
        simple words!!
        

+
+    Given an R x C grid of letters "P", "F", and ".", you have one person "P" and his friends "F".
+    The person will go visit all his friends and can walk over the empty spaces to visit his friends.
+    He visits a friend when he walks onto the friend's square.
+    He can walk over his friends.
+    Find the length of the shortest path for him to visit all his friends.
+
+    "..P..",
+    "F...F",
+    "FF.FF"
+    The answer is 9. Can someone please help solving this question?
+
+        Answer:
+        Create MST using the friends node. (Kruskal Algo) -
+        Find shortest distance to any of the friends starting from P. (BFS )
+        Ans = Cost of MST (part1) + shortest distance(part-2)
+
+    Word Search 2 Snap:
+
+        class TrieNode():
+            def __init__(self):
+                self.children = collections.defaultdict(TrieNode)
+                self.isWord = False
+        class Trie():
+            def __init__(self):
+                self.root = TrieNode()
+            
+            def insert(self, word):
+                node = self.root
+                for w in word:
+                    node = node.children[w]
+                node.isWord = True
+            
+            def search(self, word):
+                node = self.root
+                for w in word:
+                    node = node.children.get(w)
+                    if not node:
+                        return False
+                return node.isWord
+            
+        class Solution(object):
+            def findWords(self, board, words):
+                res = []
+                trie = Trie()
+                node = trie.root
+                for w in words:
+                    trie.insert(w)
+                for i in xrange(len(board)):
+                    for j in xrange(len(board[0])):
+                        self.dfs(board, node, i, j, "", res)
+                return res
+            
+            def dfs(self, board, node, i, j, path, res):
+                if node.isWord:
+                    res.append(path)
+                    node.isWord = False
+                if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
+                    return 
+                tmp = board[i][j]
+                node = node.children.get(tmp)
+                if not node:
+                    return 
+                board[i][j] = "#"
+                self.dfs(board, node, i+1, j, path+tmp, res)
+                self.dfs(board, node, i-1, j, path+tmp, res)
+                self.dfs(board, node, i, j-1, path+tmp, res)
+                self.dfs(board, node, i, j+1, path+tmp, res)
+                board[i][j] = tmp
+
+
+    Coding questions:
+        Problem 1
+
+        Custom:
+            Given an R x C grid of letters "P", "F", and ".", you have one person "P" and his friends "F".
+            The person will go visit all his friends and can walk over the empty spaces to visit his friends. 
+            He visits a friend when he walks onto the friend's square.
+            He can walk over his friends.
+            Find the length of the shortest path for him to visit all his friends.
+
+            "..P..",
+            "F...F",
+            "FF.FF"
+            
+            
+            The answer is 9.
+
+            This is a graph problem. You need to compute the distance between each person or friend, 
+            then find the size of the minimum spanning tree. Use a union find.
+
+            Worth coding this??
+
+
+    Subclassing + LRUCache but  each key has a cost instead of 1 for capiacty:
+        from collections import OrderedDict
+
+        class SCLRUCache(OrderedDict):
+            def __init__(self, max_cost: int):
+                self.curr_cost = 0
+                self.max_cost = max_cost
+            
+            def add(self, key, value, cost: int):
+                self[key] = (value, cost)
+                self.curr_cost += cost
+                while self.curr_cost > self.max_cost:
+                    _, removed_cost = self.popitem(last = False)
+                    self.curr_cost -= removed_cost
+                    
+            def read(self, key):
+                if key not in self:
+                    return - 1
+
+                self.move_to_end(key)
+                value, _ = self[key]
+                return value
+
+    241. Different Ways to Add Parentheses
+
+        Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.
+
+        
+
+        Example 1:
+
+        Input: expression = "2-1-1"
+        Output: [0,2]
+        Explanation:
+        ((2-1)-1) = 0 
+        (2-(1-1)) = 2
+        Example 2:
+
+        Input: expression = "2*3-4*5"
+        Output: [-34,-14,-10,-10,10]
+        Explanation:
+        (2*(3-(4*5))) = -34 
+        ((2*3)-(4*5)) = -14 
+        ((2*(3-4))*5) = -10 
+        (2*((3-4)*5)) = -10 
+        (((2*3)-4)*5) = 10
+        
+
+        Constraints:
+
+        1 <= expression.length <= 20
+        expression consists of digits and the operator '+', '-', and '*'.
+        All the integer values in the input expression are in the range [0, 99].
+
+
+            class Solution:
+                def diffWaysToCompute(self, expression: str) -> List[int]:
+                    
+                    if ('+' not in expression) and ('-' not in expression) and ('*' not in expression):
+                        return [int(expression)]
+                    
+                    res = []
+                    
+                    for i, v in enumerate(expression):
+                        if v == '+' or v == '-' or v == '*':
+                            left_res = self.diffWaysToCompute(expression[:i])
+                            right_res = self.diffWaysToCompute(expression[i + 1:])
+                            for left_i, left_v in enumerate(left_res):
+                                for right_i, right_v in enumerate(right_res):
+                                    if v == '+':
+                                        res.append(left_v + right_v)
+                                    elif v == '-':
+                                        res.append(left_v - right_v)
+                                    else:
+                                        res.append(left_v * right_v)
+                    return res
+
+            MEMO SOLN
+
+            class Solution:
+                def diffWaysToCompute(self, expression: str) -> List[int]:
+                    self.memo = defaultdict(list)
+                    return self.diffWaysToComputeUtil(expression)
+                    
+                    
+                def diffWaysToComputeUtil(self, expression: str) -> List[int]:
+                    if expression in self.memo:
+                        return self.memo[expression]
+                    
+                    if ('+' not in expression) and ('-' not in expression) and ('*' not in expression):
+                        return [int(expression)]
+                    
+                    res = []
+                    
+                    for i, v in enumerate(expression):
+                        if v == '+' or v == '-' or v == '*':
+                            left_res = self.diffWaysToComputeUtil(expression[:i])
+                            right_res = self.diffWaysToComputeUtil(expression[i + 1:])
+                            for left_i, left_v in enumerate(left_res):
+                                for right_i, right_v in enumerate(right_res):
+                                    if v == '+':
+                                        res.append(left_v + right_v)
+                                    elif v == '-':
+                                        res.append(left_v - right_v)
+                                    else:
+                                        res.append(left_v * right_v)
+                    self.memo[expression] = res
+                    return res
+
+
+
+
+    Quick Select impl: Get Top K most frequent elemetns:
+
+        from collections import Counter
+        class Solution:
+            def topKFrequent(self, nums: List[int], k: int) -> List[int]:
+                count = Counter(nums)
+                unique = list(count.keys())
+                def partition(left, right, pivot_index) -> int:
+                    pivot_frequency = count[unique[pivot_index]]
+                    # 1. move pivot to end
+                    unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
+                    
+                    # 2. move all less frequent elements to the left
+                    store_index = left
+                    for i in range(left, right):
+                        if count[unique[i]] < pivot_frequency:
+                            unique[store_index], unique[i] = unique[i], unique[store_index]
+                            store_index += 1
+
+                    # 3. move pivot to its final place
+                    unique[right], unique[store_index] = unique[store_index], unique[right]  
+                    
+                    return store_index
+                
+                def quickselect(left, right, k_smallest) -> None:
+                    """
+                    Sort a list within left..right till kth less frequent element
+                    takes its place. 
+                    """
+                    # base case: the list contains only one element
+                    if left == right: 
+                        return
+                    
+                    # select a random pivot_index
+                    pivot_index = random.randint(left, right)     
+                                    
+                    # find the pivot position in a sorted list   
+                    pivot_index = partition(left, right, pivot_index)
+                    
+                    # if the pivot is in its final sorted position
+                    if k_smallest == pivot_index:
+                        return 
+                    # go left
+                    elif k_smallest < pivot_index:
+                        quickselect(left, pivot_index - 1, k_smallest)
+                    # go right
+                    else:
+                        quickselect(pivot_index + 1, right, k_smallest)
+                
+                n = len(unique) 
+                # kth top frequent element is (n - k)th less frequent.
+                # Do a partial sort: from less frequent to the most frequent, till
+                # (n - k)th less frequent element takes its place (n - k) in a sorted array. 
+                # All element on the left are less frequent.
+                # All the elements on the right are more frequent.  
+                quickselect(0, n - 1, n - k)
+                # Return top k frequent elements
+                return unique[n - k:]
+
+    Circular sort:
+
+        Given an array of integers, return true or false if the numbers in the array go from 0... (N - 1) 
+        where N is the length of the array
+
+        Linear time, constant space is a requirement
+
+        example:
+        [0,1,2,3,4] = true;
+        [4,2,1,0,3] = true;
+        [0,1,5,2,4] = false;
+
+
+
+    Valid Number DFA SOLN:
+        class Solution(object):
+        def isNumber(self, s):
+            """
+            :type s: str
+            :rtype: bool
+            """
+            #define a DFA
+            state = [{}, 
+                    {'blank': 1, 'sign': 2, 'digit':3, '.':4}, 
+                    {'digit':3, '.':4},
+                    {'digit':3, '.':5, 'e':6, 'blank':9},
+                    {'digit':5},
+                    {'digit':5, 'e':6, 'blank':9},
+                    {'sign':7, 'digit':8},
+                    {'digit':8},
+                    {'digit':8, 'blank':9},
+                    {'blank':9}]
+            currentState = 1
+            for c in s:
+                if c >= '0' and c <= '9':
+                    c = 'digit'
+                if c == ' ':
+                    c = 'blank'
+                if c in ['+', '-']:
+                    c = 'sign'
+                if c not in state[currentState].keys():
+                    return False
+                currentState = state[currentState][c]
+            if currentState not in [3,5,8,9]:
+    Word Search 2: Find all words on the board:
+        https://leetcode.com/problems/word-search-ii/discuss/59790/Python-dfs-solution-(directly-use-Trie-implemented).
+        @caikehe Great solution, but no need to implement Trie.search() since the search is essentially done by dfs.
+
+        class TrieNode():
+            def __init__(self):
+                self.children = collections.defaultdict(TrieNode)
+                self.isWord = False
+        class Trie():
+            def __init__(self):
+                self.root = TrieNode()
+            def insert(self, word):
+                node = self.root
+                for w in word:
+                    node = node.children[w]
+                node.isWord = True
+            def search(self, word):
+                node = self.root
+                for w in word:
+                    node = node.children.get(w)
+                    if not node:
+                        return False
+                return node.isWord
+        class Solution(object):
+            def findWords(self, board, words):
+                res = []
+                trie = Trie()
+                node = trie.root
+                for w in words:
+                    trie.insert(w)
+                for i in xrange(len(board)):
+                    for j in xrange(len(board[0])):
+                        self.dfs(board, node, i, j, "", res)
+                return res
+            
+            def dfs(self, board, node, i, j, path, res):
+                if node.isWord:
+                    res.append(path)
+                    node.isWord = False
+                if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
+                    return 
+                tmp = board[i][j]
+                node = node.children.get(tmp)
+                if not node:
+                    return 
+                board[i][j] = "#"
+                self.dfs(board, node, i+1, j, path+tmp, res)
+                self.dfs(board, node, i-1, j, path+tmp, res)
+                self.dfs(board, node, i, j-1, path+tmp, res)
+                self.dfs(board, node, i, j+1, path+tmp, res)
+                board[i][j] = tmp


+    Similar to HRT problem:
+        There are some processes that need to be executed. Amount of a load that process causes on a server that runs it, is being represented by a single integer. Total load caused on a server is the sum of the loads of all the processes that run on that server. You have at your disposal two servers, on which mentioned processes can be run. Your goal is to distribute given processes between those two servers in the way that, absolute difference of their loads will be minimized.
+        Given an array of n integers, of which represents loads caused by successive processes, return the minimum absolute difference of server loads.
+        Example 1:
+        Input: [1, 2, 3, 4, 5]
+        Output: 1
+        Explanation:
+        We can distribute the processes with loads [1, 2, 4] to the first server and [3, 5] to the second one,
+        so that their total loads will be 7 and 8, respectively, and the difference of their loads will be equal to 1.
+        Just do the HRT problem instead!
+        Dishes, friends,  ETC, minimize the unfairness!
+    K stack pops
+    Divide subset into equal partitions Subset sum into k partitions (linkedin): 
+
+

+#################################################################################3
+#######################################################################################
+Cool Trie Code:
+    Given a dictionary of words (sorted lexicographically) and a prefix string, return all the words that start with the given prefix. @BigV
+    Couldnt you binary search the letters in the word?
+    But i guess trie is faster regardless. 
+    Rabinkarp hash the word. 
+    hash the first 8 letters of every word and put in a map. get idx, then go left and right in array!
+    from collections import defaultdict
+    class TrieNode: 
+        def __init__(self): 
+            self.ht = defaultdict(TrieNode) 
+            self.isEnd = False 
+    class Trie: 
+        def __init__(self): 
+            self.root = TrieNode() 
+        
+        def add_word(self, word): 
+            curr = self.root
+            for char in word: 
+                curr = curr.ht[char]
+            curr.isEnd = True 
+        
+        def startsWith(self, prefix, trieNode): 
+            rtn = [] 
+            
+            def dfs(currWord, currTrieNode): 
+                if currTrieNode.isEnd: 
+                    rtn.append(currWord)
+                    
+                for char, children in currTrieNode.ht.items(): 
+                    dfs(currWord+char, children) 
+                
+            dfs(prefix, trieNode)
+            return rtn 
+        
+    def get_prefix(prefix): 
+        example = ['a','abc','abs', 'b','bob']
+        trie = Trie()
+        
+        for word in example: 
+            trie.add_word(word)
+            
+        curr_root = trie.root
+        for char in prefix: 
+            curr_root = curr_root.ht[char]
+        
+        rtn = []
+        rtn.append(trie.startsWith(prefix, curr_root))
+        return rtn 
+        
+    print(get_prefix('ab'))


+#######################################
+##########################################
+RANDOM QUESTIONS & RESERVOIR SAMPLING:
+    Swap To Back Of Array O(1) Trick Random Questions
+    Q. Given an array, you have to write a function to select a random number from 
        the array such that it has not appeared in the last K calls to this function.
+        Approach : Used a combination of deque and unordered_set to generate the random number in O(N) time. Further, the 
+        interviewer required an optimisation of O(1), used a vector to do so. The vector contains the unselected elements so far. 
+        As soon as an element is visited, swap the element with the last element in the vector & do a pop_back operation to get the answer in O(1).
+
+
+        Write a RandomGenerator class:
+
+        public class RandomGenerator {
+
+            public RandomGenerator(int n) {
+            }
+            
+            public int generate() {
+                // todo
+            }
+        }
+        The contructor will be passed an integer n. generate is supposed to return a random number between 0 to n, but it 
+        is not supposed to return a number that it has already returned. If possiblities are exhauted, return -1.
+
+    Construct your own RNG:
+        You are in charge of writing a software for a slot machine. On press of a button, the slot machine should output the roll of 2 dice.
+        constraints: Do not use the random library. Probability of the dice rolls should be equal.
+
+        Take the unix timestamp in long form. Mod that number by 36. The remainder can range from 0 to 35. You can map the remainder to 
+        36 possibilities that the 2 6-side dice throws can have in terms of value pairs.
+
+        Another way to enhance:
+        Choose a few prime numbers and put them in an array of length N. Take system time down the smallest unit possible. 
+        Take the Mod of the floor of that number divided by previous selected prime. Use the result to select a new prime. 
+        Multiply system time the prime, take Mod 36 as suggested by @vishnushiva
+
+        You can also create your own Linear congruential generator
+            var seed = 0
+            func random() -> Int {
+                seed = (seed * 1103515245 + 12345) % Int(Int32.max)
+                return seed
+            }
+    Google - Generate random (x, y) within an area
+        Given a rectangular on a 2D coordinate, write a function to return (x, y) pair so that the distribution of the return value is uniform within the rectangular.
+        Followup 1: What if you are given multiple non-overlapping rectangular areas? How to generate uniformly distributed (x, y) pair?
+        Followup 2: What if you are given an arbitrary shaped area?
+        In general, to uniformly randomly select a (x,y) from inside a rectangle, you can independently sample the x coordinate and the y coordinate. 
+        Note that the rectangle might not be aligned along the axes (i.e. could be at an angle), but one can always find linear transformations to 
+        transform the rectangle so that its axes are parallel to the X and Y axes.
+
+        First part: Choose random number between 0 and x. and Random number between 0 and y. Then return (x,y)
+            random.uniform(a, b) -> 
+            Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.
+            The end-point value b may or may not be included in the range depending on floating-point 
+            rounding in the equation a + (b-a) * random()
+
+        Follow up 1: Have weigth of every rectangle. Choose a rectangle, based on weight of rectangle. Then in each rectange do above thing. 
+
+        Arbitrary shape -> grab a very THIN rectangle. 
+        grab height of shape.
+        Then find a slice based on height [0 and height]
+
+        Then for arbitrary shape, get all widths within the height. 
+        -> widths will be intervals!
+
+        Choose an interval based on lengths of interval as the weight.
+        Once in the interval, just do uniform [startOfInterva, endOfInterval] rng.  
+    Create unbiased random from biased random:
+        Suppose that you want to output 0 with probability 1/2 and 1 with probability 1/2. 
+        You have a function BIASED-RANDOM that outputs 1 with probability p and 0 with probability 1-p 
+        where 0<p<1.Write a function UNBIASED-RANDOM that uses BIASED -RANDOM and outputs 1 and 0 with equal probability.
+        Ok so roll twice, 
+        Consider one side heads 0
+        consider on side tails 1 
+        Roll until heads, then roll until tails. 
+        hmm
+        Heads Prob -> (1-p)
+        Tails prob -> p
+        (1-p) * p = p * (1-p)
+        
+        HH -> H
+        HT -> T
+        TH -> H
+        TT -> T
+        
+        ok roll twice. 
+        if you get 
+        HH, or TH -> heads
+        if you get 
+        TH or TT -> tails. 
+        (1-p)(1-p) = 1-2p + p^2  [2 heads]
+        
+        p*p = p^2  [2 tails]
+        (1-p)*p = [HT]
+        p * (1-p) = [TH]
+        so if you get different results you can use that to create fair coin
+        TH -> H, HT -> T
+        other options you have to re-roll!!
+        
+    Write a function to generate random numbers between given range (5 to 55 inclusive) using a given function "rand_0()" 
        which returns whether '0' or '1'.
+        Generate 5?
+        Generate 11? 
+        multiply...
+        hmm
+        how bout we subtract 5 from each now its 
+        0 to 50 -> hb now..
+        generate 16 * 4 = 64 -> 
+        0 - 3 
+        0 - 15
+        ...
+    Reservoir sampling:
+        Main idea:
+            we have n items we need to pick random!
+            select the first item, 
+            ok now select the second item 1/2 prob
+            ok now take third item with 1/3 prob
+            ... 
+            select nth item with 1/n prob
+            Can iterate through array and then tell me what item you got!
+            3 items:
+            prob you kept first item is
+            1 * (1/2) * (2/3) = 1/3!!
+            prob kept second item:
+            1 * 1/2 * 2/3 = 1/3!!
+            prb kept 3rd item 
+            is  whatever you have *  1/3
+        Facebook | Onsite | Generate random max index
+        Given an array of integers arr, randomly return an index of the maximum value seen by far.
+
+        Example:
+        Input: [11, 30, 2, 30, 30, 30, 6, 2, 62, 62]
+
+        Having iterated up to the at element index 5 (where the last 30 is), randomly give an index among [1, 3, 4, 5] which are indices of 30 - the max value by far. Each index should have a ¼ chance to get picked.
+
+        Having iterated through the entire array, randomly give an index between 8 and 9 which are indices of the max value 62.
+
+        ok 
+        import random 
+        max = float("-inf")
+        maxIdx = -1
+        count = 0
+        for idx, i in enumerate(arr):
+            if (i > max):
+                max = i
+                maxIdx = idx
+                count = 1
+            elif(i == max):
+                count += 1
+                if(random.randint(1, count) == 1):
+                    maxIdx = idx
+        return maxIdx
+ 
+    WEIGHTED reservoir sampling: 
+    
+        528. Random Pick with Weight
+        You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.
+        You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] 
+        (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).
+        For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the 
+        probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).
+        
+        Ok here is the weighted reservori sampling soln:
+            The algorithm by Pavlos Efraimidis and Paul Spirakis solves exactly this problem. The original paper with complete proofs 
+            is published with the title "Weighted random sampling with a reservoir" in Information Processing Letters 2006, 
+            but you can find a simple summary here.
+            The algorithm works as follows. First observe that another way to solve the unweighted reservoir sampling is to assign to 
+            each element a random id R between 0 and 1 and incrementally (say with a heap) keep track of the top k ids. Now let's look at 
+            weighted version, and let's say the i-th element has weight w_i. Then, we modify the algorithm by choosing the id of the i-th 
+            element to be R^(1/w_i) where R is again uniformly distributed in (0,1).
+            Another article talking about this algorithm is this one by the Cloudera folks.
+            in other words ->
+                rng(0,1) = 0.5 ^ 1/2 = 0.7071?


+    WEIGHTED RANDOM PICK INDEX with cumulative sums and bin search:
+        import random 
+        class Solution:
+            def __init__(self, w: List[int]):
+                
+                self.cum_sum = []
+                self.s = 0
+                for i in w:
+                    self.s += i
+                    self.cum_sum.append(self.s)
+                
+            def pickIndex(self) -> int:
+                # pick index with between 0 and total sum-1
+                '''
+                [1,2,3,4]
+                
+                so we have
+                [1, 3, 6, 10]
+                pick # between 1 and 10
+                
+                1 -> first idx
+                2 -> second
+                3 -> second idx
+                4 -> third idx
+                aka binary search for right idx which is always equal to or bigger than you!
+                never take smaller one
+                '''    
+                
+                r = random.randint(1, self.s)
+                start = 0
+                end = len(self.cum_sum) - 1
+                
+                while start < end:
+                    mid = start + (end - start)//2
+                    # print("start, mid, end, val", start, mid, end, self.cum_sum[mid])         
+                    if self.cum_sum[mid] == r:
+                        # return that idx!
+                        return  mid
+                    elif r < self.cum_sum[mid]:
+                        end = mid
+                    else:
+                        start = mid+1 # end might be the element we need!
+                
+                # also just do return bisect.bisect_left(self.cum_sum, r)
+                return start            
+                
+    WEIGHTED RANDOM PICK INDEX WITH ALIAS METHOD
+        O(1) INTELLIGENT ALIAS METHOD: (TODO WRITE NOTES ON THIS.)
+        https://leetcode.com/problems/random-pick-with-weight/discuss/671439/Python-Smart-O(1)-solution-with-detailed-explanation
+        Probably you already aware of other solutions, which use linear search with O(n) complexity and binary search with O(log n) complexity. 
+        When I first time solved this problem, I thought, that O(log n) is the best complexity you can achieve, howerer it is not! You can 
+        achieve O(1) complexity of function pickIndex(), using smart mathematical trick: let me explain it on the example: 
+        w = [w1, w2, w3, w4] = [0.1, 0.2, 0.3, 0.4]. Let us create 4 boxes with size 1/n = 0.25 and distribute original weights into 
+        our boxes in such case, that there is no more than 2 parts in each box. For example we can distribute it like this:
+        Box 1: 0.1 of w1 and 0.15 of w3
+        Box 2: 0.2 of w2 and 0.05 of w3
+        Box 3: 0.1 of w3 and 0.15 of w4
+        Box 4: 0.25 of w4
+        (if weights sum of weights is not equal to one, we normalize them first, dividing by sum of all weights).
+        this method has a name: https://en.wikipedia.org/wiki/Alias_method , here you can see it in more details.
+        Sketch of proof
+        There is always a way to distribute weights like this, it can be proved by induction, there is always be one box with weight <=1/n and 
+        one with >=1/n, we take first box in full and add the rest weight from the second, so they fill the full box. Like we did for Box 1 in our 
+        example: we take 0.1 - full w1 and 0.15 from w3. After we did it we have w2: 0.2, w3: 0.15 and w4: 0.4, and again we have one box with >=1/4 and one box with <=1/4.
+        Now, when we created all boxes, to generate our data we need to do 2 steps: first, to generate box number in O(1), because sizes of boxes are 
+        equal, and second, generate point uniformly inside this box to choose index. This is working, because of Law of total probability.
+        Complexity. Time and space complexity of preprocessing is O(n), but we do it only once. Time and space for function pickIndex is just O(1): 
+        all we need to do is generate uniformly distributed random variable twice!
+        Code is not the easiest one to follow, but so is the solution. First, I keep two dictionaries Dic_More and Dic_Less, where I distribute 
+        weights if they are more or less than 1/n. Then I Iterate over these dictionaries and choose one weight which is more than 1/n, another 
+        which is less, and update our weights. Finally when Dic_Less is empty, it means that 
+        we have only elements equal to 1/n and we put them all into separate boxes.
+        I keep boxes in the following way: self.Boxes is a list of tuples, with 3 numbers: index of first weight, index of second weight and split, 
+        for example for Box 1: 0.1 of w1 and 0.15 of w3, we keep (1, 3, 0.4). If we have only one weight in box, we keep its index.
+            class Solution:
+            def __init__(self, w):
+                ep = 10e-5
+                self.N, summ = len(w), sum(w)
+                weights = [elem/summ for elem in w]
+                Dic_More, Dic_Less, self.Boxes = {}, {}, []
+                
+                for i in range(self.N):
+                    if weights[i] >= 1/self.N:
+                        Dic_More[i] = weights[i]
+                    else:
+                        Dic_Less[i] = weights[i]
+
+                while Dic_More and Dic_Less:
+                    t_1 = next(iter(Dic_More))
+                    t_2 = next(iter(Dic_Less))
+                    self.Boxes.append([t_2,t_1,Dic_Less[t_2]*self.N])
+
+                    Dic_More[t_1] -= (1/self.N - Dic_Less[t_2])
+                    if Dic_More[t_1] < 1/self.N - ep:
+                        Dic_Less[t_1] = Dic_More[t_1]
+                        Dic_More.pop(t_1)
+                    Dic_Less.pop(t_2)
+                
+                for key in Dic_More: self.Boxes.append([key])
+            def pickIndex(self):
+                r = random.uniform(0, 1)
+                Box_num = int(r*self.N)
+                if len(self.Boxes[Box_num]) == 1:
+                    return self.Boxes[Box_num][0]
+                else:
+                    q = random.uniform(0, 1)
+                    if q < self.Boxes[Box_num][2]:
+                        return self.Boxes[Box_num][0]
+                    else:
+                        return self.Boxes[Box_num][1]
+    Generate uniform random integer
+    Problem: given function of rand3() which return uniformly random int number of [1,2,3], 
+    write a random function rand4(), which return uniformly random integer of [1,2,3,4]
+    How to test it?
+    As follow up, I was asked about how to test rand4() function, to verify it's truly random.
+    My thought is to run rand4() for 1K times, and collect the frequency of [1,2,3,4], and then run 
+    rand4() 10K times and collect frequency, then run rand4() 100K time ... 
+    to see if the frequency of each number between [1,2,3,4] converge to 25%.
+    There is a scenario like 1,2,3,4,1,2,3,4,1,2,3,4 ... ... like round robin generation. 
+    it would pass the convergence test I mentioned above, but it's not uniformly random. 
+    So does any pattern that shows a deterministic pattern, like 1,2,3,4,4,3,2,1 ...
+    Any idea about how to test rand4() is truely uniformly random?
+    Soln:
+        Can you generate a list of numbers by re-rolling rand3() the LCM of 3 and 4 which is 12.
+        So role rand3 4 times, sum it, do mod 4 -> and tell me what you get.
+        ^ Does this work??
+        Something about REJECTION SAMPLING is how you do it!!
+    Any idea about how to test rand4() is truely uniformly random?
+    There are probably better ways, but... you could run rand4 a million times, store the results in a 1MB file, 
+    and let a good compression program compress it. It should result in about 250KB. If there are easy patterns 
+    like your examples or if the distribution is significantly non-uniform, it will be significantly smaller.
+    
+    https://leetcode.com/problems/implement-rand10-using-rand7/
+    Given the API rand7() that generates a uniform random integer in the range [1, 7], 
+    write a function rand10() that generates a uniform random integer in the range [1, 10]. 
+    You can only call the API rand7(), and you shouldn't call any other API. Please do not use a language's built-in random API.
+    Each test case will have one internal argument n, the number of times that your implemented 
+    function rand10() will be called while testing. Note that this is not an argument passed to rand10().
+    Use REJECTION SAMPLING
+    Intuition
+    What if you could generate a random integer in the range 1 to 49? How would you generate a random integer 
+    in the range of 1 to 10? What would you do if the generated number is in the desired range? What if it is not?
+
+    Algorithm
+
+    This solution is based upon Rejection Sampling. The main idea is when you generate a number in the desired range,
+    output that number immediately. If the number is out of the desired range, reject it and re-sample again. 
+    As each number in the desired range has the same probability of being chosen, a uniform distribution is produced.
+
+    Obviously, we have to run rand7() function at least twice, as there are not enough numbers in the range of 1 to 10. 
+    By running rand7() twice, we can get integers from 1 to 49 uniformly. Why?
+    -> because we can generate 2 indexes into an array!
+     1 2 3 4 5 6 7
+   1 1 2 3 4 5 6 7 
+   2 8 91011121314
+   3 ...
+   4
+   5
+   6
+   7
+    aka -> 7 * i + j = 7*6 + 7 == 49?
+    
+        class Solution {
+        public:
+            int rand10() {
+                int row, col, idx;
+                do {
+                    row = rand7();
+                    col = rand7();
+                    idx = col + (row - 1) * 7;
+                } while (idx > 40);
+                return 1 + (idx - 1) % 10;
+        };  
+    
+    Why not return 1 + (idx % 10) for Approach 1?
+    -> 
+        Because we are using % operation here, we need to do a quick math trick.
+        idx is in range 1 2 3 4 5 6 7 8 9 10, if we do nothing, it'll become 1 2 3 4 5 6 7 8 9 "0" after using (% 10).
+        So we need to offset 1, to range 0 1 2 3 4 5 6 7 8 9 at first.
+        After using (% 10), then add 1 back.
+        Now it is correctly in range 1 2 3 4 5 6 7 8 9 10 again.
+        Approach 2: Utilizing out-of-range samples
+        Intuition
+        There are a total of 2.45 calls to rand7() on average when using approach 1. 
+        Can we do better? Glad that you asked. In fact, we are able to improve average 
+        number of calls to rand7() by about 10%.
+        The idea is that we should not throw away the out-of-range samples, but instead use them to 
+        increase our chances of finding an in-range sample on the successive call to rand7.
+        Algorithm
+        Start by generating a random integer in the range 1 to 49 using the aforementioned method. 
+        In the event that we could not generate a number in the desired range (1 to 40), it is equally 
+        likely that each number of 41 to 49 would be chosen. In other words, we are able to obtain integers 
+        in the range of 1 to 9 uniformly. Now, run rand7() again to obtain integers in the range of 1 to 63 uniformly. 
+        Apply rejection sampling where the desired range is 1 to 60. If the generated number is in the desired range (1 to 60), 
+        we return the number. If it is not (61 to 63), we at least obtain integers of 1 to 3 uniformly. Run rand7() again to 
+        obtain integers in the range of 1 to 21 uniformly. The desired range is 1 to 20, and in the unlikely event we 
+        get a 21, we reject it and repeat the entire process again.
+        class Solution {
+        public:
+            int rand10() {
+                int a, b, idx;
+                while (true) {
+                    a = rand7();
+                    b = rand7();
+                    idx = b + (a - 1) * 7;
+                    if (idx <= 40)
+                        return 1 + (idx - 1) % 10;
+                    a = idx - 40;
+                    b = rand7();
+                    // get uniform dist from 1 - 63
+                    idx = b + (a - 1) * 7;
+                    if (idx <= 60)
+                        return 1 + (idx - 1) % 10;
+                    a = idx - 60;
+                    b = rand7();
+                    // get uniform dist from 1 - 21
+                    idx = b + (a - 1) * 7;
+                    if (idx <= 20)
+                        return 1 + (idx - 1) % 10;
+                }
+            }
+        };
+        Complexity Analysis
+        Time Complexity: O(1)O(1) average, but O(\infty)O(∞) worst case.
+    710. Random Pick With Blacklist (HARD): (TODO: SOLVE BY YOURSELF!)
+        Hard
+        You are given an integer n and an array of unique integers blacklist. Design an algorithm to pick a random integer in 
+        the range [0, n - 1] that is not in blacklist. Any integer that is in the mentioned range and not in blacklist should 
+        be equally likely to be returned.
+        Optimize your algorithm such that it minimizes the number of calls to the built-in random function of your language.
+
+        Implement the Solution class:
+        Solution(int n, int[] blacklist) Initializes the object with the integer n and the blacklisted integers blacklist.
+        int pick() Returns a random integer in the range [0, n - 1] and not in blacklist.
+        
+        Example 1:
+        Input
+        ["Solution", "pick", "pick", "pick", "pick", "pick", "pick", "pick"]
+        [[7, [2, 3, 5]], [], [], [], [], [], [], []]
+        Output
+        [null, 0, 4, 1, 6, 1, 0, 4]
+
+        Explanation
+        Solution solution = new Solution(7, [2, 3, 5]);
+        solution.pick(); // return 0, any integer from [0,1,4,6] should be ok. Note that for every call of pick,
+                        // 0, 1, 4, and 6 must be equally likely to be returned (i.e., with probability 1/4).
+        solution.pick(); // return 4
+        solution.pick(); // return 1
+        solution.pick(); // return 6
+        solution.pick(); // return 1
+        solution.pick(); // return 0
+        solution.pick(); // return 4
+        
+        Harman Soln 
+        Just swap blacklisted elements to end of an array initalized from size 0 to N:
+        Use 2 pointer to keep track of border of elements that are blacklisted on the right side,
+        left pointer indicates elements not in blacklist and between left and right are unprocessed elements!
+        move right until you are on non blaclist item, and move left until you find a blacklist item and swap with right!
+        Then choose betweel [0 to L - sizeOfBlacklist]
+        Better soln is to have map initialized of size Blacklist and do following:
+        Treat the first N - |B| numbers as those we can pick from. Iterate through the blacklisted 
+        numbers and map each of them to to one of the remaining non-blacklisted |B| numbers
+        For picking, just pick a random uniform int in 0, N - |B|. If its not blacklisted, 
+        return the number. If it is, return the number that its mapped to
+        import random
+           
            # below solution is incorrect, not fair because for i in range is going from self.lenght, self.N, actually 
            # nvm that is correct. since we assume the blacklisted items are there we have to remap the blaklist item to 
            # a correct value that we can fetch. 
            
+        class Solution:
+            def __init__(self, N, blacklist):
+                blacklist = sorted(blacklist)
+                self.b = set(blacklist)
+                self.m = {}
+                self.length = N - len(blacklist)
+                j = 0
+                for i in range(self.length, N):
+                    if i not in self.b:
+                        self.m[blacklist[j]] = i
+                        j += 1
+
+            def pick(self):
+                i = random.randint(0, self.length - 1)
+                return self.m[i] if i in self.m else i    




+###################################################################################
+##################################################################################
+Binary Search Ultimate Handbook:
+    What is binary search?
+    Normally, to find the target in a group, such as an array of numbers, the worst case scenario is we need to go 
+    through every single element (O(n)). However, when these elements are sorted, we are able to take the privilege 
+    of this extra information to bring down the search time to O(log n), that is if we have 100 elements, 
+    the worst case scenario would be 10 searches. That is a huge performance improvement.
+    The Gif below demonstrates the power of binary search.
+    https://assets.leetcode.com/static_assets/posts/1EYkSkQaoduFBhpCVx7nyEA.gif
+    The reason behind this huge performance increase is because for each search iterations, 
+    we are able to cut the elements we will be looking at in half. Fewer elements to look at = faster search time. 
+    And this all comes from the simple fact that in a sorted list, everything to the right of n will be greater or equal to it, and vice versa.
+    Before we look at the abstract ideas of binary search, let's see the code first:
+        var search = function(nums, target) {
+            let lo = 0, hi = nums.length-1;
+            while (lo < hi) {
+                let mid = lo + Math.floor((hi-lo+1)/2);
+                if (target < nums[mid]) {
+                    hi = mid - 1
+                } else {
+                    lo = mid; 
+                }
+            }
+            return nums[lo]==target?lo:-1;
+        };
+    The fundamental idea
+    1. lo & hi
+    We define two variables, let's call them lo and hi . They will store array indexes and they work 
+    like a boundary such that we will only be looking at elements inside the boundary.
+    Normally, we would want initialize the boundary to be the entire array.
+    let lo = 0, hi = nums.length-1;
+    2. mid
+    The mid variable indicates the middle element within the boundary. It separates our boundary into 2 parts. 
+    Remember how I said binary search works by keep cutting the elements in half, the mid element works like a 
+    traffic police, it indicates us which side do we want to cut our boundary to.
+    Note when an array has even number of elements, it's your decision to use either the left mid (lower mid) or the right mid (upper mid)
+    let mid = lo + Math.floor((hi - lo) / 2); // left/lower mid
+    let mid = lo + Math.floor((hi - lo + 1) / 2); // right/upper mid
+
+    3. Comparing the target to mid
+    By comparing our target to mid, we can identify which side of the boundary does the target belong. 
+    For example, If our target is greater than mid, this means it must exist in the right of mid . In this case, 
+    there is no reason to even keep a record of all the numbers to its left. And this is the fundamental 
+    mechanics of binary search - keep shrinking the boundary.
+
+    if (target < nums[mid]) {
+        hi = mid - 1
+    } else {
+        lo = mid; 
+    }
+
+
+    4. Keep the loop going
+    Lastly, we use a while loop to keep the search going:
+
+    while (lo < hi) { ... }
+    The while loop only exits when lo == hi, which means there's only one element left. And if we implemented 
+    everything correctly, that only element should be our answer(assume if the target is in the array).
+
+    The pattern
+    It may seem like binary search is such a simple idea, but when you look closely in the code, we are 
+    making some serious decisions that can completely change the behavior of our code.
+    These decisions include:
+
+    Do I use left or right mid?
+    Do I use < or <= , > or >=?
+    How much do I shrink the boundary? is it mid or mid - 1 or even mid + 1 ?
+    ...
+    And just by messing up one of these decisions, either because you don't 
+    understand it completely or by mistake, it's going to break your code.
+    To solve these decision problems, I use the following set of rules to always keep me away from trouble, 
+    most importantly, it makes my code more consistent and predictable in all edge cases.
+    1. Choice of lo and hi, aka the boundary
+    Normally, we set the initial boundary to the number of elements in the array
+    let lo = 0, hi = nums.length - 1;
+    But this is not always the case.
+    We need to remember: the boundary is the range of elements we will be searching from.
+    The initial boundary should include ALL the elements, meaning all the possible answers should be included. 
+    Binary search can be applied to none array problems, such as Math, and this statement is still valid.
+    For example, In LeetCode 35, the question asks us to find an index to insert into the array.
+    It is possible that we insert after the last element of the array, thus the complete range of boundary becomes
+    let lo = 0, hi = nums.length;
+    2. Calculate mid
+    Calculating mid can result in overflow when the numbers are extremely big. I ll demonstrate a few ways of calculating mid from the worst to the best.
+    let mid = Math.floor((lo + hi) / 2) // worst, very easy to overflow
+    let mid = lo + Math.floor((hi - lo) / 2) // much better, but still possible
+    let mid = (lo + hi) >>> 1 // the best, but hard to understand
+    When we are dealing with even elements, it is our choice to pick the left mid or the right mid , 
+    and as I ll be explaining in a later section, a bad choice will lead to an infinity loop.
+    let mid = lo + Math.floor((hi - lo) / 2) // left/lower mid
+    let mid = lo + Math.floor((hi - lo + 1) / 2) // right/upper mid
+    
+    
+    3. How do we shrink boundary
+    I always try to keep the logic as simple as possible, that is a single pair of if...else. 
+    But what kind of logic are we using here? My rule of thumb is always use a logic that you can exclude mid.
+    Let's see an example:
+
+    if (target < nums[mid]) {
+        hi = mid - 1
+    } else {
+        lo = mid; 
+    }
+    Here, if the target is less than mid, there's no way mid will be our answer, and we can 
+    exclude it very confidently using hi = mid - 1. Otherwise, mid still has the 
+    potential to be the target, thus we include it in the boundary lo = mid.
+    On the other hand, we can rewrite the logic as:
+    if (target > nums[mid]) {
+        lo = mid + 1; // mid is excluded
+    } else {
+        hi = mid; // mid is included
+    }
+    
+    
+    4. while loop
+    To keep the logic simple, I always use
+    while(lo < hi) { ... }
+    Why? Because this way, the only condition the loop exits is lo == hi. I 
+    know they will be pointing to the same element, and I know that element always exists.
+    5. Avoid infinity loop
+    Remember I said a bad choice of left or right mid will lead to an infinity loop? Let's tackle this down.
+    Example:
+    let mid = lo + ((hi - lo) / 2); // Bad! We should use right/upper mid!
+    if (target < nums[mid]) {
+        hi = mid - 1
+    } else {
+        lo = mid; 
+    }
+    Now, imagine when there are only 2 elements left in the boundary. If the logic fell into the 
+    else statement, since we are using the left/lower mid, it's simply not doing anything. It just keeps shrinking itself to itself, and the program got stuck.
+    We have to keep in mind that, the choice of mid and our shrinking logic has to work together 
+    in a way that every time, at least 1 element is excluded.
+    let mid = lo + ((hi - lo + 1) / 2); // Bad! We should use left/lower mid!
+    if (target > nums[mid]) {
+        lo = mid + 1; // mid is excluded
+    } else {
+        hi = mid; // mid is included
+    }
+    
+    So when your binary search is stuck, think of the situation when there are only 2 elements left. Did the boundary shrink correctly?
+    TD;DR
+    My rule of thumb when it comes to binary search:
+    Include ALL possible answers when initialize lo & hi
+    Don't overflow the mid calculation
+    Shrink boundary using a logic that will exclude mid
+    Avoid infinity loop by picking the correct mid and shrinking logic
+    Always think of the case when there are 2 elements left
+    Because this problem is a failrly easy, the implementions may be pretty straight forward and you may 
+    wonder why do I need so many rules. However, binary search problems can get much much more complex, and without 
+    consistent rules, it's very hard to write predictable code. In the end, I would say 
+    everybody has their own style of binary serach, find the style that works for you!
+#######################################


###################################################################################
###################################################################################
COOL NOTES PART 0.90: DYNAMIC PROGRAMMING PATTERNS, ILLUSTRATIONS, AND EXAMPLES PART 2: 

    Patterns
    Minimum (Maximum) Path to Reach a Target
    Distinct Ways
    Merging Intervals
    DP on Strings
    Decision Making

    Minimum (Maximum) Path to Reach a Target
    Statement
    Given a target find minimum (maximum) cost / path / sum to reach the target.

    Approach
    Choose minimum (maximum) path among all possible paths 
    before the current state, then add value for the current state.

    routes[i] = min(routes[i-1], routes[i-2], ... , routes[i-k]) + cost[i]
    Generate optimal solutions for all values in the target and return the value for the target.

    for (int i = 1; i <= target; ++i) {
        for (int j = 0; j < ways.size(); ++j) {
            if (ways[j] <= i) {
                dp[i] = min(dp[i], dp[i - ways[j]] + cost / path / sum) ;
            }
        }
    }
    return dp[target]

    Similar Problems

    1.   Min Cost Climbing Stairs Easy
    for (int i = 2; i <= n; ++i) {
        dp[i] = min(dp[i-1], dp[i-2]) + (i == n ? 0 : cost[i]);
    }
    return dp[n]
    
    1.  Minimum Path Sum Medium
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + grid[i][j];
        }
    }
    return grid[n-1][m-1]
    
    1.   Coin Change Medium
    for (int j = 1; j <= amount; ++j) {
        for (int i = 0; i < coins.size(); ++i) {
            if (coins[i] <= j) {
                dp[j] = min(dp[j], dp[j - coins[i]] + 1);
            }
        }
    }

    1.   Minimum Falling Path Sum Medium
    2.   Minimum Cost For Tickets Medium
    3.   2 Keys Keyboard Medium
    4.   Perfect Squares Medium
    5.    Last Stone Weight II Medium
    6.   Triangle Medium
    7.   Ones and Zeroes Medium
    8.   Maximal Square Medium
    9.   Coin Change Medium
    10.   Tiling a Rectangle with the Fewest Squares Hard
    11.  Dungeon Game Hard
    12.  Minimum Number of Refueling Stops Hard

    Distinct Ways
    Statement
    Given a target find a number of distinct ways to reach the target.

    Approach
    Sum all possible ways to reach the current state.

    routes[i] = routes[i-1] + routes[i-2], ... , + routes[i-k]
    Generate sum for all values in the target and return the value for the target.

    for (int i = 1; i <= target; ++i) {
        for (int j = 0; j < ways.size(); ++j) {
            if (ways[j] <= i) {
                dp[i] += dp[i - ways[j]];
            }
        }
    }
    return dp[target]
    
    Similar Problems
    1.  Climbing Stairs easy
    for (int stair = 2; stair <= n; ++stair) {
        for (int step = 1; step <= 2; ++step) {
            dp[stair] += dp[stair-step];   
        }
    }

    1.  Unique Paths Medium
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = dp[i][j-1] + dp[i-1][j];
        }
    }

    1.    Number of Dice Rolls With Target Sum Medium

        You have d dice, and each die has 
        f faces numbered 1, 2, ..., f.

        Return the number of possible ways (out of f^d total ways) modulo 10^9 + 7 
        to roll the dice so the sum of the face up numbers equals target.
        Example 1:
        Input: d = 1, f = 6, target = 3
        Output: 1
        Explanation: 
        You throw one die with 6 faces.  There is only one way to get a sum of 3.
        
        Example 2:
        Input: d = 2, f = 6, target = 7
        Output: 6
        Explanation: 
        You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
        1+6, 2+5, 3+4, 4+3, 5+2, 6+1.

    for (int rep = 1; rep <= d; ++rep) {
        vector<int> new_ways(target+1);
        for (int already = 0; already <= target; ++already) {
            for (int pipe = 1; pipe <= f; ++pipe) {
                if (already - pipe >= 0) {
                    new_ways[already] += ways[already - pipe];
                    new_ways[already] %= mod;
                }
            }
        }
        ways = new_ways;
    }


    Note

    Some questions point out the number of repetitions, 
    in that case, add one more loop to simulate every repetition.

    1.   Knight Probability in Chessboard Medium
    2.   Target Sum Medium
    3.   Combination Sum IV Medium
    4.   Knight Dialer Medium
    5.    Dice Roll Simulation Medium
    6.   Partition Equal Subset Sum Medium
    7.   Soup Servings Medium
    8.   Domino and Tromino Tiling Medium
    9.   Minimum Swaps To Make Sequences Increasing
    10.  Number of Longest Increasing Subsequence Medium
    11. Unique Paths II Medium
    12.  Out of Boundary Paths Medium
    13.   Number of Ways to Stay in the Same Place After Some Steps Hard
    14.   Count Vowels Permutation Hard

    Merging Intervals
    Statement
    Given a set of numbers find an optimal solution 
    for a problem considering the current number 
    and the best you can get from the left and right sides.

    Approach
    Find all optimal solutions for every interval 
    and return the best possible answer.

    // from i to j
    dp[i][j] = dp[i][k] + result[k] + dp[k+1][j]
    Get the best from the left and right sides and add a solution for the current position.

    for(int l = 1; l<n; l++) {
        for(int i = 0; i<n-l; i++) {
            int j = i+l;
            for(int k = i; k<j; k++) {
                dp[i][j] = max(dp[i][j], dp[i][k] + result[k] + dp[k+1][j]);
            }
        }
    }
    return dp[0][n-1]
    
    
    Similar Problems
    1.    Minimum Cost Tree From Leaf Values Medium


    Given an array arr of positive integers, 
    consider all binary trees that can be possibly constructed
    from the arr:

    Each node has either 0 or 2 children;                               
    The values of arr correspond to the values of each 
    leaf in an in-order traversal of the tree.  
    The value of each non-leaf node is equal to the 
    product of the largest leaf value 
    in its left and right subtree respectively.
    Among all possible binary trees considered, return the 
    smallest possible sum of the values of each non-leaf node.  
    It is guaranteed this sum fits into a 32-bit integer.

    for (int l = 1; l < n; ++l) {
        for (int i = 0; i < n - l; ++i) {
            int j = i + l;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; ++k) {
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + maxs[i][k] * maxs[k+1][j]);
            }
        }
    }

    1.  Unique Binary Search Trees Medium
    2.    Minimum Score Triangulation of Polygon Medium
    3.   Remove Boxes Medium
    4.    Minimum Cost to Merge Stones Medium
    5.   Burst Balloons Hard
    6.   Guess Number Higher or Lower II Medium

    DP on Strings
    General problem statement for this pattern 
    can vary but most of the time you are given 
    two strings where lengths of those strings are not big

    Statement
    Given two strings s1 and s2, return some result.

    Approach
    Most of the problems on this pattern requires 
    a solution that can be accepted in O(n^2) complexity.

    // i - indexing string s1
    // j - indexing string s2
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = /*code*/;
            } else {
                dp[i][j] = /*code*/;
            }
        }
    }

    If you are given one string s the approach may little vary

    for (int l = 1; l < n; ++l) {
        for (int i = 0; i < n-l; ++i) {
            int j = i + l;
            if (s[i] == s[j]) {
                dp[i][j] = /*code*/;
            } else {
                dp[i][j] = /*code*/;
            }
        }
    }

    1.    Longest Common Subsequence Medium
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    
    1.   Palindromic Substrings Medium
    for (int l = 1; l < n; ++l) {
        for (int i = 0; i < n-l; ++i) {
            int j = i + l;
            if (s[i] == s[j] && dp[i+1][j-1] == j-i-1) {
                dp[i][j] = dp[i+1][j-1] + 2;
            } else {
                dp[i][j] = 0;
            }
        }
    }

    1.   Longest Palindromic Subsequence Medium
    2.    Shortest Common Supersequence Medium
    3.  Edit Distance Hard
    4.   Distinct Subsequences Hard
    5.   Minimum ASCII Delete Sum for Two Strings Medium
    6. Longest Palindromic Substring Medium

    Decision Making
    The general problem statement for this pattern is 
    forgiven situation decide whether to use or 
    not to use the current state. So, the 
    problem requires you to make a decision at a current state.

    Statement
    Given a set of values find an answer with an 
    option to choose or ignore the current value.

    Approach
    If you decide to choose the current value use the 
    previous result where the value was ignored; 
    vice-versa, if you decide to ignore the 
    current value use previous result where value was used.

    // i - indexing a set of values
    // j - options to ignore j values
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j <= k; ++j) {
            dp[i][j] = max({dp[i][j], dp[i-1][j] + arr[i], dp[i-1][j-1]});
            dp[i][j-1] = max({dp[i][j-1], dp[i-1][j-1] + arr[i], arr[i]});
        }
    }

    1.   House Robber Easy

    for (int i = 1; i < n; ++i) {
        dp[i][1] = max(dp[i-1][0] + nums[i], dp[i-1][1]);
        dp[i][0] = dp[i-1][1];
    }
    
    1.   Best Time to Buy and Sell Stock Easy
    2.   Best Time to Buy and Sell Stock with Transaction Fee Medium
    3.   Best Time to Buy and Sell Stock with Cooldown Medium
    4.   Best Time to Buy and Sell Stock III Hard
    5.   Best Time to Buy and Sell Stock IV Hard

###########################################################################################
############################################################################################
COOL NOTES 0.95 CODEFORCES DP 1

    MEMOIZATION VS FORWARD STYLE DP VS BACKWARD STYLE DP AND RECOVERING THE DP SOLUTION

    Memoization vs DP:

    The memoization approach does not spend time on unnecessary states — 
    it is a lazy algorithm. Only the states which influence the 
    final answer are processed. Here are the pros and cons of memoization 
    over 
    DP: 1[+]. Sometimes easier to code. 
    2[+]. Does not require to specify order on states explicitly. 
    3[+]. Processes only necessary states. 
    4[-]. Works only in the backward-style DP. 
    5[-]. Works a bit slower than DP (by constant).

    Forward vs backward DP style
    backward style. The schema is: iterate through all the states 
    and for each of them calculate the result by looking backward 
    and using the already known DP results of previous states. 
    This style can also be called recurrent since it uses recurrent 
    equations directly for calculation. The relations for backward-style 
    DP are obtained by examining the best solution for the state 
    and trying to decompose it to lesser states. 
    (YOU HAVE SEEN BACKWARD STYLE PLENTY OF TIMES!)

    There is also forward-style DP. Surprisingly it is often more 
    convenient to use. The paradigm of this style is to iterate 
    through all the DP states and from each state perform some transitions 
    leading forward to other states. Each transition modifies the currently 
    stored result for some unprocessed states. When the state is considered, 
    its result is already determined completely. The forward formulation does 
    not use recurrent equations, so it is more complex to prove the correctness 
    of solution strictly mathematically. The recurrent relations used in forward-style 
    DP are obtained by considering one partial solution for the state and trying to 
    continue it to larger states. To perform forward-style DP it is necessary to 
    fill the DP results with neutral values before starting the calculation
    
    Problem 1:
    Given a list of n coins, their weights W1, W2, ..., Wn; and the total sum S. 
    Find the minimum number of coins the overall weight of which is 
    S (we can use as many coins of each type as we want)

    FOWARD STYLE DP (IT USES RELAXTION LIKE DJIKSTRA/BELLMAN FORD):

    The first example will be combinatoric coins problem. 
    Suppose that you have a partial solution with P overall weight. 
    Then you can add arbitrary coin with weight Wi and get overall weight P+Wi. 
    So you get a transition from state (P) to state (P+Wi). When 
    this transition is considered, the result for state (P) is added to 
    the result of state (P+Wi) which means that all the ways to get P weight 
    can be continued to the ways to get P+Wi weight by adding i-th coin. 
    Here is the code.

    /* Recurrent relations (transitions) of DP:
    {k[0] = 1;
    {(P)->k ===> (P+Wi)->nk    add k to nk
    */
    //res array is automatically filled with zeroes
    res[0] = 1;                                 //DP base is the same
    for (int p = 0; p<s; p++)                   //iterate through DP states
        for (int i = 0; i<n; i++) {               //iterate through coin to add
        int np = p + wgt[i];                    //the new state is (np)
        if (np > s) continue;                   //so the transition is (p) ==> (np)
        res[np] += res[p];                      //add the DP result of (p) to DP result of (np)
        }
    int answer = res[s];                        //problem answer is the same

    PROBLEM 2: LCS
    The second example is longest common subsequence problem. 
    It is of maximization-type, so we have to fill the results array 
    with negative infinities before calculation. The DP base is state (0,0)->0 
    which represents the pair of empty prefixes. When we consider partial 
    solution (i,j)->L we try to continue it by three ways: 
    1. Add the next letter of first word to the prefix, do not change subsequence. 
    2. Add the next letter of the second word to the prefix, do not change subsequence. 
    3. Only if the next letters of words are the same: add next letter 
        to both prefixes and include it in the subsequence. 
    
    For each transition we perform so-called relaxation of the larger DP state result. 
    We look at the currently stored value in that state: if it is 
    worse that the proposed one, then it is replaced with the proposed one, 
    otherwise it is not changed. The implementation code and compact 
    representation of DP relations are given below.

    /* Recurrent relations (transitions) of DP:
    {L[0,0] = 0;
    |          /> (i+1,j)->relax(L)
    {(i,j)->L ==> (i,j+1)->relax(L)
                \> (i+1,j+1)->relax(L+1)  (only if next symbols are equal)
    */
    void relax(int &a, int b) {                   //relaxation routine
    if (a < b) a = b;
    }
    
    memset(lcs, -63, sizeof(lcs));              //fill the DP results array with negative infinity
    lcs[0][0] = 0;                              //set DP base: (0,0)->0
    for (int i = 0; i<=n1; i++)
        for (int j = 0; j<=n2; j++) {             //iterate through all states
        int tres = lcs[i][j];
        // Improve the next states from the current state, and what we see rn. 
        // Djikstra relax style!
        relax(lcs[i+1][j], tres);               //try transition of type 1
        relax(lcs[i][j+1], tres);               //try transition of type 2
        if (str1[i] == str2[j])                 //and if next symbols are the same
            relax(lcs[i+1][j+1], tres + 1);       //then try transition of type 3
        }
    int answer = lcs[n1][n2];

    Recovering the best solution for optimization problems

    DP finds only the goal function value itself. 
    It does not produce the best solution along 
    with the numerical answer. 

    There are two ways to get the solution path.
    METHOD 1: REVERSE DP
    The first way is to recalculate the DP from the end to the start. 
    First we choose the final state (f) we want to trace the path from. 
    Then we process the (f) state just like we did it in the DP: 
    iterate through all the variants to get it. Each variant originates 
    in a previous state (p). If the variant produces the result equal 
    to DP result of state (f), then the variant if possible. There is 
    always at least one possible variant to produce the DP result for 
    the state, though there can be many of them. If the variant 
    originating from (p) state is possible, then there is at least 
    one best solution path going through state (p). Therefore we can
    move to state (p) and search the path from starting state to 
    state (p) now. We can end path tracing when we reach the starting state.

    METHOD 2: BACKLINKS

    Another way is to store back-links along with the DP result. 
    For each state (s) we save the parameters of the previous state (u) 
    that was continued. When we perform a transition (u) ==> (s) which 
    produces better result than the currently stored in (s) then we set 
    the back-link to (s) to the state (u). To trace the DP solution path 
    we need simply to repeatedly move to back-linked state until the 
    starting state is met. Note that you can store any additional 
    info about the way the DP result was obtained to simplify 
    solution reconstruction.

    Use first method if memory is a problem. Otherwise use method 2. 

    EXAMPLE OF RECOVERING THE SOLUTION PATH:

    /* Consider the input data: S=11, n=3, W = {1,3,5}
    The DP results + back-links table is:
    P  = 0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |10|11
    -------+--+--+--+--+--+--+--+--+--+--+--
    mink = 0 |1 |2 |1 |2 |1 |2 |3 |2 |3 |2 |3
    prev = ? |S0|S1|S0|S1|S0|S1|S2|S3|S4|S5|S6
    item = ? |I0|I0|I1|I1|I2|I2|I2|I2|I2|I2|I2
    */
    
    int mink[MAXW];                       //the DP result array
    int prev[MAXW], item[MAXW];           //prev  -  array for back-links to previous state
    int k;                                //item  -  stores the last item index
    int sol[MAXW];                        //sol[0,1,2,...,k-1] would be the desired solution
    
    memset(mink, 63, sizeof(mink));     //fill the DP results with positive infinity
    mink[0] = 0;                        //set DP base (0)->0
    for (int p = 0; p<s; p++)           //iterate through all states
        for (int i = 0; i<n; i++) {       //try to add one item
        int np = p + wgt[i];            //from (P)->k we get
        int nres = mink[p] + 1;         //to (P+Wi)->k+1
        if (mink[np] > nres) {          //DP results relaxation
            mink[np] = nres;              //in case of success
            prev[np] = p;                 //save the previous state
            item[np] = i;                 //and the used last item
        }
        }
    
    int answer = mink[s];
    int cp = s;                         //start from current state S
    while (cp != 0) {                   //until current state is zero
        int pp = prev[cp];                //get the previous state from back-link
        sol[k++] = item[cp];              //add the known item to solution array
        cp = pp;                          //move to the previous state
    }


###########################################################################################
############################################################################################
COOL NOTES 0.98 CODEFORCES DP 2 
    OPTIMIZING DP

    -> Consolidate equivalent states

        Consider TSP problem as an example. The bruteforce recursive 
        solution searches over all simple paths from city 0 recursively. 
        State of this solution is any simple path which starts from city 0. 
        In this way a state domain is defined and the recursive relations 
        for them are obvious. But then we can notice that states (0,A,B,L) 
        and (0,B,A,L) are equivalent. It does not matter in what order the 
        internal cities were visited — only the set of visited cities and the 
        last city matter. It means that our state domain is redundant, so let's 
        merge the groups of equivalent states. We will get state domain (S,L)->R 
        where S is set of visited states, L is the last city in the path and R is the 
        minimal possible length of such path. The recursive solution with O((N-1)!) 
        states is turned into DP over subsets with O(2^N*N) states.

    -> Prune impossible states
        The state is impossible if its result is always equal to 
        zero(combinatorial) / infinity(minimization). Deleting such a state is a 
        good idea since it does not change problem answer for sure. 
        The impossible states can come from several sources:
       
       1. Explicit parameter dependence. The state domain is (A,B) and we know that for all 
          possible states B = f(A) where f is some function (usually analytic and simple). In 
          such a case B parameter means some unnecessary information and can be deleted from 
          state description. The state domain will be just (A). If we ever need parameter B to 
          perform a transition, we can calculate it as f(A). As the result the size of 
          state domain is decreased dramatically.

       2. Implicit parameter dependence. This case is worse. We have state domain (A,B) 
          where A represents some parameters and we know that for any possible state f(A,B) = 0. 
          In other words, for each possible state some property holds. The best idea is of course 
          to express one of parameters as explicitly dependent on the others. Then we can go to 
          case 1 and be happy=) Also if we know that B is either f1(A) or f2(A) or f3(A) or ... or fk(A) 
          then we can change state domain from (A,B) to (A,i) where i=1..k is a number of equation 
          B variant. If nothing helps, we can always use approach 4.

       3. Inequalities on parameters. The common way to exploit inequalities 
          for parameters is to set tight loop bounds. For example, if state domain 
          is (i,j) and i<j then we can write for(i=0;i<N;i++) for(j=i+1;j<N;j++) 
          or for(j=0;j<N;j++) for(i=0;i<j;i++). In this case we avoid processing impossible 
          states and average speedup is x(2). If there are k parameters which are 
          non-decreasing, speedup will raise to x(k!).

       4. No-thinking ways. Even if it is difficult to determine 
           which states are impossible, the fact of their existence 
           itself can be exploited. There are several ways:

            A. Discard impossible states and do not process them. Just add 
            something like "if (res[i][j]==0) continue;" inside loop which iterates over 
            DP states and you are done. This optimization should be always used because 
            overhead is tiny but speedup can be substantial. It does not decrease size of 
            state domain but saves time from state processing.

            B. Use recursive search with memoization. It will behave very similar to DP 
            but will process only possible states. The decision to use it must be made before 
            coding starts because the code differs from DP code a lot.

            C. Store state results in a map. This way impossible states do not eat memory and time at all. 
            Unfortunately, you'll have to pay a lot of time complexity for this technique: O(log(N)) with 
            ordered map and slow O(1) with unordered map. And a big part of code must 
            be rewritten to implement it.>

    -> Store results only for two layers of DP state domain

        The usual case is when result for state (i,A) is dependent only on 
        results for states (i-1,*). In such a DP only two neighbouring layers 
        can be stored in memory. Results for each layer are discarded after 
        the next one is calculated. The memory is then reused for the next layer and so on.


        Memory contains two layers only. One of them is current layer and another one is previous layer 
        (in case of forward DP one layer is current and another is next). After current layer is fully processed, 
        layers are swapped. There is no need to swap their contents, just swap their pointers/references. 
        Perhaps the easiest approach is to always store even-numbered layers in one memory buffer(index 0) 
        and odd-numbered layers in another buffer(index 1). To rewrite complete and working 
        DP solution to use this optimization you need to do only: 
        1. In DP results array definition, change the first size of array to two. 
        2. Change indexation [i] to [i&1] in all accesses to this array. (i&1 is equal to i modulo 2). 
        3. In forward DP clear the next layer contents immediately after loop over layer 
              index i. Here is the code example of layered forward-style minimization DP: 
              int res[2][MAXK]; //note that first dimension is only 2 
              for (int i = 0; i<N; i++) 
              { 
                  memset(res[(i+1)&1], 63, sizeof(res[0])); //clear the contents of next layer (set to infinity) 
                  for (int j = 0; j<=K; j++) { //iterate through all states as usual 
                    int ni = i + 1; //layer index of transition destination is fixed 
                    int nj, nres = DPTransition(res[i&1][j], ???); //get additional parameters and results somehow 
                    if (res[ni&1][nj] > nres) //relax destination result 
                        res[ni&1][nj] = nres; //note &1 in first index 
                  } 
            }

        This technique reduces memory requirement in O(N) times which is 
        often necessary to achieve desired space complexity. If sizeof(res) 
        reduces to no more than several megabytes, then speed performance 
        can increase due to cache-friendly memory accesses.

        Sometimes you have to store more than two layers. If DP transition relations 
        use not more that k subsequent layers, then you can store only k layers in memory. 
        Use modulo k operation to get the array index 
        for certain layer just like &1 is used in the example.

        There is one problem though. In optimization problem there is no simple way 
        to recover the path(solution) of DP. You will get the goal function of best solution, 
        but you won't get the solution itself. To recover solution in usual 
        way you have to store all the intermediate results.

        There is a general trick which allows to recover path 
        without storing all the DP results. The path can be recovered using divide and conquer method. 
        Divide layers into approximately two halves and choose the middle layer with index m in 
        between. Now expand the result of DP from (i,A)->R to (i,A)->R,mA where mA is the "middle state". 
        It is value of additional parameter A of the state that lies both on the path and in the middle layer. 
        Now let's see the following: 
        1. After DP is over, problem answer is determined as minimal result in final layer (with certain properties maybe). 
        2. Let this result be R,mA. Then (m,mA) is the state in the middle of the path we want to recover.
        3. The results mA can be calculated via DP. Now we know the final and the middle states of the desired path. 
        4. Divide layers into two halves and launch the same DP for each part recursively. 
        5. Choose final state as answer for the right half and middle state as answer for the left half. 
        6. Retrieve two states in the middle of these halves and continue recursively.
        7.  This technique requires additional O(log(N)) time complexity because result for each layer 
        8.  is recalculated approximately log(N) times.
        9.  If some additional DP parameter is monotonous (for each transition (i,A) — (i+1,B) inequality A<=B holds) 
            then domain of this parameter can also be divided into two halves by the middle point. 
            In such a case asymptotic time complexity does not increase at all.


    -> Precalculate

        Often DP solution can benefit from precalculating something. 
        Very often the precalculation is simple DP itself.

        A lot of combinatorial problems require precalculation of binomial coefficients. 
        You can precalculate prefix sums of an array so that you can calculate sum of 
        elements in segment in O(1) time. Sometimes it is beneficial to 
        precalculate first k powers of a number.

        Although the term precalculation refers to the calculations which are going 
        before the DP, a very similar thing can be done in the DP process. 
        For example, if you have state domain (a,b)->R you may find it useful to create 
        another domain (a,k)->S where S is sum of all R(a,b) with b<k. 
        It is not precisely precalculation since it expands the DP state domain, 
        but it serves the same goal: spend some additional 
        time for the ability to perform a particular operation quickly.>

    -> Rotate the optimization problem (ROTATION TECHNIQUE DONE IN ConnectTheCities.py in Competition folder)


        There is a DP solution with state domain (W,A)->R for maximization problem, 
        where W is weight of partial solution, A is some additional parameter 
        and R is maximal value of partial solution that can be achieved. 
        The simple problem unbounded knapsack problem will serve as an example for DP rotation.

        Let's place additional requirement on the DP: if we increase weight W of partial 
        solution without changing other parameters including result the solution worsens. 
        Worsens means that the solution with increased weight can be discarded if the initial 
        solution is present because the initial solution leads to better problem answer 
        than the modified one. Notice that the similar statement must true for result R in 
        any DP: if we increase the result R of partial solution the solution improves. 
        In case of knapsack problem the requirement is true: we have some partial solution; 
        if another solution has more weight and less value, then it is surely worse t
        han the current one and it is not necessary to process it any further. 
        The requirement may be different in sense of sign (minimum/maximum. worsens/improves).

        This property allows us to state another "rotated" DP: (R,A)->W where R is the value of partial solution, 
        A is the same additional parameter, and W is the minimal possible weight for such a partial solution. 
        In case of knapsack we try to take items of exactly R overall value with the least overall 
        weight possible. The transition for rotated DP is performed the same way. 
        The answer for the problem is obtained as usual: iterate through all 
        states (R,A)->W with desired property and choose solution with maximal value.

        To understand the name of the trick better imagine a grid on the plane 
        with coordinates (W,R) where W is row index and R is column index. 
        As we see, the DP stores only the result for rightmost(max-index) cell in each row. 
        The rotated DP will store only the uppermost(min-index) cell in each column. 
        Note the DP rotation will be incorrect if the requirement stated above does not hold.

        The rotation is useful only if the range of possible values R is much less than the 
        range of possible weights W. The state domain will take O(RA) memory instead of O(WA) 
        which can help sometimes. For example consider the 0-1 knapsack problem with arbitrary 
        positive real weights and values. DP is not directly applicable in this case. 
        But rotated DP can be used to create fully polynomial approximation scheme which 
        can approximate the correct answer with relative error not more than arbitrary threshold. 
        The idea is to divide all values by small eps and round to the nearest integer. 
        Then solve DP with state domain (k,R)->W where k is number of already processed items, 
        R is overall integer value of items and W is minimal possible overall weight. Note that 
        you cannot round weights in knapsack problem because the optimal solution you obtain 
        this way can violate the knapsack size constraint.

    -> Calculate matrix power by squaring

        This technique deals with layered combinatorial DP solution with transition independent 
        of layer index. Two-layered DP has state domain (i,A)->R and recurrent rules 
        in form R(i+1,A) = sum(R(i,B)*C(B)) over all B parameter values. It is important that 
        recurrent rules does not depend on the layer index.

        Let's create a vector V(i) = (R(i,A1), R(i,A2), ..., R(i,Ak)) where Aj iterates through 
        all possible values of A parameter. This vector contains all the results on i-th layer. 
        Then the transition rule can be formulated as matrix multiplication: V(i+1) = M * V(i) 
        where M is transition matrix. The answer for the problem is usually determined by the 
        results of last layer, so we need to calculate V(N) = M^N * V(0).

        The DP solution is to get V(0) and then multiply it by M matrix N times. It requires 
        O(N * A^2) time complexity, or more precisely it requires O(N * Z) time where Z is 
        number of non-zero elements of matrix M. Instead of one-by-one matrix multiplication, 
        exponentiation by squaring can be used. It calculates M^N using O(log(N)) matrix multiplications. 
        After the matrix power is available, we multiply vector V(0) by it and instantly get the results 
        for last layer. The overall time complexity is O(A^3 * log(N)). This trick is necessary when A is 
        rather small (say 200) and N is very large (say 10^9).

    -> Use complex data structures and algorithms

        Sometimes tough DP solutions can be accelerated by using complex acceleration 
        structures or algorithms. Binary search, segment trees (range minimum/sum query), 
        binary search tree (map) are good at accelerating particular operations. If you 
        are desperate at inventing DP solution of Div1 1000 problem with 
        proper time complexity, it may be a good idea to recall these things.

        For example, longest increasing subsequence problem DP solution can be accelerated to 
        O(N log(N)) with dynamic range minimum query data structure 
        or with binary search depending on the chosen state domain.

#######################################################################################################
#######################################################################################################

CODEFORCES DP 3 - STATE TRANSITIONS, AND STATES, AND RECURRENT RELATIONSHIPS

-> Overview 
    Solution Code of DP solution usually contains an array representing 
    subresults on the state domain. For example, classic knapsack problem solution will be like (FORWARD DP):

    int maxcost[items+1][space+1];
    memset(maxcost, -63, sizeof(maxcost));   //fill with negative infinity
    maxcost[0][0] = 0;                       //base of DP
    for (int i = 0; i<items; i++)            //iterations over states in proper order
        for (int j = 0; j<=space; j++) {
        int mc = maxcost[i][j];              //we handle two types forward transitions
        int ni, nj, nmc;                     //from state (i,j)->mc to state (ni,nj)->nmc
    
        ni = i + 1;                          //forward transition: do not add i-th item
        nj = j;
        nmc = mc;      
        if (maxcost[ni][nj] < nmc)           //relaxing result for new state
            maxcost[ni][nj] = nmc;
    
        ni = i + 1;                          //forward transition: add i-th item
        nj = j + size[i];
        nmc = mc + cost[i];
        if (nj <= space && maxcost[ni][nj] < nmc)
            maxcost[ni][nj] = nmc;
        }
    int answer = -1000000000;                //getting answer from state results
    for (j = 0; j<=space; j++)
        if (maxcost[items][j] > answer)
        answer = maxcost[items][j];
    return answer;

    Here (i,j) is state of DP with result equal to maxcost[i][j]. 
    The result here means the maximal cost of items we can get by taking some of first i items with 
    overall size of exactly j. So the set of (i,j) pairs and concept of maxcost[i][j] here 
    comprise a state domain. The forward transition is adding or not adding the i-th item to 
    the set of items we have already chosen.

    The order of iterations through all DP states is important. The code above 
    iterates through states with pairs (i,j) sorted lexicographically. It is correct 
    since any transition goes from set (i,*) to set (i+1,*), so we see that i is increasing 
    by one. Speaking in backward (recurrent) style, the result for each state (i,j) directly 
    depends only on the results for the states (i-1,*).

    To determine order or iteration through states we have to define order on state domain. 
    We say that state (i1,j1) is greater than state (i2,j2) if (i1,j1) directly or indirectly 
    (i.e. through several other states) depends on (i2,j2). This is definition of order on the 
    state domain used. In DP solution any state must be considered after all the lesser states. 
    Else the solution would give incorrect result.

-> Multidimensional array 
    The knapsack DP solution described above is an example of multidimensional array state domain (with 2 
    dimensions). A lot of other problems have similar state domains. Generally 
    speaking, in this category states are represented by k   parameters: (i1, i2, i3, ..., ik). 
    So in the code we define a multidimensional array for state results like: 
    int Result[N1][N2][N3]...    [Nk]. Of course there are some transition rules 
    (recurrent relations). These rules themselves can be complex, but the order of states   
    is usually simple.

    In most cases the states can be iterated through in lexicographical order. 
    To do this you have to ensure that if I = (i1, i2, i3, ...,  ik) directly 
    depends on J = (j1, j2, j3, ..., jk) then I is lexicographically greater that J. 
    This can be achieved by permuting  parameters (like using (j,i) instead of (i,j)) 
    or reversing them. But it is usually easier to change the order and direction of nested   
    loops. Here is general code of lexicographical traversion:

      for (int i1 = 0; i1<N1; i1++)
        for (int i2 = 0; i2<N1; i2++)
          ...
            for (int ik = 0; ik<Nk; ik++) {
              //get some states (j1, j2, j3, ..., jk) -> jres by performing transitions
              //and handle them
            }
    Note: changing order of DP parameters in array and order of nested loops 
    can noticably affect performance on modern computers due to    
    CPU cache behavior.

    This type of state domain is the easiest to understand and implement, that's why 
    most DP tutorials show problems of this type. But it   is not the most 
    frequently used type of state domain in SRMs. DP over subsets is much more popular.

->  Subsets of a given set

    The problems of this type has some set X. The number of elements in this set is small: 
    less than 20. The idea of DP solution is to  consider all subsets of X as 
    state domain. Often there are additional parameters. So generally we 
    have state domain in form (s,a) where  s is a subset of X and "a" represents additional parameters.

    Consider TSP problem as an example. The set of cities X={0, 1, 2, ..., N-1} 
    is used here. State domain will have two parameters: s and  a. 
    The state (s,a)->R means that R is the shortest path from city 0 to 
    city "a" which goes through all the vertices from subset s    
    exactly once. The transition is simply adding one city v to the 
    end of path: (s,a)->R turns into (s+{v},v)->R + M[a,v]. Here M[i,j] 
    is     distance between i-th and j-th city. Any hamiltonian cycle is a 
    path which goes through each vertex exactly once plus the edge which    
    closes the cycle, so the answer for TSP problem can be computed as min(R[X,a]+M[a,0]) among all vertices "a".

    It is very convenient to encode subsets with binary numbers. 
    Look recipe "Representing sets with bitfields" for detailed explanation.

    The state domain of DP over subsets is usually ordered by set 
    inclusion. Each forward transition adds some elements to the current  
    subset, but does not subtract any. So result for each state (s,a) depends 
    only on the results of states (t,b) where t is subset of s.    
    If state domain is ordered like this, then we can iterate through subsets 
    in lexicographical order of binary masks. Since subsets are  
    usually represented with binary integers, we can iterate through 
    all subsets by iterating through all integers from 0 to 2^N — 1. For    
    example in TSP problem solution looks like:

      int res[1<<N][N];
      memset(res, 63, sizeof(res));       //filling results with positive infinity
      res[1<<0][0] = 0;                   //DP base
    
      for (int s = 0; s < (1<<N); s++)    //iterating through all subsets in lexicographical order
        for (int a = 0; a < N; a++) {
          int r = res[s][a];
          for (int v = 0; v < N; v++) {   //looking through all transitions (cities to visit next)
            if (s & (1<<v)) continue;     //we cannot visit cities that are already visited
            int ns = s | (1<<v);          //perform transition
            int na = v;
            int nr = r + matr[a][v];      //by adding edge (a &mdash; v) distance
            if (res[ns][na] > nr)         //relax result for state (ns,na) with nr
              res[ns][na] = nr;
          }
        }
      int answer = 1000000000;            //get TSP answer
      for (int a = 0; a < N; a++)
        answer = min(answer, res[(1<<N)-1][a] + matr[a][0]);

    Often in DP over subsets you have to iterate through all subsets or 
    supersets of a given set s. The bruteforce implementation will  
    require O(4^N) time for the whole DP, but it can be easily optimized to 
    take O(3^N). Please read recipe "Iterating Over All Subsets of a Set".


-> Substrings of a given string

    There is a fixed string or a fixed segment. According to the problem 
    definition, it can be broken into two pieces, then each of pieces  
    can be again divided into two pieces and so forth until we get unit-length strings. 
    And by doing this we need to achieve some goal.

    Classical example of DP over substrings is context-free grammar parsing algorithm. 
    Problems which involve putting parentheses to    
    arithmetic expression and problems that ask to optimize the overall 
    cost of recursive breaking are often solved by DP over substrings.     
    In this case there are two special parameters L and R which represent indices of left 
    and right borders of a given substring. There can     
    be some additional parameters, we denote them as "a". So each 
    state is defined by (L,R,a). To calculate answer for each state, 
    all the  ways to divide substring into two pieces are considered. 
    Because of it, states must be iterated through in order or non-decreasing   
    length. Here is the scheme of DP over substrings (without additional parameters):

      res[N+1][N+1];                          //first: L, second: R
      for (int s = 0; s<=N; s++)              //iterate size(length) of substring
        for (int L = 0; L+s<=N; L++) {        //iterate left border index
          int R = L + s;                      //right border index is clear
          if (s <= 1) {                       
            res[L][R] = DPBase(L, R);         //base of DP &mdash; no division
            continue;
          }
          tres = ???;                          
          for (int M = L+1; M<=R-1; M++)      //iterate through all divisions
            tres = DPInduction(tres, res[L][M], res[M][R]);
          res[L][R] = tres;
        }
      answer = DPAnswer(res[0][N]);

-> Subtrees(vertices) of a given rooted tree

    The problem involves a rooted tree. Sometimes a graph is given 
    and its DFS search tree is used. Some sort of result can be calculated   
    for each subtree. Since each subtree is uniquely identified by its root, 
    we can treat DP over subtrees as DP over vertices. The result    
    for each non-leaf vertex is determined by the results of its immediate children.

    The DP over subtree has a state domain in form (v,a) where v is a root of 
    subtree and "a" may be some additional parameters. states are     
    ordered naturally be tree order on vertices. Therefore the easiest way to 
    iterate through states in correct order is to launch DFS from     
    the root of tree. When DFS exits from a vertex, its result must 
    be finally computed and stored in global memory. 
    The code generally     looks like:

      bool vis[N];                                  //visited mark for DFS
      res[N];                                       //DP result array
    
      void DFS(int v) {                             //visit v-rooted subtree recursively
        vis[v] = true;                              //mark vertex as visited
        res[v] = ???;                               //initial result, which is final result in case v is leaf
        for (int i = 0; i<nbr[v].size(); i++) {     //iterate through all sons s
          int s = nbr[v][i];                        
          if (!vis[s]) {                            //if vertex is not visited yet, then it's a son in DFS tree
            DFS(s);                                 //visit it recursively
            res[v] = DPInduction(res[v], res[s]);   //recalculate result for current vertex
          }
        }
      }
      ...
      memset(vis, false, sizeof(vis));              //mark all vertices as not visited
      DFS(0);                                       //run DFS from the root = vertex 0
      answer = DPAnswer(res[0]);                    //get problem answer from result of root

    Sometimes the graph of problem is not connected (e.g. a forest). 
    In this case run a series of DFS over the whole graph. The results for     
    roots of individual trees are then combined in some way. 
    Usually simple summation/maximum or a simple formula is enough 
    but in tough cases this "merging problem" can turn out to require 
    another DP solution.

    The DPInduction is very simple in case when there are no additional parameters. 
    But very often state domain includes the additional     
    parameters and becomes complicated. DPInduction turns out to be 
    another(internal) DP in this case. Its state domain is (k,a) where k is     
    number of sons of vertex considered so far and "a" is additional info. 
    Be careful about the storage of results of this internal DP. 
    If  you are solving optimization problem and you are required to 
    recover the solution (not only answer) then you have to save results of     
    this DP for solution recovering. In this case you'll have an array 
    globalres[v,a] and an array internalres[v,k,a]. Topcoder problems    r
    arely require solution, so storage of internal DP results is not necessary. 
    It is easier not to store them globally. In the code below    
    internal results for a vertex are initialized after all the sons 
    are traversed recursively and are discarded after DFS exits a vertex.     
    This case is represented in the code below:

      bool vis[N];
      gres[N][A];
      intres[N+1][A];
    
      void DFS(int v) {
        vis[v] = true;
    
        vector<int> sons;
        for (int i = 0; i<nbr[v].size(); i++) {    //first pass: visit all sons and store their indices
          int s = nbr[v][i];
          if (!vis[s]) {
            DFS(s);
            sons.push_back(s);
          }
        }
    
        int SK = sons.size();                      //clear the internal results array
        for (int k = 0; k<=SK; k++)
          memset(intres[k], ?, sizeof(intres[k]));
    
        for (int a = 0; a<A; a++)                  //second pass: run internal DP over array of sons
          intres[0][a] = InternalDPBase(v, a);
        for (int k = 0; k<SK; k++)                 //k = number of sons considered so far
          for (int a = 0; a<A; a++)                //a = additional parameter for them
            for (int b = 0; b<A; b++) {            //b = additional parameter for the son being added
              int na = DPTransition(v, a, b);
              int nres = DPInduction(intres[k][a], gres[sons[k]][b]);
              intres[k+1][na] = DPMerge(intres[k+1][na], nres);
            }
        for (int a = 0; a<A; a++)                  //copy answer of internal DP to result for vertex
          gres[v][a] = intres[SK][a];
      }
      ...
      memset(vis, false, sizeof(vis));              //series of DFS
      for (int v = 0; v<N; v++) if (!vis[v]) {
        DFS(v);
        ???                                         //handle results for connected component
      }
      ???                                           //get the answer in some way

    It is very important to understand how time/space complexity is calculated 
    for DP over subtrees. For example, the code just above   
    requires O(N*A^2) time. Though dumb analysis says it 
    is O(N^2*A^2): {N vertices} x {SK<=N sons for each} x A x A. 
    Let Ki denote number    of sons of vertex i. Though each Ki may 
    be as large as N-1, their sum is always equal to N-1 in a rooted tree. 
    This fact is the key to     
    further analysis. Suppose that DFS code for i-th vertex runs in not 
    more than Ki*t time. Since DFS is applied only once to each vertex,     
    the overall time will be TC(N) = sum(Ki*t) <= N*t. Consider t=A^2 for the case 
    above and you'll get O(N*A^2) time complexity. To    benefit from this acceleration, 
    be sure not to iterate through all vertices of graph in DFS. For example above, 
    running memset for the     whole intres array in DFS will raise the time complexity. 
    Time of individual DFS run will become O(N*A + Ki*A^2) instead of O(Ki*A^2).  
    The overall time complexity will become O(N^2*A + N*A^2) which is great regress 
    in case if A is much smaller that N. Using the same  approach you may achieve 
    O(N*A) space complexity in case you are asked to recover solution. We have already 
    said that to recover     solution you have to store globally the array 
    internalres[v,k,a]. If you allocate memory for this array dynamically, then 
    you can   ignore completely states with k>Ki. Since the sum of all Ki is N, 
    you will get O(N*A) space.

-> Layer count + layer profile

    This is the toughest type of DP state domain. It is usually used in 
    tiling or covering problems on special graphs. The classic examples     
    are: calculate number of ways to tile the rectangular board with dominoes 
    (certain cells cannot be used); or put as many chess figures  
    on the chessboard as you can so that they do not hit each other 
    (again, some cells may be restricted).

    Generally speaking, all these problems can be solved with DP over subsets 
    (use set of all cells of board). DP with profiles is an   
    optimization which exploits special structure in this set. The board we have 
    to cover/tile is represented as an array of layers. We try   
    to consider layers one by one and store partial solutions after each layer. 
    In simple rectangular board case layer is one row of the  board. 
    The profile is a subset of cells in current row which are already tiled.

    The state domain has form (k,p) where k is number of fully processed layers 
    and p is so-called profile of solution. Profile is the  necessary information 
    about solution in layers that are not fully processed yet. 
    The transitions go from (k,p) to (k+1,q) where q is     
    some new profile. The number of transitions for each state is usually large, 
    so they all are iterated through by recursive search,  
    sometimes with pruning. The search has to find all the ways to 
    increase the partial solution up to the next layer.

    The example code below calculates the number of way to fully 
    cover empty cells on the given rectangular board with dominoes.

    int res[M+1][1<<N];                     
                                            //k = number of fully tiled rows               
    int k, p, q;                            //p = profile of k-th row = subset of tiled cells
    bool get(int i) {                       //q = profile of the next row (in search)        
      return matr[k][i] == '#'              
          || (p & (1<<i));                  //check whether i-th cell in current row is not free
    }
    void Search(int i) {                    //i = number of processed cells in current row
      if (i == N) {
        add(res[k+1][q], res[k][p]);        //the current row processed, make transition
        return;
      }
    
      if (get(i)) {                         //if current cell is not free, skip it
        Search(i+1);
        return;
      }
    
      if (i+1<N && !get(i+1))               //try putting (k,i)-(k,i+1) domino
        Search(i+2);
    
      if (k+1<M && matr[k+1][i] != '#') {   //try putting (k,i)-(k+1,i) domino
        q ^= (1<<i);                        //note that the profile of next row is changed
        Search(i+1);
        q ^= (1<<i);
      }
    }
    ...
    res[0][0] = 1;                          //base of DP
    for (k = 0; k<M; k++)                   //iterate over number of processed layers
      for (p = 0; p<(1<<N); p++) {          //iterate over profiles
        q = 0;                              //initialize the new profile
        Search(0);                          //start the search for all transitions
      }
    int answer = res[M][0];                 //all rows covered with empty profile = answer

    The asymptotic time complexity is not easy to calculate exactly. 
    Since search for i performs one call to i+1 and one call to i+2, the   
    complexity of individual search is not more than N-th Fibonacci number = fib(N). 
    Moreover, if profile p has only F free cells it will     
    require O(fib(F)) time due to pruning. If we sum C(N,F) fib(F) for all F we'll 
    get something like (1+phi)^N, where phi is golden ratio.  
    The overall time complexity is O(M * (1+phi)^N). Empirically it is even lower.

    The code is not optimal. Almost all DP over profiles should use "storing two layers" 
    space optimization. Look "Optimizing DP solution"  recipe. Moreover DP over 
    broken profiles can be used. In this DP state domain (k,p,i) is used, 
    where i is number of processed cells in   a row. No recursive search is 
    launched since it is converted to the part of DP. The time 
    complexity is even lower with this solution.

    The hard DP over profiles examples can include extensions like: 
    1. Profile consists of more than one layer. 
       For example to cover the    grid with three-length tiles you need to store 
       two layers in the profile. 
    2. Profile has complex structure. For example to find optimal    
       in some sense hamiltonian cycle on the rectangular board you have 
       to use matched parentheses strings as profiles. 
    3. Distinct profile  structure. Set of profiles may be different for each layer. 
       You can store profiles in map in this case.

DONE READING -> GO TO COMPETITIVE/PROBLEMS to see all the usages of above techniques. 
https://codeforces.com/blog/entry/43256

#####################################################################################################################
#####################################################################################################################

COOL NOTES PART 1: DYNAMIC PROGRAMMING RECURRENCES EXAMPLES: 
(In the code ->  &mdash; means minus sign. The html was parsed wrong)
(For dp, define subproblem, then recurrence, then base cases, then implement)

1) Given n, find number of diff ways to write n as sum of 1, 3, 4
    Let Dn be the number of ways to write n as the sum of 1, 3, 4
    Recurrence: well n = x1 + x2 + ... + xm. If xm = 1, other terms sum to n-1
    Sums that end with xm=1 is Dn-1

    Recurrence => Dn = Dn-1 + Dn-3 + Dn-4
    Solve base cases D0 = 1, Dn = 0 for all negative n. 
    Code:    
    D[0] = D[1] = D[2] = 1; D[3] = 2;
    for(i = 4; i <= n; i++)
        D[i] = D[i-1] + D[i-3] + D[i-4]


2) Given 2 strings x and y. Find Longest common subsequence. 
    Let Dij be the length of LCS of x1...i, y1...j
    D[i,j] = D[i-1,j-1] + 1 if xi = yj
    D[i,j] = max{D[i-1, j], D[i, j-1]}  otherwise

    Find and solve base cases(In top down, this is the base case for the recursion): 
    D[i, 0] = D[0, j] = 0
    D[0, all the js] = 0
    D[all the is, 0] = 0
    When implementing remember to look at your base cases and 
    understand you have to start with those and build up!
    Helps you figure out directionality!

    def lcs(X , Y): 
        # find the length of the strings 
        m = len(X) 
        n = len(Y) 
    
        # declaring the array for storing the dp values 
        L = [[None]*(n+1) for i in xrange(m+1)] 
    
        """Following steps build L[m+1][n+1] in bottom up fashion 
        Note: L[i][j] contains length of LCS of X[0..i-1] 
        and Y[0..j-1]"""
        for i in range(m+1): 
            for j in range(n+1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j] , L[i][j-1]) 
    
        # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
        return L[m][n] 

    
3) Interval DP
    Given a string x = x1..n, find the min number of chars that need to be inserted to make it a palindrome. 
    Let D[i, j] be the min number of char that need to be inserted to make xi..j into a palindrome

    Consider shortest palindrome y1..k containing xi..j. Either y1 = xi or yk = xj.
    y[2..k-1] is then an optimal solution for x[i+1..j] or x[i..j-1] or x[i+1..j-1]
    Recurrence:

    Dij = 1 + min(D[i+1, j], D[i, j-1]) if xi != xj
    Dij  = D[i+1,j-1] if xi=xj  otherwise   
    
    Base Case: 
    D[i,i] = D[i, i-1] = 0 for all i
    
    Solution subproblem is D[1, n] => i = 1, j = n

    Directionality is hard here:
    //fill in base cases here
    for(t = 2, t<= n; t++):
        for(i = 1, j =t, j <= n; ++i, ++j):
            //fill in D[i][j] here

    => We used t to fill in table correctly! This is interval DP! 
    => This is the correct directionality. DIRECTIONALITY IS BASED ON YOUR BASE CASESE!!!
    => A good way to figure out how to fill up table is just to write out example D[i][j] and figure
        out how you are filling them up with your base cases.

    Solution 2:
    Reverse x to get x-reversed.
    The answer is n - L where L is the length of the LCS of x and x-reversed
    
4) Subset DP => requires bitmasking!
    Given a weighted graph with n nodes, find the shortest path that visits every node exactly once (TSP)
    
    D[S, v] the length of optimal path that visits every node in the set S exactly once and ends at v. 
    there are n2^n subproblems
    Answer is min{v in V such that D[V, v]} where V is the given set of nodes. 

    Base Case: 
    For each node v, D[{v}, v] = 0
    Recurrence:
    consider TSP path. Right before arivign at v, the path comes from some u in S - {v}, and that subpath
    has to be the optimal one that covers S- {v} ending at u.
    Just try all possible candidates for u

    D[S, v] = min[u in S - {v}] (D[S-{v}][u] + cost(u, v) )

    Use integer to represet set. 19 = 010011 represents set {0, 1, 4}
    Union of two sets x and y: x | y
    Set intersection: x & y
    Symmetic difference: x ^ y
    singleton set {i}: 1 << i
    Membership test: x & (1 << i) != 0



##############
################################

TODO: ADD TRAVELING SALESMAN PROBLEM WITH BITMASK SOUTION HERE..


#######################################
##########################################

COOL Notes PART 1.5: DP with bitmasking example: 
    Problem
    Your task will be to calculate number of different assignments of n different 
    topics to n students such that everybody gets exactly one topic he likes.
    1 means the student likes the subject. 0 means they dont.

    Solution:
        Defining the DP state
        So, we can define our DP state with two variables, 
        'k' and 'B' as :

        DP(k,B) = Total no. of possible arrangements of 
        students 0 to k choosing subset of subjects as per bitmask B.

        Time for some optimization. The best way to optimize a DP solution is to 
        reduce its "dimension", i.e, reduce the no. of state variables. 
        In or case, we can see that k = No. of set bits in B. So, k 
        can be easily calculated from B and hence not a part of our DP states anymore.

        The new DP state is :

        DP(B) = Total no. of possible arrangements of students 
        0 to k assigning subjects as per bitmask B.

        where k = count set bits of B
        The base case should be DP(0) = 1 , as there is 
        exactly 1 way to arrange 0 students and 0 subjects.

        For a state B , we can define the recurrence relation as follows :

        where k = number of set bits in B,

            DP(B) = for each i'th set bit in B, 
            check  whether k'th student likes i'th subject. 
            if true, DP(B) += DP(unset i'th bit from B)

        basically, for any arrangement, we remove 
        one bit at a time and add up the resulting 
        state if it satisfies the "liking" criteria. 
        For example :

        DP(1011) = DP(0011) + DP(1001) + DP(1010)

        (assuming 3rd student likes 1st, 3rd and 
        4th subjects. If he didn't like any of 
        those, then those disliked states wouldn't be added)

        logically, this can be explained as follows :

        for a DP state 1011, the 3rd student can be assigned 
        to either 1st,3rd or 4th subject. Now, if the student 
        was assigned to 1st subject, then the number of ways to 
        assign the previous students is given by DP(0011). 
        Similarly, if 3rd student gets 3rd subject, we 
        add DP(1001), and for 4th subject we add DP(1010).

        Implementation
        in practice, we create a single dimensional array of 
        size 1<<20 (i.e. 2^20) . Let it be called DP[ ].
        First, set DP[0] = 1 ;
        then , run a loop from 1 to (1<<n)-1 (i.e. (2^n)-1) to generate all possible bitmasks
        for each index of the loop, apply the recurrence relation as discussed above
        finally, DP[(1<<n)-1] gives the answer.

        //base case
	    dp[0] = 1;
      
        //recurrence relation implemented
        for (int j = 1; j < (1 << n) ; j++) {
            int idx = Integer.bitCount(j);
            for (int k = 0; k < n; k++) {
                if (likes[idx-1][k] == false || (j & (1 << k)) == 0)
                    continue;
                dp[j] += dp[(j & ~(1 << k))];
            }
        }
        
        //final answer
        System.out.println(dp[(1 << n) -1]);

#######################################
##########################################

COOL NOTES 2: EXPRESSION PARSING PROBLEMS

    Reverse Polish notation
        Parsing of simple expressions
        Unary operators
        Right-associativity
        A string containing a mathematical expression containing numbers 
        and various operators is given. We have to compute the value of it in O(n), 
        where n is the length of the string.

        The algorithm discussed here translates an expression into the so-called 
        reverse Polish notation (explicitly or implicitly), and evaluates this expression.

        Reverse Polish notation
        The reverse Polish notation is a form of writing mathematical expressions, 
        in which the operators are located after their operands. 
        
        For example the following expression
        a+b∗c∗d+(e−f)∗(g∗h+i)

        can be written in reverse Polish notation in the following way:
        abc∗d∗+ef−gh∗i+∗+

        The convenience of the reverse Polish notation is, that expressions 
        in this form are very easy to evaluate in linear time. We use a stack, 
        which is initially empty. We will iterate over the operands and operators 
        of the expression in reverse Polish notation. If the current element is a number, 
        then we put the value on top of the stack, if the current element is an 
        operator, then we get the top two elements from the stack, perform the operation, 
        and put the result back on top of the stack. In the end there will be 
        exactly one element left in the stack, which will be the value of the expression.

        Obviously this simple evaluation runs in O(n) time.



    Parsing of simple expressions

        For the time being we only consider a 
        simplified problem: we assume that all operators 
        are binary (i.e. they take two arguments), and all are 
        left-associative (if the priorities are equal, 
        they get executed from left to right). Parentheses are allowed.

        We will set up two stacks: one for numbers, and one for operators 
        and parentheses. Initially both stacks are empty. For the second 
        stack we will maintain the condition that all operations are 
        ordered by strict descending priority. If there are parenthesis on the stack, 
        than each block of operators (corresponding to one pair of parenthesis) 
        is ordered, and the entire stack is not necessarily ordered.

        We will iterate over the characters of the expression from left to right. 
        If the current character is a digit, then we put the value of 
        this number on the stack. If the current character is an 
        opening parenthesis, then we put it on the stack. If the current 
        character is a closing parenthesis, the we execute all operators on the stack 
        until we reach the opening bracket (in other words we perform all 
        operations inside the parenthesis). Finally if the current character 
        is an operator, then while the top of the stack has 
        an operator with the same or higher priority, we will execute 
        this operation, and put the new operation on the stack.

        After we processed the entire string, some operators might 
        still be in the stack, so we execute them.

        Here is the implementation of this method for the four operators + − ∗ /:

        bool delim(char c) {
            return c == ' ';
        }

        bool is_op(char c) {
            return c == '+' || c == '-' || c == '*' || c == '/';
        }

        int priority (char op) {
            if (op == '+' || op == '-')
                return 1;
            if (op == '*' || op == '/')
                return 2;
            return -1;
        }

        void process_op(stack<int>& st, char op) {
            int r = st.top(); st.pop();
            int l = st.top(); st.pop();
            switch (op) {
                case '+': st.push(l + r); break;
                case '-': st.push(l - r); break;
                case '*': st.push(l * r); break;
                case '/': st.push(l / r); break;
            }
        }

        int evaluate(string& s) {
            stack<int> st;
            stack<char> op;
            for (int i = 0; i < (int)s.size(); i++) {
                if (delim(s[i]))
                    continue;

                if (s[i] == '(') {
                    op.push('(');
                } else if (s[i] == ')') {
                    while (op.top() != '(') {
                        process_op(st, op.top());
                        op.pop();
                    }
                    op.pop();
                } else if (is_op(s[i])) {
                    char cur_op = s[i];
                    while (!op.empty() && priority(op.top()) >= priority(cur_op)) {
                        process_op(st, op.top());
                        op.pop();
                    }
                    op.push(cur_op);
                } else {
                    int number = 0;
                    while (i < (int)s.size() && isalnum(s[i]))
                        number = number * 10 + s[i++] - '0';
                    --i;
                    st.push(number);
                }
            }

            while (!op.empty()) {
                process_op(st, op.top());
                op.pop();
            }
            return st.top();
        }

        Thus we learned how to calculate the value of an expression in O(n), 
        at the same time we implicitly used the reverse Polish notation.
        By slightly modifying the above implementation it is also possible 
        to obtain the expression in reverse Polish notation in an explicit form.

    Parsing of all expressions include unary and right associative expr: 

        Unary operators
            Now suppose that the expression also contains unary operators 
            (operators that take one argument). The unary plus and 
            unary minus are common examples of such operators.

            One of the differences in this case, is that we need to 
            determine whether the current operator is a unary or a binary one.

            You can notice, that before an unary operator, there always is 
            another operator or an opening parenthesis, or nothing at 
            all (if it is at the very beginning of the expression). On the contrary 
            before a binary operator there will always be an operand (number) 
            or a closing parenthesis. Thus it is easy to flag 
            whether the next operator can be unary or not.

            Additionally we need to execute a unary and a binary operator 
            differently. And we need to chose the priority of a binary operator 
            higher than all of the binary operations.

            In addition it should be noted, that some unary operators 
            (e.g. unary plus and unary minus) are actually right-associative.

        Right-associativity
            Right-associative means, that whenever the priorities are equal, 
            the operators must be evaluated from right to left.

            As noted above, unary operators are usually right-associative. 
            Another example for an right-associative operator is the 
            exponentiation operator (a∧b∧c is usually perceived as a^(b^c) and not as (a^b)^c.

            What difference do we need to make in order to correctly handle 
            right-associative operators? It turns out that the changes 
            are very minimal. The only difference will be, if the priorities 
            are equal we will postpone the execution of the right-associative operation.

            The only line that needs to be replaced is

            while (!op.empty() && priority(op.top()) >= priority(cur_op))

            with:

            while (!op.empty() && (
                    (left_assoc(cur_op) && priority(op.top()) >= priority(cur_op)) ||
                    (!left_assoc(cur_op) && priority(op.top()) > priority(cur_op))
                ))

            where left_assoc is a function that decides if an 
            operator is left_associative or not.

        Here is an implementation for the binary 
        operators + − ∗ / and the unary operators + and −.

            bool delim(char c) {
                return c == ' ';
            }

            bool is_op(char c) {
                return c == '+' || c == '-' || c == '*' || c == '/';
            }

            bool is_unary(char c) {
                return c == '+' || c=='-';
            }

            int priority (char op) {
                if (op < 0) // unary operator get highest priority
                    return 3; // Negative operators are right associative.
                if (op == '+' || op == '-')
                    return 1;
                if (op == '*' || op == '/')
                    return 2;
                return -1;
            }

            void process_op(stack<int>& st, char op) {
                if (op < 0) {
                    int l = st.top(); st.pop();
                    switch (-op) { // Negative operators are right associative. 
                        case '+': st.push(l); break;
                        case '-': st.push(-l); break;
                    }
                } else {
                    int r = st.top(); st.pop();
                    int l = st.top(); st.pop();
                    switch (op) {
                        case '+': st.push(l + r); break;
                        case '-': st.push(l - r); break;
                        case '*': st.push(l * r); break;
                        case '/': st.push(l / r); break;
                    }
                }
            }

            int evaluate(string& s) {
                stack<int> st;
                stack<char> op;
                bool may_be_unary = true;
                for (int i = 0; i < (int)s.size(); i++) {
                    if (delim(s[i]))
                        continue;

                    if (s[i] == '(') {
                        op.push('(');
                        may_be_unary = true;
                    } else if (s[i] == ')') {
                        while (op.top() != '(') {
                            process_op(st, op.top());
                            op.pop();
                        }
                        op.pop();
                        may_be_unary = false;
                    } else if (is_op(s[i])) {
                        char cur_op = s[i];
                        if (may_be_unary && is_unary(cur_op))
                            cur_op = -cur_op;
                        while (!op.empty() && (
                                (cur_op >= 0 && priority(op.top()) >= priority(cur_op)) ||
                                (cur_op < 0 && priority(op.top()) > priority(cur_op))
                            )) {
                            process_op(st, op.top());
                            op.pop();
                        }
                        op.push(cur_op);
                        may_be_unary = true;
                    } else {
                        int number = 0;
                        while (i < (int)s.size() && isalnum(s[i]))
                            number = number * 10 + s[i++] - '0';
                        --i;
                        st.push(number);
                        may_be_unary = false;
                    }
                }

                while (!op.empty()) {
                    process_op(st, op.top());
                    op.pop();
                }
                return st.top();
            }

########################
##########################  

COOL NOTES PART -4.69 ARRAY STUFF


    Intermediate Level:

    Implement an algorithm to rotate an NxN matrix by 90 degrees.
        Technique: Transpose and reverse or use of extra space.

    Implement an algorithm to find pairs in an array that sum up to a specific target.
        Technique: Two-pointer approach or hash table.

    Implement an algorithm to perform a cyclic rotation in a given array.
        Technique: Cyclic Sort.

    Find the "celebrity" in a party using a given array of people.
        Technique: Two-pointer approach.

    Implement an algorithm to find the contiguous subarray with the largest sum (Kadane's algorithm).
        Technique: Dynamic Programming.

    Implement an algorithm to merge two sorted arrays.
        Technique: Merge Sort or Two-pointer approach.

    Implement an algorithm to search for an element in a rotated sorted array.
        Technique: Modified Binary Search.

    Implement an algorithm to find the intersection of two arrays.
        Technique: Hashing or Two-pointer approach.

    Implement an algorithm to find the majority element in an array.
        Technique: Boyer-Moore Voting Algorithm.

    Implement an algorithm to find the longest increasing subsequence in an array.
        Technique: Dynamic Programming.
    Advanced Level:

    Implement an algorithm to find the first missing positive integer in an unsorted array.
        Technique: Cyclic Sort.

    Implement an algorithm to sort an array of 0s, 1s, and 2s (Dutch National Flag problem).
        Technique: Dutch National Flag algorithm or Two-pointer approach.

    Implement an algorithm to find the longest subarray with at most two distinct elements.
        Technique: Sliding Window.

    Implement an algorithm to find the median of two sorted arrays.
        Technique: Binary Search.

    Implement an algorithm to rotate an array to the right by k steps without using extra space.
        Technique: Reverse the array in parts.

    Implement an algorithm to find the shortest subarray with a sum at least K.
        Technique: Sliding Window.

    Implement an algorithm to find the equilibrium index of an array.
        Technique: Prefix Sum.

    Implement an algorithm to find the peak element in an array.
        Technique: Binary Search.

    Implement an algorithm to count the number of subarrays with a given sum.
        Technique: Prefix Sum.

    Implement an algorithm to find the contiguous subarray with equal numbers of 0s and 1s.
        Technique: Prefix Sum or Hashing.
    Expert Level:

    Implement an algorithm to efficiently find the maximum product of three numbers in an array.
        Technique: Sorting or finding max/min elements.

    Implement an algorithm to find the length of the longest subarray with sum divisible by k.
        Technique: Prefix Sum and Hashing.

    Implement an algorithm to find the minimum window in an array that contains all elements of another array.
        Technique: Sliding Window.

    Implement an algorithm to find the longest increasing subarray with absolute difference less than or equal to a given number.
        Technique: Sliding Window.

    Implement an algorithm to find the smallest subarray with sum greater than a given value.
        Technique: Sliding Window.

    Implement an algorithm to find the minimum number of platforms needed for a set of train arrivals and departures.
        Technique: Merge Intervals or Sorting.

    Implement an algorithm to find the maximum length subarray with sum less than or equal to a given sum.
        Technique: Sliding Window.

    Implement an algorithm to find the subarray with the least average.
        Technique: Sliding Window.

    Implement an algorithm to efficiently rotate an NxN matrix by 90 degrees in place.
        Technique: Transpose and reverse.

    Implement an algorithm to find the longest subarray with at most K distinct elements.
        Technique: Sliding Window or Hashing.
    These categorizations should provide you with an overview of the types of techniques commonly used for each question at different difficulty levels. Keep in mind that the optimal approach might vary based on specific constraints and requirements of the problem.


#######################################################
#######################################################
COOL NOTES PART -4: Graph Algorithms

    Bellman Ford:

        #Class to represent a graph 
        class Graph: 
        
            def __init__(self,vertices): 
                self.V= vertices #No. of vertices 
                self.graph = [] # default dictionary to store graph 

            # function to add an edge to graph 
            def addEdge(self,u,v,w): 
                self.graph.append([u, v, w]) 

            # The main function that finds shortest distances from src to 
            # all other vertices using Bellman-Ford algorithm.  The function 
            # also detects negative weight cycle 
            def BellmanFord(self, src): 
        
                # Step 1: Initialize distances from src to all other vertices 
                # as INFINITE 
                dist = [float("Inf")] * self.V 
                dist[src] = 0 
        
        
                # Step 2: Relax all edges |V| - 1 times. A simple shortest  
                # path from src to any other vertex can have at-most |V| - 1  
                # edges 
                for i in range(self.V - 1): 
                    # Update dist value and parent index of the adjacent vertices of 
                    # the picked vertex. Consider only those vertices which are still in 
                    # queue 
                    for u, v, w in self.graph: 
                        if dist[u] != float("Inf") and dist[u] + w < dist[v]: 
                                dist[v] = dist[u] + w 
        
                # Step 3: check for negative-weight cycles.  The above step  
                # guarantees shortest distances if graph doesn't contain  
                # negative weight cycle.  If we get a shorter path, then there 
                # is a cycle. 
        
                for u, v, w in self.graph: 
                        if dist[u] != float("Inf") and dist[u] + w < dist[v]: 
                                print "Graph contains negative weight cycle"
                                return
                                
                # print all distance 
                self.printArr(dist) 

    Floyd Warshall:
        We initialize the solution matrix same as 
        the input graph matrix as a first step. 
        Then we update the solution matrix by considering all 
        vertices as an intermediate vertex. 
        The idea is to one by one pick all vertices and updates all shortest 
        paths which include the picked vertex as an intermediate vertex in the 
        shortest path. When we pick vertex number k as an intermediate vertex, 
        we already have considered vertices {0, 1, 2, .. k-1} as intermediate vertices. 

        For every pair (i, j) of the source and destination 
        vertices respectively, there are two possible cases.

        1)  k is not an intermediate vertex in shortest path from i to j. 
            We keep the value of dist[i][j] as it is.

        2)  k is an intermediate vertex in shortest path from i to j. We update 
            the value of dist[i][j] as dist[i][k] + dist[k][j] if dist[i][j] > dist[i][k] + dist[k][j]


        # Solves all pair shortest path via Floyd Warshall Algorithm 
        def floydWarshall(graph): 
        
            """ dist[][] will be the output matrix that will finally 
                have the shortest distances between every pair of vertices """
            """ initializing the solution matrix same as input graph matrix 
            OR we can say that the initial values of shortest distances 
            are based on shortest paths considering no  
            intermediate vertices """

            dist = map(lambda i : map(lambda j : j , i) , graph) 
            
            """ Add all vertices one by one to the set of intermediate 
            vertices. 
            ---> Before start of an iteration, we have shortest distances 
            between all pairs of vertices such that the shortest 
            distances consider only the vertices in the set  
            {0, 1, 2, .. k-1} as intermediate vertices. 
            ----> After the end of a iteration, vertex no. k is 
            added to the set of intermediate vertices and the  
            set becomes {0, 1, 2, .. k} 
            """

            for k in range(V): 
        
                # pick all vertices as source one by one 
                for i in range(V): 
        
                    # Pick all vertices as destination for the 
                    # above picked source 
                    for j in range(V): 
        
                        # If vertex k is on the shortest path from  
                        # i to j, then update the value of dist[i][j] 
                        dist[i][j] = min(dist[i][j] , 
                                        dist[i][k]+ dist[k][j] 
                                        ) 
            printSolution(dist)
        
        graph = [[0,5,INF,10], 
                [INF,0,3,INF], 
                [INF, INF, 0,   1], 
                [INF, INF, INF, 0] 
            ] 

        # Print the solution 
        floydWarshall(graph); 
        Following matrix shows the shortest distances between every pair of vertices
        0      5      8      9
        INF      0      3      4
        INF    INF      0      1
        INF    INF    INF      0

    DJIKSTRA CLEAN:

        def dijkstra(G, s, w=None):

            def get_weight(u, v):
                return w[u, v] if w else 1

            dist = {s: 0}
            entries = {}
            pq = []
            for v in G[s]:
                d = get_weight(s, v)
                entry = [d, v, True]
                dist[v] = d
                entries[v] = entry
                heappush(pq, entry)

            while pq:
                u_dist, u, valid = heappop(pq)
                if valid:
                    for v in G[u]:
                        new_dist = u_dist + get_weight(u, v)
                        if not v in dist or new_dist < dist[v]:
                            dist[v] = new_dist
                            entry = [new_dist, v, True]
                            if v in entries:
                                entries[v][2] = False
                            entries[v] = entry
                            heappush(pq, entry)

            return dist

    DJIKSTRA CLEAN 2:
        https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/solutions/992958/python-most-readable-shortest-path-dijkstra/
        For a city A, find its shortest distance to every other city. then we know how many cites are within the threshold.
        To get the shortest distance, we can use bellman ford or dijkstra algorithm.
        But, our goal is to have a standarized template for leetcode problems. And dijkstra is better(you will know why after you practice 10 shortest path problems)

        class Solution:
            def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
                # 4:09pm 12/29/2020 as a pratice while preparing the graph templates
                
                # do dijkstra for every city, then count reachable cities within threshold
                
                
                graph = defaultdict(list)
                for src, dst, dist in edges:
                    graph[src].append((dst, dist))
                    graph[dst].append((src, dist))
                
                def dijkstra(city):
                    dist = [float('inf')]*n
                    processed = [False]*n
                    dist[city] = 0
                    q = [(0, city)]
                    
                    while q:
                        cur_dist, cur_city = heapq.heappop(q)
                        for next_city, dist_to_next_city in graph[cur_city]:
                            if not processed[next_city]:
                                new_dist = dist[cur_city] + dist_to_next_city
                                if new_dist < dist[next_city]:
                                    dist[next_city] = new_dist
                                    heapq.heappush(q, (new_dist, next_city))
                        processed[cur_city] = True
                        
                    return dist
                        
                ans = []
                for city in range(n):
                    distance = dijkstra(city)
                    ans.append((sum([d <= distanceThreshold for d in distance]),city))
                    
                return sorted(ans, key=lambda x: (x[0], -x[1]))[0][1]



    DJIKSTRA:

        from collections import defaultdict
        from heapq import *

        def dijkstra(edges, f, t):
            g = defaultdict(list)
            for l,r,c in edges:
                g[l].append((c,r))

            q, seen, mins = [(0,f,())], set(), {f: 0}
            while q:
                (cost,v1,path) = heappop(q)
                if v1 not in seen:
                    seen.add(v1)
                    path = (v1, path)
                    if v1 == t: return (cost, path)

                    for c, v2 in g.get(v1, ()):
                        if v2 in seen: continue
                        prev = mins.get(v2, None)
                        next = cost + c
                        if prev is None or next < prev:
                            mins[v2] = next
                            heappush(q, (next, v2, path))

            return float("inf")

        if __name__ == "__main__":
            edges = [
                ("A", "B", 7),
                ("A", "D", 5),
                ("B", "C", 8),
                ("B", "D", 9),
                ("B", "E", 7),
                ("C", "E", 5),
                ("D", "E", 15),
                ("D", "F", 6),
                ("E", "F", 8),
                ("E", "G", 9),
                ("F", "G", 11)
            ]

            print "=== Dijkstra ==="
            print edges
            print "A -> E:"
            print dijkstra(edges, "A", "E")
            print "F -> G:"
            print dijkstra(edges, "F", "G")

    LEETCODE DJIKSTRA QUESTION:



        743. Network Delay Time
        Medium
        Topics
        Companies
        Hint
        You are given a network of n nodes, labeled from 1 to n. You are also given times, 
        a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, 
        vi is the target node, and wi is the time it takes for a signal to travel from source to target.

        We will send a signal from a given node k. Return the minimum time it takes for all the n nodes 
        to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.

        

        Example 1:


        Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
        Output: 2
        Example 2:

        Input: times = [[1,2,1]], n = 2, k = 1
        Output: 1
        Example 3:

        Input: times = [[1,2,1]], n = 2, k = 2
        Output: -1


        Since the graph of network delay times is a weighted, connected graph (if the graph isn't 
        connected, we can return -1) with non-negative weights, we can find the shortest path from root 
        node K into any other node using Dijkstra's algorithm. If we want to find how long it will take 
        for all nodes to receive the signal, we need to find the maximum of the shortest paths from node K to any other node.

        We can do this by running dijkstra's algorithm starting with node K, and shortest path length 
        to node K, 0. We can keep track of the lengths of the shortest paths from K to every other node 
        in a set S, and if the length of S is equal to N, we know that the graph is connected (if not, return -1). 
        We can then return the maximum of the shortest path lengths in S to get how long it will take for all nodes to receive the signal.

        def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
                graph = collections.defaultdict(list)
                for (u, v, w) in times:
                    graph[u].append((v, w))
                    
                priority_queue = [(0, K)]
                shortest_path = {}
                while priority_queue:
                    w, v = heapq.heappop(priority_queue)
                    if v not in shortest_path:
                        shortest_path[v] = w
                        for v_i, w_i in graph[v]:
                            heapq.heappush(priority_queue, (w + w_i, v_i))
                            
                if len(shortest_path) == N:
                    return max(shortest_path.values())
                else:
                    return -1


#############################################################################
#############################################################################
COOL NOTES PART -2: HOW TO USE HEAP DICTIONARIES WITH DECREASE KEY USING HEAPQ!

        -> Sort stability: how do you get two tasks with equal priorities 
        to be returned in the order they were originally added?

        -> In the future with Python 3, tuple comparison breaks for (priority, task) 
        pairs if the priorities are equal and the tasks do not have a default comparison order.

        -> If the priority of a task changes, how do you 
        move it to a new position in the heap?

        -> Or if a pending task needs to be deleted, 
        how do you find it and remove it from the queue?

        A solution to the first two challenges is to store entries as 3-element list 
        including the priority, an entry count, and the task. The entry count serves 
        as a tie-breaker so that two tasks with the same priority are returned 
        in the order they were added. And since no two entry counts are the same, 
        the tuple comparison will never attempt to directly compare two tasks.

        The remaining challenges revolve around finding a pending task and 
        making changes to its priority or removing it entirely. 
        Finding a task can be done with a dictionary pointing to an entry in the queue.

        Removing the entry or changing its priority is more difficult because 
        it would break the heap structure invariants. So, a possible solution 
        is to mark the existing entry as 
        Removed and add a new entry with the revised priority:

        pq = []                         # list of entries arranged in a heap
        entry_finder = {}               # mapping of tasks to entries
        REMOVED = '<removed-task>'      # placeholder for a removed task
        counter = itertools.count()     # unique sequence count

        def add_task(task, priority=0):
            'Add a new task or update the priority of an existing task'
            if task in entry_finder:
                remove_task(task)
            count = next(counter)
            entry = [priority, count, task]
            entry_finder[task] = entry
            heappush(pq, entry)

        def remove_task(task):
            'Mark an existing task as REMOVED.  Raise KeyError if not found.'
            entry = entry_finder.pop(task)
            entry[-1] = REMOVED

        def pop_task():
            'Remove and return the lowest priority task. Raise KeyError if empty.'
            while pq:
                priority, count, task = heappop(pq)
                if task is not REMOVED:
                    del entry_finder[task]
                    return task
            raise KeyError('pop from an empty priority queue')



#################################################################################3
#######################################################################################
#######################################################################################
COOL NOTES PART -1: SORTING, SEARCHING, Quick selecting
  
    BINARY SEARCH and TERNARY SEARCH
        #Recursive
        def binarySearch (arr, l, r, x): 
        
            # Check base case 
            if r >= l: 
        
                mid = l + (r - l)/2
        
                # If element is present at the middle itself 
                if arr[mid] == x: 
                    return mid 
                
                # If element is smaller than mid, then it  
                # can only be present in left subarray 
                elif arr[mid] > x: 
                    return binarySearch(arr, l, mid-1, x) 
        
                # Else the element can only be present  
                # in right subarray 
                else: 
                    return binarySearch(arr, mid + 1, r, x) 
        
            else: 
                # Element is not present in the array 
                return -1
                
        # Iterative Binary Search Function 
        # It returns location of x in given array arr if present, 
        # else returns -1 
        def binarySearch(arr, l, r, x): 
        
            while l <= r: 
        
                mid = l + (r - l)/2; 
                
                # Check if x is present at mid 
                if arr[mid] == x: 
                    return mid 
        
                # If x is greater, ignore left half 
                elif arr[mid] < x: 
                    l = mid + 1
        
                # If x is smaller, ignore right half 
                else: 
                    r = mid - 1
            
            # If we reach here, then the element was not present 
            return -1

        # RECURSIVE Ternary Search 
        def ternarySearch(l, r, key, ar): 
        
            if (r >= l): 
        
                # Find the mid1 and mid2 
                mid1 = l + (r - l) //3
                mid2 = r - (r - l) //3
        
                # Check if key is present at any mid 
                if (ar[mid1] == key):  
                    return mid1 
                
                if (ar[mid2] == key):  
                    return mid2 
                
                # Since key is not present at mid, 
                # check in which region it is present 
                # then repeat the Search operation 
                # in that region 
                if (key < ar[mid1]):  
        
                    # The key lies in between l and mid1 
                    return ternarySearch(l, mid1 - 1, key, ar) 
                
                elif (key > ar[mid2]):  
        
                    # The key lies in between mid2 and r 
                    return ternarySearch(mid2 + 1, r, key, ar) 
                
                else:  
        
                    # The key lies in between mid1 and mid2 
                    return ternarySearch(mid1 + 1,  
                                        mid2 - 1, key, ar) 
                
            # Key not found 
            return -1

        # ITERATIVE Ternary Search 
        def ternarySearch(l, r, key, ar): 
            while r >= l: 
                
                # Find mid1 and mid2 
                mid1 = l + (r-l) // 3
                mid2 = r - (r-l) // 3
        
                # Check if key is at any mid 
                if key == ar[mid1]: 
                    return mid1 
                if key == mid2: 
                    return mid2 
        
                # Since key is not present at mid,  
                # Check in which region it is present 
                # Then repeat the search operation in that region 
                if key < ar[mid1]: 
                    # key lies between l and mid1 
                    r = mid1 - 1
                elif key > ar[mid2]: 
                    # key lies between mid2 and r 
                    l = mid2 + 1
                else: 
                    # key lies between mid1 and mid2 
                    l = mid1 + 1
                    r = mid2 - 1
        
            # key not found 
            return -1

     Counting sort is following:
        def counting_sort(array, maxval):
            """in-place counting sort"""
            n = len(array)
            m = maxval + 1
            count = [0] * m               # init with zeros
            for a in array:
                count[a] += 1             # count occurences
            i = 0
            for a in range(m):            # emit
                for c in range(count[a]): # - emit 'count[a]' copies of 'a'
                    array[i] = a
                    i += 1
            return array

        print(counting_sort( [1, 4, 7, 2, 1, 3, 2, 1, 4, 2, 3, 2, 1], 7 ))

        Another version:

        def countSort(arr): 
        
            # The output character array that will have sorted arr 
            output = [0 for i in range(256)] 
        
            # Create a count array to store count of inidividul 
            # characters and initialize count array as 0 
            count = [0 for i in range(256)] 
        
            # For storing the resulting answer since the  
            # string is immutable 
            ans = ["" for _ in arr] 
        
            # Store count of each character 
            for i in arr: 
                count[ord(i)] += 1
        
            # Change count[i] so that count[i] now contains actual 
            # position of this character in output array 
            for i in range(256): 
                count[i] += count[i-1] 
        
            # Build the output character array 
            for i in range(len(arr)): 
                output[count[ord(arr[i])]-1] = arr[i] 
                count[ord(arr[i])] -= 1
        
            # Copy the output array to arr, so that arr now 
            # contains sorted characters 
            for i in range(len(arr)): 
                ans[i] = output[i] 
            return ans  
        
        # Driver program to test above function 
        arr = "geeksforgeeks"
        ans = countSort(arr) 
        print "Sorted character array is %s"  %("".join(ans)) 


    Radix sort is following: 
        def countingSort(arr, exp1): 
        
            n = len(arr) 
        
            # The output array elements that will have sorted arr 
            output = [0] * (n) 
        
            # initialize count array as 0 
            count = [0] * (10) 
        
            # Store count of occurrences in count[] 
            for i in range(0, n): 
                index = (arr[i]/exp1) 
                count[ (index)%10 ] += 1
        
            # Change count[i] so that count[i] now contains actual 
            #  position of this digit in output array 
            for i in range(1,10): 
                count[i] += count[i-1] 
        
            # Build the output array 
            i = n-1
            while i>=0: 
                index = (arr[i]/exp1) 
                output[ count[ (index)%10 ] - 1] = arr[i] 
                count[ (index)%10 ] -= 1
                i -= 1
        
            # Copying the output array to arr[], 
            # so that arr now contains sorted numbers 
            i = 0
            for i in range(0,len(arr)): 
                arr[i] = output[i] 
        
        # Method to do Radix Sort 
        def radixSort(arr): 
        
            # Find the maximum number to know number of digits 
            max1 = max(arr) 
        
            # Do counting sort for every digit. Note that instead 
            # of passing digit number, exp is passed. exp is 10^i 
            # where i is current digit number 
            exp = 1
            while max1/exp > 0: 
                countingSort(arr,exp) 
                exp *= 10
        
        # Driver code to test above 
        arr = [ 170, 45, 75, 90, 802, 24, 2, 66] 
        radixSort(arr) 
        
        for i in range(len(arr)): 
            print(arr[i]), 


    Another Radix Sort

        def counting_sort_for_radix(arr, exp):
            d = {}
            for num in range(0,10):
                d[num] = 0
            for num in arr:
                d[(num / exp) % 10] += 1
            for num in range(1, 10):
                d[num] = d[num - 1] + d[num]
            output = [0 for n in arr]
            for i in range(len(arr) - 1, -1, -1):
                num = arr[i]
                output[d[(num / exp) % 10] - 1] = num
                d[(num / exp) % 10] -= 1
            for i in range(len(arr)):
                arr[i] = output[i]
        
        def radix_sort(arr):
            if not arr:
                return []
            max_num = max(arr)
            exp = 1
            while max_num/exp > 0:
                counting_sort_for_radix(arr, exp)
                exp *= 10
            return arr

    ITERATIVE inplace O(1) QUICKSORT EXAMPLE 1: 

        def quick_sort(arr):
            # This quicksort can easily be modified so that it happens
            # in place and no extra space is used.
        
            # If the array is empty, it is already sorted.
            if not arr:
                return []
        
            # Otherwise, sort from indices 0 to (len(arr) - 1) inclusive.
            else:
                return quick_sort_iterative(arr, 0, len(arr) - 1)
        
        def quick_sort_iterative(arr, left, right):
            # We do iterative quick sort with the use of a stack.
            stack = []
            stack.append(left)
            stack.append(right)
            # Continue while we still have an unsorted portion.
            while stack:
                r = stack.pop()
                l = stack.pop()
                # Move everything less than pivot left of it, and
                # everything greater to the right of it.
                pivot = quick_sort_partition(arr, l, r)
                # If there is stuff left of it that needs to be
                # sorted
                if pivot - 1 > l:
                    stack.append(l)
                    stack.append(pivot - 1)
                # If there is stuff right of it that needs to be
                # sorted
                if pivot + 1 < r:
                    stack.append(pivot + 1)
                    stack.append(r)
            return arr
        
        def quick_sort_partition(arr, left, right):
            # Right now we are implementing randomized quicksort,
            # but we could just as easily skip the first two lines
            # and just always use the first element in the list as a
            # pivot.
            random_index = random.choice(range(left, right + 1))
            arr[left], arr[random_index] = arr[random_index], arr[left]
            pivot = arr[left]
            orig_left = left
            left += 1
            # Do the switching that is required.
            while True:
                while left <= right and arr[left] <= pivot:             
                    left += 1         
                
                while right >= left and arr[right] >= pivot:
                    right -= 1
                
                if right <= left:
                    break
                arr[left], arr[right] = arr[right], arr[left]
            arr[orig_left], arr[right] = arr[right], arr[orig_left]
            return right

        

    INPLACE QUICKSORT 2:

        def sub_partition(array, start, end, idx_pivot):

            'returns the position where the pivot winds up'
            if not (start <= idx_pivot <= end):
                raise ValueError('idx pivot must be between start and end')

            array[start], array[idx_pivot] = array[idx_pivot], array[start]
            pivot = array[start]
            i = start + 1
            j = start + 1

            # i refers to first big element, j refers to last big element. 
            while j <= end:
                if array[j] <= pivot:
                    array[j], array[i] = array[i], array[j]
                    i += 1
                j += 1

            # it swaps with i-1 because i-1 is the end of the smaller elements, 
            # and i refers to first big element
            array[start], array[i - 1] = array[i - 1], array[start]
            return i - 1

        def quicksort(array, start=0, end=None):

            if end is None:
                end = len(array) - 1

            if end - start < 1:
                return

            idx_pivot = random.randint(start, end)
            i = sub_partition(array, start, end, idx_pivot)
            #print array, i, idx_pivot
            quicksort(array, start, i - 1)
            quicksort(array, i + 1, end)

    INPLACE MERGE SORT O(N^2) example

        '''
        Now, merge sort on the other hand is pretty hard to do in place. 
        I’ll leave it for future Nishanth to explore it 
        further, but here’s a simple O(n^2) sort .
        '''
        def merge_sort(xs):
            """Inplace merge sort of array without recursive. The basic idea
            is to avoid the recursive call while using iterative solution. 
            The algorithm first merge chunk of length of 2, then merge chunks
            of length 4, then 8, 16, .... , until 2^k where 2^k is large than 
            the length of the array
            """
            
            unit = 1
            while unit <= len(xs):
                h = 0
                for h in range(0, len(xs), unit * 2):
                    l, r = h, min(len(xs), h + 2 * unit)
                    mid = h + unit
                    # merge xs[h:h + 2 * unit]
                    p, q = l, mid
                    while p < mid and q < r:
                        # use <= for stable merge merge
                        if xs[p] <= xs[q]: p += 1
                        else:
                            tmp = xs[q]
                            xs[p + 1: q + 1] = xs[p:q]
                            xs[p] = tmp
                            p, mid, q = p + 1, mid + 1, q + 1

                unit *= 2
            
            return xs
    



    INPLACE MERGE SORT 2:

        // Merges two subarrays of arr[]. 
        // First subarray is arr[l..m] 
        // Second subarray is arr[m+1..r] 
        // Inplace Implementation 
        void merge(int arr[], int start, int mid, int end) 
        { 
            int start2 = mid + 1; 
        
            // If the direct merge is already sorted 
            if (arr[mid] <= arr[start2]) { 
                return; 
            } 
        
            // Two pointers to maintain start 
            // of both arrays to merge 
            while (start <= mid && start2 <= end) { 
        
                // If element 1 is in right place 
                if (arr[start] <= arr[start2]) { 
                    start++; 
                } 
                else { 
                    int value = arr[start2]; 
                    int index = start2; 
        
                    // Shift all the elements between element 1 
                    // element 2, right by 1. 
                    while (index != start) { 
                        arr[index] = arr[index - 1]; 
                        index--; 
                    } 
                    arr[start] = value; 
        
                    // Update all the pointers 
                    start++; 
                    mid++; 
                    start2++; 
                } 
            } 
        } 
        
        /* l is for left index and r is right index of the  
        sub-array of arr to be sorted */
        void mergeSort(int arr[], int l, int r) 
        { 
            if (l < r) { 
        
                // Same as (l + r) / 2, but avoids overflow 
                // for large l and r 
                int m = l + (r - l) / 2; 
        
                // Sort first and second halves 
                mergeSort(arr, l, m); 
                mergeSort(arr, m + 1, r); 
        
                merge(arr, l, m, r); 
            } 
        } 



    BUCKET SORT:

        Bucket sort is mainly useful when input is uniformly distributed over a range. 
        For example, consider the following problem. 
        Sort a large set of floating point numbers which are in range from 0.0 to 1.0 
        and are uniformly distributed across the range. 
        How do we sort the numbers efficiently?

        Counting sort can not be applied here as we use keys as index in counting sort. 
        Here keys are floating point numbers. 
        The idea is to use bucket sort. 
        Following is bucket algorithm.

        bucketSort(arr[], n)
        1) Create n empty buckets (Or lists).
        2) Do following for every array element arr[i].
        .......a) Insert arr[i] into bucket[n*array[i]]
        3) Sort individual buckets using insertion sort.
        4) Concatenate all sorted buckets.

        # Python3 program to sort an array  
        # using bucket sort  
        def insertionSort(b): 
            for i in range(1, len(b)): 
                up = b[i] 
                j = i - 1
                while j >=0 and b[j] > up:  
                    b[j + 1] = b[j] 
                    j -= 1
                b[j + 1] = up     
            return b       
                    
        def bucketSort(x): 
            arr = [] 
            slot_num = 10 # 10 means 10 slots, each 
                        # slot's size is 0.1 
            for i in range(slot_num): 
                arr.append([]) 
                
            # Put array elements in different buckets  
            for j in x: 
                index_b = int(slot_num * j)  
                arr[index_b].append(j) 
            
            # Sort individual buckets  
            for i in range(slot_num): 
                arr[i] = insertionSort(arr[i]) 
                
            # concatenate the result 
            k = 0
            for i in range(slot_num): 
                for j in range(len(arr[i])): 
                    x[k] = arr[i][j] 
                    k += 1
            return x 

    QUICK SELECT IMPLEMENTATION:

        # Python3 program of Quick Select 
    
        # Standard partition process of QuickSort().  
        # It considers the last element as pivot  
        # and moves all smaller element to left of  
        # it and greater elements to right 
        def partition(arr, l, r): 
            
            x = arr[r] 
            i = l 
            for j in range(l, r): 
                
                if arr[j] <= x: 
                    arr[i], arr[j] = arr[j], arr[i] 
                    i += 1
                    
            arr[i], arr[r] = arr[r], arr[i] 
            return i 
        
        # finds the kth position (of the sorted array)  
        # in a given unsorted array i.e this function  
        # can be used to find both kth largest and  
        # kth smallest element in the array.  
        # ASSUMPTION: all elements in arr[] are distinct 
        def kthSmallest(arr, l, r, k): 
        
            # if k is smaller than number of 
            # elements in array 
            if (k > 0 and k <= r - l + 1): 
        
                # Partition the array around last 
                # element and get position of pivot 
                # element in sorted array 
                index = partition(arr, l, r) 
        
                # if position is same as k 
                if (index - l == k - 1): 
                    return arr[index] 
        
                # If position is more, recur  
                # for left subarray  
                if (index - l > k - 1): 
                    return kthSmallest(arr, l, index - 1, k) 
        
                # Else recur for right subarray  
                return kthSmallest(arr, index + 1, r,  
                                    k - index + l - 1) 
            return INT_MAX 
        
        # Driver Code 
        arr = [ 10, 4, 5, 8, 6, 11, 26 ] 
        n = len(arr) 
        k = 3
        print("K-th smallest element is ", end = "") 
        print(kthSmallest(arr, 0, n - 1, k)) 
        
        # This code is contributed by Muskan Kalra. 


#################################################################################
#################################################################################
GRAPH TRAVERSAL ALL TYPES:


    Morris Traversal
        Have you ever wanted to do In-order traversal in O(1) space? 
        So have I! Too bad it doesn’t exist…..

        def find_predecessor(node):
            tmp = node.left
            while tmp.right and tmp.right != node:
                tmp = tmp.right
            return tmp
        
        def morris_traversal(root):
            ans = []
            curr = root
            while curr:
                if not curr.left:
                    ans.append(curr)
                    curr = curr.right
                else:
                    pred = find_predecessor(curr)
                    if pred.right == None:
                        pred.right = curr
                        curr = curr.left
                    else:
                        pred.right = None
                        ans.append(current)
                        current = current.right
            return ans

        The find_predecessor function finds the inorder predecessor of a node. Then, 
        in our traversal, we use the right pointers to store where we need 
        to get back to at all times. This way, we don’t need a stack nor recursion.


    Binary Tree Traversal
    Morris Traversal:
    Covered in another post, scroll to that post to see explanation and implementation.

    Inorder Traversal Recursive:
        def in_order_recursive(root):
            if root == None:
                return
            post_order_recursive(root.left)
            print root.data
            post_order_recursive(root.right)

    Inorder Traversal Iterative:
        def inorder_traversal(root):
            if root == None:
                return []
            current = root
            done = False
            ans = []
            s = []
            while not done:
                if current:
                    s.append(current)
                    current = current.left
                else:
                    if len(stack) == 0:
                        done = True
                    else:
                        current = stack.pop()
                        ans.append(current.val)
                        current = current.right
            return ans

    Postorder Traversal Recursive:
        def post_order_recursive(root):
            if root == None:
                return
            post_order_recursive(root.left)
            post_order_recursive(root.right)
            print root.data

    Postorder Traversal Iterative:
        from collections import deque
        def post_order_traversal(root):
            if root == None:
                return []
            s1 = []
            ans = deque()
            s1.append(root)
            while len(s1) > 0:
                current = s1.pop()
                ans.appendleft(current.val)
                if current.left:
                    s1.append(current.left)
                if current.right:
                    s1.append(current.right)
            return ans

    Preorder Traversal Recursive:
        def pre_order_recursive(root):
            if root == None:
                return
            print root.data
            pre_order_recursive(root.left)
            pre_order_recursive(root.right)

    Preorder Traversal Iterative:
        def pre_order_traversal(root):
            if root == None:
                return []
            s = []
            ans = []
            s.append(root)
            while len(s) > 0:
                current = s.pop()
                ans.append(current.val)
                if current.right:
                    s.append(current.right)
                if current.left:
                    s.append(current.left)
            return ans

    Level Order Traversal Iterative:
    def level_order_traversal(root):
        if root == None:
            return []
        q = deque()
        ans = []
        level = []
        curr_count = 1
        next_count = 0
        while len(q) > 0:
            current = q.popleft()
            level.append(current.val)
            curr_count -= 1
            if current.left:
                    q.append(current.left)
                    next_count += 1
            if current.right:
                    q.append(current.right)
                    next_count += 1
            if curr_count == 0:
                ans.append(level)
                level = []
                curr_count = next_count
                next_count = 0
        return ans

    Normal Graph Traversal
    DFS Traversal
        def DFS(node):
            s = stack()
            s.append(node)
            visited = {}
            while len(s) > 0:
                current = s.pop()
                if current not in visited:
                    visited[current] = 1
                    print current
                    for neighbor in current.neighbors:
                        if neighbor not in visited:
                            s.append(neighbor)
    BFS Traversal
        def BFS(node):
            q = deque()
            q.append(node)
            visited = {}
            while len(q) > 0:
                current = q.popleft()
                visited[current] = 1
                for neighbor in current.neighbors:
                    if neighbor not in visited:
                        q.append(neighbor)



    Topological Sort

        In DFS, we start from a vertex, we first print it and then recursively 
        call DFS for its adjacent vertices. In topological sorting, we use a 
        temporary stack. We don’t print the vertex immediately, we first recursively 
        call topological sorting for all its adjacent vertices, then push it to a stack.
        Finally, print contents of stack. Note that a vertex is pushed to stack only 
        when all of its adjacent vertices (and their adjacent vertices and so on) are already in stack.

        def recursive_topological_sort(graph, node):
            result = []
            seen = set()

            def recursive_helper(node):
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        recursive_helper(neighbor)
                result.insert(0, node)              # this line replaces the result.append line

            recursive_helper(node)
            return result

        def iterative_topological_sort(graph, start):
            seen = set()
            stack = []    # path variable is gone, stack and order are new
            order = []    # order will be in reverse order at first
            q = [start]
            while q:
                v = q.pop()
                if v not in seen:
                    seen.add(v) # no need to append to path any more
                    q.extend(graph[v])

                    while stack and v not in graph[stack[-1]]: # new stuff here!
                        order.append(stack.pop())
                    stack.append(v)

            return stack + order[::-1]   # new return value!

        That's it! One line gets removed and a similar one gets added at a different location. 
        If you care about performance, you should probably do result.append in the second
        helper function too, and do return result[::-1] in the top level 
        recursive_topological_sort function. But using insert(0, ...) is a more minimal change.

        Its also worth noting that if you want a topological order of the whole graph, 
        you shouldn't need to specify a starting node. Indeed, there may not be a single
        node that lets you traverse the entire graph, so you may need to do several 
        traversals to get to everything. An easy way to make that happen in the 
        iterative topological sort is to initialize q to list(graph) 
        (a list of all the graph's keys) instead of a list with only a single starting node. 
        For the recursive version, replace the call to recursive_helper(node) with a loop 
        that calls the helper function on every node in the graph if it's not yet in seen.




###########################################################
########################################################

COOL NOTES PART 6: UNION FIND PYTHON RECIPEE

        """
        MakeSet(x) initializes disjoint set for object x
        Find(x) returns representative object of the set containing x
        Union(x,y) makes two sets containing x and y respectively into one set

        Some Applications:
        - Kruskal's algorithm for finding minimal spanning trees
        - Finding connected components in graphs
        - Finding connected components in images (binary)
        """

        def MakeSet(x):
            x.parent = x
            x.rank   = 0

        def Union(x, y):
            xRoot = Find(x)
            yRoot = Find(y)
            if xRoot.rank > yRoot.rank:
                yRoot.parent = xRoot
            elif xRoot.rank < yRoot.rank:
                xRoot.parent = yRoot
            elif xRoot != yRoot: # Unless x and y are already in same set, merge them
                yRoot.parent = xRoot
                xRoot.rank = xRoot.rank + 1

        def Find(x):
            if x.parent == x:
                return x
            else:
                x.parent = Find(x.parent)
                return x.parent

        """"""""""""""""""""""""""""""""""""""""""
        # sample code using Union-Find (not needed)

        import itertools

        class Node:
            def __init__ (self, label):
                self.label = label
            def __str__(self):
                return self.label
            
        l = [Node(ch) for ch in "abcdefg"]      #list of seven objects with distinct labels
        print ""
        print "objects labels:\t\t\t", [str(i) for i in l]

        [MakeSet(node) for node in l]       #starting with every object in its own set

        sets =  [str(Find(x)) for x in l]
        print "set representatives:\t\t", sets
        print "number of disjoint sets:\t", len([i for i in itertools.groupby(sets)])

        assert( Find(l[0]) != Find(l[2]) )
        Union(l[0],l[2])        #joining first and third
        assert( Find(l[0]) == Find(l[2]) )

        assert( Find(l[0]) != Find(l[1]) )
        assert( Find(l[2]) != Find(l[1]) )
        Union(l[0],l[1])        #joining first and second
        assert( Find(l[0]) == Find(l[1]) )
        assert( Find(l[2]) == Find(l[1]) )

        Union(l[-2],l[-1])        #joining last two sets
        Union(l[-3],l[-1])        #joining last two sets

        sets = [str(Find(x)) for x in l]
        print "set representatives:\t\t", sets
        print "number of disjoint sets:\t", len([i for i in itertools.groupby(sets)])

        for o in l:
            del o.parent

        ---------------------------------------------


        # A class to represent a disjoint set
        class DisjointSet:
            parent = {}

            # stores the depth of trees
            rank = {}

            # perform MakeSet operation
            def makeSet(self, universe):
                # create `n` disjoint sets (one for each item)
                for i in universe:
                    self.parent[i] = i
                    self.rank[i] = 0

            # Find the root of the set in which element `k` belongs
            def Find(self, k):
                # if `k` is not the root
                if self.parent[k] != k:
                    # path compression
                    self.parent[k] = self.Find(self.parent[k])
                return self.parent[k]

            # Perform Union of two subsets
            def Union(self, a, b):
                # find the root of the sets in which elements `x` and `y` belongs
                x = self.Find(a)
                y = self.Find(b)

                # if `x` and `y` are present in the same set
                if x == y:
                    return

                # Always attach a smaller depth tree under the root of the deeper tree.
                if self.rank[x] > self.rank[y]:
                    self.parent[y] = x
                elif self.rank[x] < self.rank[y]:
                    self.parent[x] = y
                else:
                    self.parent[x] = y
                    self.rank[y] = self.rank[y] + 1


        def printSets(universe, ds):
            print([ds.Find(i) for i in universe])


        if __name__ == '__main__':

            # universe of items
            universe = [1, 2, 3, 4, 5]

            # initialize `DisjointSet` class
            ds = DisjointSet()

            # create a singleton set for each element of the universe
            ds.makeSet(universe)
            printSets(universe, ds)

            ds.Union(4, 3)        # 4 and 3 are in the same set
            printSets(universe, ds)

            ds.Union(2, 1)        # 1 and 2 are in the same set
            printSets(universe, ds)

            ds.Union(1, 3)        # 1, 2, 3, 4 are in the same set
            printSets(universe, ds)

        Download  Run Code

        Output:

        [1, 2, 3, 4, 5]
        [1, 2, 3, 3, 5]
        [1, 1, 3, 3, 5]
        [3, 3, 3, 3, 5]

###############################################################
################################################################3
COOL NOTES PART 6.3
Useful bit manipulation:

Bit Manipulation - Useful Tricks for efficient coding.

Create a number that has only set bit as k-th bit --> 1 << (k-1)
Check whether k-th bit is set or not -->
if (n & (1 << (k - 1)))
    cout << "SET";

Set k-th bit to 1 --> n | (1 << (k - 1))
Clearing the k-th bit --> n & ~(1 << (k - 1))
Toggling the k-th bit --> n ^ (1 << (k – 1))

Check whether n is power of 2 or not
    if(x && (!( x&(x-1) ))
        cout<<"Power of 2";


(x<<y) is equivalent to multiplying x with 2^y (2 raised to power y).
(x>>y) is equivalent to dividing x with 2^y.

Swapping two numbers
x = x ^ y
y = x ^ y
x = x ^ y

Average of two numbers --> (x+y) >> 1

Convert character ch from Upper to Lower case --> ch = ch | ' '
Convert character ch from Lower to Upper case -->ch = ch & '_'

Check if n is odd or even -->
if(n & 1)
cout<<"odd"
else
cout<<"even";

Bitwise operations are very useful as they mostly operate in O(1) time.
Please upvote if its helpful and suggestions are welcome.



###############################################################
################################################################3
COOL NOTES PART 6.5
    BITMASK OPERATIONS FOR BITMASP DP AND HANDLING SETS. 

    1. Representation. You know that shit cause you in CS. 

    2. To multiply/divide an integer by 2: 
        We only need to shift the bits in the integer left/right, respectively.
        Notice that the truncation in the shift right operation automatically rounds the division-by-2 down,
        e.g. 17/2  = 8.

        For example:        A = 34 (base 10)                  = 100010 (base 2)
                            A = A << 1 = A * 2 = 68 (base 10) = 1000100 (base 2)
                            A = A >> 2 = A / 4 = 17 (base 10) = 10001 (base 2)
                            A = A >> 1 = A / 2 = 8 (base 10) = 1000 (base 2) <- LSB( Least Significant Bit )is gone

    3. Add the jth object to the subset (set the jth bit from 0 to 1):
        use the bitwise OR operation A |= (1 << j).

        For example:     A = 34 (base 10) = 100010 (base 2)
                        j = 3, 1 << j    = 001000 <- bit ‘1’ is shifted to the left 3 times
                                            -------- OR (true if either of the bits is true)
                        A = 42 (base 10) = 101010 (base 2) // update A to this new value 42

    4. Remove the jth object from the subset (set the jth bit from 1 to 0):
        use the bitwise AND operation A &= ∼(1 << j).

        For example:         A = 42 (base 10) = 101010 (base 2)
                            j = 1, ~(1 << j) = 111101 <- ‘~’ is the bitwise NOT operation
                                                -------- AND
                            A = 40 (base 10) = 101000 (base 2) // update A to this new value 40

    5. Check whether the jth object is in the subset (check whether jth bit is 1):
        use the bitwise AND operation T = A & (1 << j).
        If T = 0, then the j-th item of the set is off.
        If T != 0 (to be precise, T = (1 << j)), then the j-th item of the set is on.

    For example:    A = 42 (base 10) = 101010 (base 2)
                    j = 3, 1 << j    = 001000 <- bit ‘1’ is shifted to the left 3 times
                                        -------- AND (only true if both bits are true)
                    T = 8 (base 10)  = 001000 (base 2) -> not zero, the 3rd item is on

    6. To toggle (flip the status of) the j-th item of the set:
    use the bitwise XOR operation A ∧ = (1 << j).

    For example:       A = 40 (base 10) = 101000 (base 2)
                        j = 2, (1 << j)  = 000100 <- bit ‘1’ is shifted to the left 2 times
                                            -------- XOR <- true if both bits are different
                        A = 44 (base 10) = 101100 (base 2) // update A to this new value 44

    7. To get the value of the least significant bit that is on (first from the right):
    use T = (A & (-A)).

    For example:     A =  40 (base 10) = 000...000101000 (32 bits, base 2)
                    -A = -40 (base 10) = 111...111011000 (two’s complement)
                                        ----------------- AND
                        T =   8 (base 10) = 000...000001000 (3rd bit from right is on)

    8. To turn on all bits in a set of size n: (be careful with overflows)
    use A = (1 << n) - 1 ;

    9. Iterate through all subsets of a set of size n:
            for ( x = 0; x < (1 << n); ++x )  

    10. Iterate through all subsets of a subset y (not including empty set):
                for ( x = y; x > 0; x = ( y & (x-1) ) )
    Example of a subset problem: given a set of numbers, we want to find the sum of all subsets.

    Sol: This is easy to code using bitmasks. we can use an array to store all the results.

    int sum_of_all_subset ( vector< int > s ){
                int n = s.size() ;
                int results[ ( 1 << n ) ] ;     // ( 1 << n )= 2^n

            //initialize results to 0
                memset( results, 0, sizeof( results ) ) ;

            // iterate through all subsets

            for( int i = 0 ; i < ( 1 << n ) ; ++ i ) {    // for each subset, O(2^n)
                    for ( int j = 0; j < n ; ++ j ) {       // check membership, O(n)
                        i f ( ( i & ( 1 << j ) ) ! = 0 )    // test if bit ‘j’ is turned on in subset ‘i’?
                            results[i] += s [j] ;          // if yes, process ‘j’
                        }
                    }
            }

    11. LIMITATIONS:
        a. Always check the size of the set to determine whether to use an int or long long or not using bitmask at all
        b. Always use parenthesis to indicate the precedence of operations when doing bitwise operations!
            When it involves bitwise operators and not putting parenthesis can yield undesirable results!

            For example, let x = 5. Then x - 1 << 2 = 16, but x - (1 << 2) = 1




################################################################33
#################################################################33
COOL NOTES PART 7: 

    SUBMASK ENUMERATION:

        Submask Enumeration
        
        Given a bitmask m, you want to efficiently iterate through all of its submasks, 
        that is, masks s in which only bits that were included in mask m are set.

        Consider the implementation of this algorithm, based on tricks with bit operations:

        int s = m;
        while (s > 0) {
        ... you can use s ...
        s = (s-1) & m;
        }

        or, using a more compact for statement:

        for (int s=m; s; s=(s-1)&m)
        ... you can use s ...

        In both variants of the code, the submask equal to zero will not 
        be processed. We can either process it outside the loop, 
        or use a less elegant design, for example:

        for (int s=m; ; s=(s-1)&m) {
        ... you can use s ...
        if (s==0)  break;
        }

        Let us examine why the above code visits all submasks of m, without repetition, and in descending order.

        Suppose we have a current bitmask s, and we want to move on to the next bitmask. 
        By subtracting from the mask s one unit, we will remove the rightmost set bit 
        and all bits to the right of it will become 1. Then we remove all the "extra" 
        one bits that are not included in the mask m and therefore can't be a part of a 
        submask. We do this removal by using the bitwise operation (s-1) & m. As a result, 
        we "cut" mask s−1 to determine the highest value that it can take, 
        that is, the next submask after s in descending order.

        Thus, this algorithm generates all submasks of this mask in descending order,
        performing only two operations per iteration.

        A special case is when s=0. After executing s−1 we get a mask where all bits are 
        set (bit representation of -1), and after (s-1) & m we will have that s will be equal to m. 
        Therefore, with the mask s=0 be careful — if the loop does not end at zero, 
        the algorithm may enter an infinite loop.

    ITERATING THROUGH MASKS AND THEIR SUBMASKS:


        Iterating through all masks with their submasks. Complexity O(3n)
        In many problems, especially those that use bitmask dynamic programming, 
        you want to iterate through all bitmasks and for each mask, iterate through all of its submasks:

        for (int m=0; m<(1<<n); ++m)
            for (int s=m; s; s=(s-1)&m)
        ... s and m ...


        Let's prove that the inner loop will execute a total of O(3n) iterations.

        First proof: Consider the i-th bit. There are exactly three options for it:

            it is not included in the mask m (and therefore not included in submask s),
            it is included in m, but not included in s, or
            it is included in both m and s.
        As there are a total of n bits, there will be 3n different combinations.

        Second proof: Note that if mask m has k enabled bits, then it will have 2k submasks. 
        As we have a total of (nk) masks with k enabled bits (see binomial coefficients), 
        then the total number of combinations for all masks will be:

        ∑k=0n(nk)⋅2k
        To calculate this number, note that the sum above is equal to the expansion of (1+2)n using the binomial theorem. 
        Therefore, we have 3n combinations, as we wanted to prove.





#####################################################################################################################
#######################################################################################################################3


########################################################################################################################
######################################################################################################################3

COMPETITIVE PROGRAMMING GRAPH ALGORITHM GUIDE:

    DFS ------------
        While running DFS, we assign colors to the vertices (initially white).

        dfs (v):
                color[v] = gray
                for u in adj[v]:
                        if color[u] == white
                                then dfs(u)
                color[v] = black

        Time complexity : O(n + m).

    DFS tree --------
        let T be a new tree
        dfs (v):
                color[v] = gray
                for u in adj[v]:
                        if color[u] == white
                                then dfs(u) and par[u] = v (in T)

                color[v] = black
                
        Lemma: There is no cross edges, it means if there is an edge between v and u, 
               then v = par[u] or u = par[v].

    Starting time, finishing time --------
        TIME = 0
        dfs (v):
                st[v] = TIME ++
                color[v] = gray
                for u in adj[v]:
                        if color[u] == white
                                then dfs(u)
                color[v] = black
                ft[v] = TIME // or we can use TIME ++
        
        It is useable in specially data structure problems (convert the tree into an array).
        Lemma: If we run dfs(root) in a rooted tree, then v is an 
               ancestor of u if and only if stv ≤ stu ≤ ftu ≤ ftv .
        So, given arrays st and ft we can rebuild the tree.

    Finding cut edges -------------------
        The code below works properly because the lemma above (first lemma):
        h is the height of the vertex. v is the parent. u is the child.

        We need compute for each subtree, the lowest node in the DFS tree that a back edge can reach. 
        This value can either be the depth of the other end point, or the discovery time. 
        Cut edges can, also, be seen as edges that needs to be removed 
        to end up with strongly connected components.



        h[root] = 0
        par[v] = -1
        dfs (v):
                d[v] = h[v]
                color[v] = gray
                for u in adj[v]:
                        if color[u] == white
                                then par[u] = v and dfs(u) and d[v] = min(d[v], d[u])
                                if d[u] > h[v]
                                        then the edge v-u is a cut edge
                        else if u != par[v])
                                then d[v] = min(d[v], h[u])
                color[v] = black

        In this code, h[v] =  height of vertex v in the DFS tree and d[v] = min(h[w] where 
                                            there is at least vertex u in subtree of v in 
                                      the DFS tree where there is an edge between u and w).

    Finding cut vertices -----------------
        The code below works properly because the lemma above (first lemma):

        h[root] = 0
        par[v] = -1
        dfs (v):
                d[v] = h[v]
                color[v] = gray
                for u in adj[v]:
                        if color[u] == white
                                then par[u] = v and dfs(u) and d[v] = min(d[v], d[u])
                                if d[u] >= h[v] and (v != root or number_of_children(v) > 1)
                                        then the edge v is a cut vertex
                        else if u != par[v])
                                then d[v] = min(d[v], h[u])
                color[v] = black

        In this code, h[v] =  height of vertex v in the DFS tree and d[v] = min(h[w] where 
        there is at least vertex u in subtree of v in the DFS tree where there is an edge between u and w).

    Finding Eulerian tours ----------------
        It is quite like DFS, with a little change :

        vector E
        dfs (v):
                color[v] = gray
                for u in adj[v]:
                        erase the edge v-u and dfs(u)
                color[v] = black
                push v at the end of e
        e is the answer.


    BFS tree
    BFS tree is a rooted tree that is built like this :

        let T be a new tree
        
        for each vertex i
                do d[i] = inf
        d[v] = 0
        queue q
        q.push(v)
        while q is not empty
            u = q.front()
            q.pop()
            for each w in adj[u]
                if d[w] == inf
                    then d[w] = d[u] + 1, q.push(w) and par[w] = u (in T)

    SCC
    The most useful and fast-coding algorithm for finding SCCs is Kosaraju.
    1) In this algorithm, first of all we run DFS on the graph 
       and sort the vertices in decreasing of their finishing time 
       (we can use a stack).
    
    2) Then, we start from the vertex with the greatest finishing time, and 
        (OPERATE ON THE REVERSED GRAPH)
        for each vertex  v that is not yet in any SCC do : 

            for each u that v is reachable by u and u is not yet in any SCC, 
                put it in the SCC of vertex v. 


    Dijkstra -----------------------
    This algorithm is a single source shortest path (from one source to any other vertices). 
    Pay attention that you can't have edges with negative weight.

        dijkstra(v) :
                d[i] = inf for each vertex i
                d[v] = 0
                s = new empty set
                while s.size() < n
                        x = inf
                        u = -1
                        for each i in V-s //V is the set of vertices
                                if x >= d[i]
                                        then x = d[i], u = i
                        insert u into s
                        // The process from now is called Relaxing
                        for each i in adj[u]
                                d[i] = min(d[i], d[u] + w(u,i))
                                
        There are two different implementations for this. Both are useful (C++11).

        One) O(n2)

        int mark[MAXN];
        void dijkstra(int v){
            fill(d,d + n, inf);
            fill(mark, mark + n, false);
            d[v] = 0;
            int u;
            while(true){
                int x = inf;
                u = -1;
                for(int i = 0;i < n;i ++)
                    if(!mark[i] and x >= d[i])
                        x = d[i], u = i;
                if(u == -1)	break;
                mark[u] = true;
                for(auto p : adj[u]) //adj[v][i] = pair(vertex, weight)
                    if(d[p.first] > d[u] + p.second)
                        d[p.first] = d[u] + p.second;
            }
        }
        Two) 

        1) Using std :: set :

        void dijkstra(int v){
            fill(d,d + n, inf);
            d[v] = 0;
            int u;
            set<pair<int,int> > s;
            s.insert({d[v], v});
            while(!s.empty()){
                u = s.begin() -> second;
                s.erase(s.begin());
                for(auto p : adj[u]) //adj[v][i] = pair(vertex, weight)
                    if(d[p.first] > d[u] + p.second){
                        s.erase({d[p.first], p.first});
                        d[p.first] = d[u] + p.second;
                        s.insert({d[p.first], p.first});
                    }
            }
        }
        2) Using std :: priority_queue (better):

        bool mark[MAXN];
        void dijkstra(int v){
            fill(d,d + n, inf);
            fill(mark, mark + n, false);
            d[v] = 0;
            int u;
            priority_queue<pair<int,int>,vector<pair<int,int> >, less<pair<int,int> > > pq;
            pq.push({d[v], v});
            while(!pq.empty()){
                u = pq.top().second;
                pq.pop();
                if(mark[u])
                    continue;
                mark[u] = true;
                for(auto p : adj[u]) //adj[v][i] = pair(vertex, weight)
                    if(d[p.first] > d[u] + p.second){
                        d[p.first] = d[u] + p.second;
                        pq.push({d[p.first], p.first});
                    }
            }
        }

    Floyd-Warshall---------------------
        Floyd-Warshal algorithm is an all-pairs shortest 
        path algorithm using dynamic programming.

            Floyd-Warshal(graph)
                # start by assigning all the graph weights to d[i][j],
                # and set the non-neighbors as infinite

                d[v][u] = inf for each pair (v,u)
                d[v][v] = 0 for each vertex v
                for k = 1 to n
                    for i = 1 to n
                        for j = 1 to n
                            d[i][j] = min(d[i][j], d[i][k] + d[k][j])
            Time complexity : O(n3).

    Bellman-Ford------------------------------
        Bellman-Ford is an algorithm for single source shortest path where 
        edges can be negative (but if there is a cycle with negative weight, 
        then this problem will be NP).

        The main idea is to relax all the edges exactly n - 1 times 
        (read relaxation above in dijkstra). 
        You can prove this algorithm using induction.

        If in the n - th step, we relax an edge, 
        then we have a negative cycle (this is if and only if).


        Bellman-Ford(int v)
            d[i] = inf for each vertex i
            d[v] = 0
            for step = 1 to n
                for all edges like e
                    i = e.first // first end
                    j = e.second // second end
                    w = e.weight
                    if d[j] > d[i] + w
                        if step == n
                            then return "Negative cycle found"
                        d[j] = d[i] + w
        Time complexity : O(nm).


    SPFA -----------------------------
        SPFA (Shortest Path Faster Algorithm) is a fast and simple algorithm (single source) 
        that its complexity is not calculated yet. But if m = O(n2) it's 
        better to use the first implementation of Dijkstra.

        Its code looks like the combination of Dijkstra and BFS :

        SPFA(v):
            d[i] = inf for each vertex i
            d[v] = 0
            queue q
            q.push(v)
            while q is not empty
                u = q.front()
                q.pop()
                for each i in adj[u]
                    if d[i] > d[u] + w(u,i)
                        then d[i] = d[u] + w(u,i)
                        if i is not in q
                            then q.push(i)
        Time complexity : Unknown!.

    Kruskals Algo:
        # Sort all the edges of the graph from low weight to high.
        # Take the edge of the lowest weight and add it to the required spanning tree. If adding this edge creates a cycle in the graph, then reject this edge.
        # Repeat this process until all the vertices are covered with the edges.


        #Implementing Disjoint Set data structure and its functions
        class DisjointSet:
            def __init__(self, vertices):
                self.vertices = vertices
                self.parent = {}
                for v in vertices:
                    self.parent[v] = v
                self.rank = dict.fromkeys(vertices, 0)

            def find(self, item):
                if self.parent[item] == item:
                    return item
                else: # could do  path compression here
                    return self.find(self.parent[item])

            def union(self, x, y):
                xroot = self.find(x)
                yroot = self.find(y)
                if self.rank[xroot] < self.rank[yroot]:
                    self.parent[xroot] = yroot
                elif self.rank[xroot] > self.rank[yroot]:
                    self.parent[yroot] = xroot
                else:
                    self.parent[yroot] = xroot
                    self.rank[xroot] += 1


        #Function to implement Kruskal's Algorithm
        def kruskalAlgo(self):
            i, e = 0, 0
            ds = dst.DisjointSet(self.nodes)
            self.graph = sorted(self.graph, key=lambda item: item[2])
            while e < self.V - 1:
                s, d, w = self.graph[i]
                i += 1
                x = ds.find(s)
                y = ds.find(d)
                if x != y:
                    e += 1
                    self.MST.append([s,d,w])
                    ds.union(x,y)
            self.printSolution(s,d,w)

        g = Graph(5)
        g.addNode("A")
        g.addNode("B")
        g.addNode("C")
        g.addNode("D")
        g.addNode("E")
        g.addEdge("A", "B", 5)
        g.addEdge("A", "C", 13)
        g.addEdge("A", "E", 15)
        g.addEdge("B", "A", 5)
        g.addEdge("B", "C", 10)
        g.addEdge("B", "D", 8)
        g.addEdge("C", "A", 13)
        g.addEdge("C", "B", 10)
        g.addEdge("C", "E", 20)
        g.addEdge("C", "D", 6)
        g.addEdge("D", "B", 8)
        g.addEdge("D", "C", 6)
        g.addEdge("E", "A", 15)
        g.addEdge("E", "C", 20)

        g.kruskalAlgo()



    Prims Algo:
        In Prims algorithm for a minimum spanning tree, is the starting vertex arbitrary?

        Yes. The key observation is that for each cut in the graph, the cheapest of the edges that
        form the cut can be included in the minimum spanning tree (MST). The Jarník-Prim algorithm uses this observation
        repeatedly to “grow” an MST: in the beginning you start from any single vertex, and then in each iteration you
        split the vertices into two parts: those that already form the tree you are growing, and the rest of the graph.

        from collections import defaultdict
        import heapq


        def create_spanning_tree(graph, starting_vertex):
            mst = defaultdict(set)
            visited = set([starting_vertex])
            edges = [
                (cost, starting_vertex, to)
                for to, cost in graph[starting_vertex].items()
            ]
            heapq.heapify(edges)

            while edges:
                cost, frm, to = heapq.heappop(edges)
                if to not in visited:
                    visited.add(to)
                    mst[frm].add(to)
                    for to_next, cost in graph[to].items():
                        if to_next not in visited:
                            heapq.heappush(edges, (cost, to, to_next))

            return mst

        example_graph = {
            'A': {'B': 2, 'C': 3},
            'B': {'A': 2, 'C': 1, 'D': 1, 'E': 4},
            'C': {'A': 3, 'B': 1, 'F': 5},
            'D': {'B': 1, 'E': 1},
            'E': {'B': 4, 'D': 1, 'F': 1},
            'F': {'C': 5, 'E': 1, 'G': 1},
            'G': {'F': 1},
        }

        dict(create_spanning_tree(example_graph, 'A'))

        # {'A': set(['B']),
        #  'B': set(['C', 'D']),
        #  'D': set(['E']),
        #  'E': set(['F']),
        #  'F': set(['G'])}


    Prims and Kruskals Impl:
    1584. Min Cost to Connect All Points

        You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].

        The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.

        Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.

        Prims Soln:
            class Solution:
                def minCostConnectPoints(self, points: List[List[int]]) -> int:

                    def manhattan_distance(point_1, point_2):
                        x1, y1 = point_1
                        x2, y2 = point_2
                        return abs(x2 - x1) + abs(y2 - y1)

                    n = len(points)
                    graph = collections.defaultdict(list)
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = manhattan_distance(points[i], points[j])
                            graph[i].append((dist, j))
                            graph[j].append((dist, i))

                    visited = set()
                    visited.add(0)
                    heap = [_ for _ in graph[0]]
                    heapq.heapify(heap)
                    cost = 0

                    while heap and len(visited) != len(points):
                        dist, neighbor = heapq.heappop(heap)
                        if neighbor not in visited:
                            visited.add(neighbor)
                            cost += dist
                            for new_dist, new_neighbor in graph[neighbor]:
                                if new_neighbor not in visited:
                                    heapq.heappush(heap, (new_dist, new_neighbor))

                    return cost

        Kruskals Soln:

            class DisjointSetUnion:
                def __init__(self, n):
                    self.parent = [i for i in range(n)]
                    self.size = [1 for i in range(n)]

                def find(self, x):
                    if self.parent[x] != x:
                        self.parent[x] = self.find(self.parent[x])
                    return self.parent[x]

                def union(self, x, y):
                    px, py = self.find(x), self.find(y)
                    if px == py:
                        return True

                    if self.size[px] > self.size[py]:
                        px, py = py, px

                    self.size[py] += self.size[px]
                    self.parent[px] = py
                    return False


            class Solution:
                def minCostConnectPoints(self, points: List[List[int]]) -> int:

                    def manhattan_distance(point_1, point_2):
                        x1, y1 = point_1
                        x2, y2 = point_2
                        return abs(x2 - x1) + abs(y2 - y1)

                    n = len(points)
                    heap = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            heap.append((manhattan_distance(points[i], points[j]), i, j))

                    heapq.heapify(heap)
                    cost = 0
                    num_edges = n - 1
                    edge_count = 0

                    dsu = DisjointSetUnion(n)
                    while heap and edge_count < num_edges:
                        dist, i, j  = heapq.heappop(heap)
                        if dsu.find(i) != dsu.find(j):
                            cost += dist
                            edge_count += 1
                            dsu.union(i,j)

                    return cost




    Kruskal ------------------------------------------
        In this algorithm, first we sort the edges in ascending order of 
        their weight in an array of edges.
        
        MERGE IS DONE WITH UNION FIND!!

        Then in order of the sorted array, we add ech edge if and only if 
        after adding it there won't be any cycle (check it using DSU).

        Kruskal()
            solve all edges in ascending order of their weight in an array e
            ans = 0
            for i = 1 to m
                v = e.first
                u = e.second
                w = e.weight
                if merge(v,u) // there will be no cycle
                    then ans += w
        Time complexity : .

    Prim -----------------------------------------------
        In this approach, we act like Dijkstra. We have a set of vertices S, 
        in each step we add the nearest vertex to S, in S 
        (distance of v from  where weight(i, j) 
        is the weight of the edge from i to j) .

        So, pseudo code will be like this:

        Prim()
            S = new empty set
            for i = 1 to n
                d[i] = inf
            while S.size() < n
                x = inf
                v = -1
                for each i in V - S // V is the set of vertices
                    if x >= d[v]
                        then x = d[v], v = i
                d[v] = 0
                S.insert(v)
                for each u in adj[v]
                    do d[u] = min(d[u], w(v,u))
        C++ code:

        One) O(n2)

        bool mark[MAXN];
        void prim(){
            fill(d, d + n, inf);
            fill(mark, mark + n, false);
            int x,v;
            while(true){
                x = inf;
                v = -1;
                for(int i = 0;i < n;i ++)
                    if(!mark[i] and x >= d[i])
                        x = d[i], v = i;
                if(v == -1)
                    break;
                d[v] = 0;
                mark[v] = true;
                for(auto p : adj[v]){ //adj[v][i] = pair(vertex, weight)
                    int u = p.first, w = p.second;
                    d[u] = min(d[u], w);
                }
            }
        }
        Two) 

        void prim(){
            fill(d, d + n, inf);
            set<pair<int,int> > s;
            for(int i = 0;i < n;i ++)
                s.insert({d[i],i});
            int v;
            while(!s.empty()){
                v = s.begin() -> second;
                s.erase(s.begin());
                for(auto p : adj[v]){
                    int u = p.first, w = p.second;
                    if(d[u] > w){
                        s.erase({d[u], u});
                        d[u] = w;
                        s.insert({d[u], u});
                    }
                }
            }
        }
        As Dijkstra you can use std :: priority_queue instead of std :: set. (I think set begin works because its a binary tree)

    Maximum Flow
    
        algorithm EdmondsKarp
            input:
                C[1..n, 1..n] (Capacity matrix)
                E[1..n, 1..?] (Neighbour lists)
                s             (Source)
                t             (Sink)
            output:
                f             (Value of maximum flow)
                F             (A matrix giving a legal flow with the maximum value)
            f := 0 (Initial flow is zero)
            F := array(1..n, 1..n) (Residual capacity from u to v is C[u,v] - F[u,v])
            forever
                m, P := BreadthFirstSearch(C, E, s, t, F)
                if m = 0
                    break
                f := f + m
                (Backtrack search, and write flow)
                v := t
                while v ≠ s
                    u := P[v]
                    F[u,v] := F[u,v] + m
                    F[v,u] := F[v,u] - m
                    v := u
            return (f, F)
            
            
            
        algorithm BreadthFirstSearch
            input:
                C, E, s, t, F
            output:
                M[t]          (Capacity of path found)
                P             (Parent table)
            P := array(1..n)
            for u in 1..n
                P[u] := -1
            P[s] := -2 (make sure source is not rediscovered)
            M := array(1..n) (Capacity of found path to node)
            M[s] := ∞
            Q := queue()
            Q.offer(s)
            while Q.size() > 0
                u := Q.poll()
                for v in E[u]
                    (If there is available capacity, and v is not seen before in search)
                    if C[u,v] - F[u,v] > 0 and P[v] = -1
                        P[v] := u
                        M[v] := min(M[u], C[u,v] - F[u,v])
                        if v ≠ t
                            Q.offer(v)
                        else
                            return M[t], P
            return 0, P
            
            
            
        EdmondsKarp pseudo code using Adjacency nodes:

        algorithm EdmondsKarp
            input:
                graph (Graph with list of Adjacency nodes with capacities,flow,reverse and destinations)
                s             (Source)
                t             (Sink)
            output:
                flow             (Value of maximum flow)
            flow := 0 (Initial flow to zero)
            q := array(1..n) (Initialize q to graph length)
            while true
                qt := 0            (Variable to iterate over all the corresponding edges for a source)
                q[qt++] := s    (initialize source array)
                pred := array(q.length)    (Initialize predecessor List with the graph length)
                for qh=0;qh < qt && pred[t] == null
                    cur := q[qh]
                    for (graph[cur]) (Iterate over list of Edges)
                        Edge[] e :=  graph[cur]  (Each edge should be associated with Capacity)
                        if pred[e.t] == null && e.cap > e.f
                            pred[e.t] := e
                            q[qt++] : = e.t
                if pred[t] == null
                    break
                int df := MAX VALUE (Initialize to max integer value)
                for u = t; u != s; u = pred[u].s
                    df := min(df, pred[u].cap - pred[u].f)
                for u = t; u != s; u = pred[u].s
                    pred[u].f  := pred[u].f + df
                    pEdge := array(PredEdge)
                    pEdge := graph[pred[u].t]
                    pEdge[pred[u].rev].f := pEdge[pred[u].rev].f - df;
                flow := flow + df
            return flow
            
    Dinic's algorithm -------------------------------------
        Here is Dinic's algorithm as you wanted.

        Input: A network G = ((V, E), c, s, t).

        Output: A max s - t flow.

        1.set f(e) = 0 for each e in E
        2.Construct G_L from G_f of G. if dist(t) == inf, then stop and output f 
        3.Find a blocking flow fp in G_L
        4.Augment flow f by fp  and go back to step 2.
        Time complexity : .

        Theorem: Maximum flow = minimum cut.

 
    Maximum matching in bipartite graphs is ---------------------
        solvable also by maximum flow like below :

        Add two vertices S, T to the graph, every edge from X to Y (graph parts) 
        has capacity 1, add an edge from S with capacity 1 to every vertex in X, 
        add an edge from every vertex in Y with capacity 1 to T.

        Finally, answer = maximum matching from S to T .

        But it can be done  easier using DFS.

        As, you know, a bipartite matching is the maximum matching 
        if and only if there is no augmenting path 
        (read Introduction to graph theory).

        The code below finds a augmenting path:

        bool dfs(int v){
            // v is in X, it reaturns true if and only 
            // if there is an augmenting path starting from v
            if(mark[v])
                return false;
            mark[v] = true;
            for(auto &u : adj[v])
                if(match[u] == -1 or dfs(match[u]))
                    // match[i] = the vertex i is matched
                    // with in the current matching, initially -1
                    return match[v] = u, match[u] = v, true;
            return false;
        }

        An easy way to solve the problem is:

        for(int i = 0;i < n;i ++)if(match[i] == -1){
            memset(mark, false, sizeof mark);
            dfs(i);
        }

        But there is a faster way:

        while(true){
            memset(mark, false, sizeof mark);
            bool fnd = false;
            for(int i = 0;i < n;i ++) if(match[i] == -1 && !mark[i])
                fnd |= dfs(i);
            if(!fnd)
                break;
        }

        In both cases, time complexity = O(nm).

##################################################################

PRIMS ALGORITHM WITH MIN HEAP AND DECREASE KEY IMPLEMETED (C++):

2.313) PRIMS runs FASTER WITH MIN HEAP, ESP FIBONNACI HEAP
        (We show implementation with normal heap + all heap operations implemented):
            // C / C++ program for Prim's MST for adjacency list representation of graph

            #include <limits.h>
            #include <stdio.h>
            #include <stdlib.h>

            // A structure to represent a node in adjacency list
            struct AdjListNode {
                int dest;
                int weight;
                struct AdjListNode* next;
            };

            // A structure to represent an adjacency list
            struct AdjList {
                struct AdjListNode* head; // pointer to head node of list
            };

            // A structure to represent a graph. A graph is an array of adjacency lists.
            // Size of array will be V (number of vertices in graph)
            struct Graph {
                int V;
                struct AdjList* array;
            };

            // A utility function to create a new adjacency list node
            struct AdjListNode* newAdjListNode(int dest, int weight)
            {
                struct AdjListNode* newNode = (struct AdjListNode*)malloc(sizeof(struct AdjListNode));
                newNode->dest = dest;
                newNode->weight = weight;
                newNode->next = NULL;
                return newNode;
            }

            // A utility function that creates a graph of V vertices
            struct Graph* createGraph(int V)
            {
                struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
                graph->V = V;

                // Create an array of adjacency lists.  Size of array will be V
                graph->array = (struct AdjList*)malloc(V * sizeof(struct AdjList));

                // Initialize each adjacency list as empty by making head as NULL
                for (int i = 0; i < V; ++i)
                    graph->array[i].head = NULL;

                return graph;
            }

            // Adds an edge to an undirected graph
            void addEdge(struct Graph* graph, int src, int dest, int weight)
            {
                // Add an edge from src to dest.  A new node is added to the adjacency
                // list of src.  The node is added at the beginning
                struct AdjListNode* newNode = newAdjListNode(dest, weight);
                newNode->next = graph->array[src].head;
                graph->array[src].head = newNode;

                // Since graph is undirected, add an edge from dest to src also
                newNode = newAdjListNode(src, weight);
                newNode->next = graph->array[dest].head;
                graph->array[dest].head = newNode;
            }

            // Structure to represent a min heap node
            struct MinHeapNode {
                int v;
                int key;
            };

            // Structure to represent a min heap
            struct MinHeap {
                int size; // Number of heap nodes present currently
                int capacity; // Capacity of min heap
                int* pos; // This is needed for decreaseKey()
                struct MinHeapNode** array;
            };

            // A utility function to create a new Min Heap Node
            struct MinHeapNode* newMinHeapNode(int v, int key)
            {
                struct MinHeapNode* minHeapNode = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
                minHeapNode->v = v;
                minHeapNode->key = key;
                return minHeapNode;
            }

            // A utilit function to create a Min Heap
            struct MinHeap* createMinHeap(int capacity)
            {
                struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
                minHeap->pos = (int*)malloc(capacity * sizeof(int));
                minHeap->size = 0;
                minHeap->capacity = capacity;
                minHeap->array = (struct MinHeapNode**)malloc(capacity * sizeof(struct MinHeapNode*));
                return minHeap;
            }

            // A utility function to swap two nodes of min heap. Needed for min heapify
            void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
            {
                struct MinHeapNode* t = *a;
                *a = *b;
                *b = t;
            }

            // A standard function to heapify at given idx
            // This function also updates position of nodes when they are swapped.
            // Position is needed for decreaseKey()
            void minHeapify(struct MinHeap* minHeap, int idx)
            {
                int smallest, left, right;
                smallest = idx;
                left = 2 * idx + 1;
                right = 2 * idx + 2;

                if (left < minHeap->size && minHeap->array[left]->key < minHeap->array[smallest]->key)
                    smallest = left;

                if (right < minHeap->size && minHeap->array[right]->key < minHeap->array[smallest]->key)
                    smallest = right;

                if (smallest != idx) {
                    // The nodes to be swapped in min heap
                    MinHeapNode* smallestNode = minHeap->array[smallest];
                    MinHeapNode* idxNode = minHeap->array[idx];

                    // Swap positions
                    minHeap->pos[smallestNode->v] = idx;
                    minHeap->pos[idxNode->v] = smallest;

                    // Swap nodes
                    swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);

                    minHeapify(minHeap, smallest);
                }
            }

            // A utility function to check if the given minHeap is ampty or not
            int isEmpty(struct MinHeap* minHeap)
            {
                return minHeap->size == 0;
            }

            // Standard function to extract minimum node from heap
            struct MinHeapNode* extractMin(struct MinHeap* minHeap)
            {
                if (isEmpty(minHeap))
                    return NULL;

                // Store the root node
                struct MinHeapNode* root = minHeap->array[0];

                // Replace root node with last node
                struct MinHeapNode* lastNode = minHeap->array[minHeap->size - 1];
                minHeap->array[0] = lastNode;

                // Update position of last node
                minHeap->pos[root->v] = minHeap->size - 1;
                minHeap->pos[lastNode->v] = 0;

                // Reduce heap size and heapify root
                --minHeap->size;
                minHeapify(minHeap, 0);

                return root;
            }

            // Function to decrease key value of a given vertex v. This function
            // uses pos[] of min heap to get the current index of node in min heap
            void decreaseKey(struct MinHeap* minHeap, int v, int key)
            {
                // Get the index of v in  heap array
                int i = minHeap->pos[v];

                // Get the node and update its key value
                minHeap->array[i]->key = key;

                // Travel up while the complete tree is not hepified.
                // This is a O(Logn) loop
                while (i && minHeap->array[i]->key < minHeap->array[(i - 1) / 2]->key) {
                    // Swap this node with its parent
                    minHeap->pos[minHeap->array[i]->v] = (i - 1) / 2;
                    minHeap->pos[minHeap->array[(i - 1) / 2]->v] = i;
                    swapMinHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);

                    // move to parent index
                    i = (i - 1) / 2;
                }
            }

            // A utility function to check if a given vertex
            // 'v' is in min heap or not
            bool isInMinHeap(struct MinHeap* minHeap, int v)
            {
                if (minHeap->pos[v] < minHeap->size)
                    return true;
                return false;
            }

            // A utility function used to print the constructed MST
            void printArr(int arr[], int n)
            {
                for (int i = 1; i < n; ++i)
                    printf("%d - %d\n", arr[i], i);
            }

            // The main function that constructs Minimum Spanning Tree (MST)
            // using Prim's algorithm
            void PrimMST(struct Graph* graph)
            {
                int V = graph->V; // Get the number of vertices in graph
                int parent[V]; // Array to store constructed MST
                int key[V]; // Key values used to pick minimum weight edge in cut

                // minHeap represents set E
                struct MinHeap* minHeap = createMinHeap(V);

                // Initialize min heap with all vertices. Key value of
                // all vertices (except 0th vertex) is initially infinite
                for (int v = 1; v < V; ++v) {
                    parent[v] = -1;
                    key[v] = INT_MAX;
                    minHeap->array[v] = newMinHeapNode(v, key[v]);
                    minHeap->pos[v] = v;
                }

                // Make key value of 0th vertex as 0 so that it
                // is extracted first
                key[0] = 0;
                minHeap->array[0] = newMinHeapNode(0, key[0]);
                minHeap->pos[0] = 0;

                // Initially size of min heap is equal to V
                minHeap->size = V;

                // In the following loop, min heap contains all nodes
                // not yet added to MST.
                while (!isEmpty(minHeap)) {
                    // Extract the vertex with minimum key value
                    struct MinHeapNode* minHeapNode = extractMin(minHeap);
                    int u = minHeapNode->v; // Store the extracted vertex number

                    // Traverse through all adjacent vertices of u (the extracted
                    // vertex) and update their key values
                    struct AdjListNode* pCrawl = graph->array[u].head;
                    while (pCrawl != NULL) {
                        int v = pCrawl->dest;

                        // If v is not yet included in MST and weight of u-v is
                        // less than key value of v, then update key value and
                        // parent of v
                        if (isInMinHeap(minHeap, v) && pCrawl->weight < key[v]) {
                            key[v] = pCrawl->weight;
                            parent[v] = u;
                            decreaseKey(minHeap, v, key[v]);
                        }
                        pCrawl = pCrawl->next;
                    }
                }

                // print edges of MST
                printArr(parent, V);
            }

            // Driver program to test above functions
            int main()
            {
                // Let us create the graph given in above fugure
                int V = 9;
                struct Graph* graph = createGraph(V);
                addEdge(graph, 0, 1, 4);
                addEdge(graph, 0, 7, 8);
                addEdge(graph, 1, 2, 8);
                addEdge(graph, 1, 7, 11);
                addEdge(graph, 2, 3, 7);
                addEdge(graph, 2, 8, 2);
                addEdge(graph, 2, 5, 4);
                addEdge(graph, 3, 4, 9);
                addEdge(graph, 3, 5, 14);
                addEdge(graph, 4, 5, 10);
                addEdge(graph, 5, 6, 2);
                addEdge(graph, 6, 7, 1);
                addEdge(graph, 6, 8, 6);
                addEdge(graph, 7, 8, 7);

                PrimMST(graph);

                return 0;
            }





#######################################################################################################################
########################################################################################################################

GRAPH REFERENCE (From cheatsheet algos.py)

    WHITE = 0
    GREY = 1
    BLACK = 2

    def breadth_first_search(G, s):
        color = {v: WHITE for v in G}
        color[s] = GREY
        pi = {s: None}
        dist = {s: 0}

        queue = [s]
        while queue:
            u = queue.pop(0)
            for v in G[u]:
                if color[v] == WHITE:
                    color[v] = GREY
                    pi[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
            color[u] = BLACK

        return color, pi, dist

    def depth_first_search(G, s):
        color = {v: WHITE for v in G}
        pi = {s: None}
        time = 0
        finish_time = {}

        stack = [s]
        while stack:
            u = stack.pop()
            if color[u] == WHITE:
                color[u] = GREY
                stack.append(u)
                for v in G[u]:
                    if color[v] == WHITE:
                        stack.append(v)
                        pi[v] = u
            elif color[u] == GREY:
                color[u] = BLACK
                finish_time[u] = time
                time += 1

        return color, pi, finish_time

    def depth_first_search_rec(G, s=None):
        color = {v: WHITE for v in G}
        pi = {}
        time = 0
        finish_time = {}

        def visit(u):
            nonlocal time

            color[u] = GREY
            for v in reversed(G[u]):
                if color[v] == WHITE:
                    pi[v] = u
                    visit(v)
            color[u] = BLACK
            finish_time[u] = time
            time += 1

        if s:
            # Single source
            pi[s] = None
            visit(s)
        else:
            # DFS forest
            for s in G:
                if color[s] == WHITE:
                    pi[s] = None
                    visit(s)

        return color, pi, finish_time

    def bipartition(G):
        seen = set()
        partition = {}
        for s in G:
            if not s in seen:
                color, _, dist = breadth_first_search(G, s)
                seen.update(v for v, c in color.items() if c == BLACK)
                for v, d in dist.items():
                    partition[v] = d % 2

        for u in G:
            for v in G[u]:
                if partition[u] == partition[v]:
                    return False

        return partition

    def undirected_has_cycle(G):
        color = {v: WHITE for v in G}
        cycle = False

        def visit(u, p):
            nonlocal cycle
            if cycle:
                return

            color[u] = GREY
            for v in G[u]:
                if color[v] == WHITE:
                    visit(v, u)
                elif v != p and color[v] == GREY:
                    cycle = True
            color[u] = BLACK

        for s in G:
            if color[s] == WHITE:
                visit(s, None)
                if cycle:
                    return True

        return cycle

    def topological_sort(G):
        color = {v: WHITE for v in G}
        order = []
        dag = True

        def visit(u):
            nonlocal dag
            if not dag:
                return

            order.append(u)
            color[u] = GREY
            for v in G[u]:
                if color[v] == WHITE:
                    visit(v)
                elif color[v] == GREY:
                    dag = False
            color[u] = BLACK

        for s in G:
            if color[s] == WHITE:
                visit(s)
                if not dag:
                    return False

        return order

    def topological_sort_alt(G):
        in_degree = {v : 0 for v in G}
        for neighbours in G.values():
            for v in neighbours:
                in_degree[v] += 1

        queue = []
        for v in G:
            if in_degree[v] == 0:
                queue.append(v)

        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in G[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) < len(G):
            return False
        return order

    def reverse_graph(G):
        Grev = {v: [] for v in G}
        for u, neighbours in G.items():
            for v in neighbours:
                Grev[v].append(u)
        return Grev

    def kosaraju(G):
        f = depth_first_search_rec(G)[2] # finish times
        Grev = reverse_graph(G)

        color = {v: WHITE for v in G}
        sccs = []

        def visit(u):
            color[u] = GREY
            sccs[-1].append(u)

            for v in Grev[u]:
                if color[v] == WHITE:
                    visit(v)

            color[u] = BLACK

        for s in sorted(G.keys(), key=lambda v: f[v], reverse=True):
            if color[s] == WHITE:
                sccs.append([])
                visit(s)

        return sccs

    def kruskal(G, w, s):
        mst = {v: [] for v in G}
        edges = sorted(w.keys(), key=lambda e: w[e])
        for u, v in edges:
            mst[u].append(v)
            mst[v].append(u)
            if undirected_has_cycle(mst):
                mst[u].pop()
                mst[v].pop()

        # return parents map
        return breadth_first_search(mst, s)[1]

    def prim(G, w, s):

        def get_weight(u, v):
            if (u, v) in w:
                return w[u, v]
            return w[v, u]

        pi = {s: None}
        connected = {s}

        pq = []
        for v in G[s]:
            heappush(pq, (get_weight(s, v), s, v))

        while pq:
            _, u, v = heappop(pq)
            assert u in connected

            if not v in connected:
                connected.add(v)
                pi[v] = u
                for z in G[v]:
                    heappush(pq, (get_weight(v, z), v, z))

        return pi

    def bfs_sssp(G, s):
        return breadth_first_search(G, s)[2]

    def dijkstra(G, s, w=None):

        def get_weight(u, v):
            return w[u, v] if w else 1

        dist = {s: 0}
        entries = {}
        pq = []
        for v in G[s]:
            d = get_weight(s, v)
            entry = [d, v, True]
            dist[v] = d
            entries[v] = entry
            heappush(pq, entry)

        while pq:
            u_dist, u, valid = heappop(pq)
            if valid:
                for v in G[u]:
                    new_dist = u_dist + get_weight(u, v)
                    if not v in dist or new_dist < dist[v]:
                        dist[v] = new_dist
                        entry = [new_dist, v, True]
                        if v in entries:
                            entries[v][2] = False
                        entries[v] = entry
                        heappush(pq, entry)

        return dist

    def dag_sssp(G, s, w=None):

        def get_weight(u, v):
            return w[u, v] if w else 1

        dist = {s: 0}
        order = topological_sort(G)
        assert order

        for u in order:
            for v in G[u]:
                new_dist = dist[u] + get_weight(u, v)
                if not v in dist or new_dist < dist[v]:
                    dist[v] = new_dist

        return dist

    def floyd_warshall(W):
        n = len(W)
        assert all(len(row) == n for row in W)

        D = [row[:] for row in W]
        E = [[0] * n for _ in range(n)]

        for m in range(n):
            for i in range(n):
                for j in range(n):
                    E[i][j] = min(D[i][j], D[i][m] + D[m][j])
            D, E = E, D

        return D

    def complement_graph(G):
        return {u: [v for v in G if u != v and not v in G[u]] for u in G}

    ### Part 5: Intractability and undecidability

    def clique_to_vertex_cover(G, k, vertex_cover):
        n = len(G)
        assert k <= n

        Gcomp = complement_graph(G)
        return vertex_cover(Gcomp, n - k)


#######################################
##########################################

COOL NOTES 2: EXPRESSION PARSING PROBLEMS

    Reverse Polish notation
        Parsing of simple expressions
        Unary operators
        Right-associativity
        A string containing a mathematical expression containing numbers
        and various operators is given. We have to compute the value of it in O(n),
        where n is the length of the string.

        The algorithm discussed here translates an expression into the so-called
        reverse Polish notation (explicitly or implicitly), and evaluates this expression.

        Reverse Polish notation
        The reverse Polish notation is a form of writing mathematical expressions,
        in which the operators are located after their operands.

        For example the following expression
        a+b∗c∗d+(e−f)∗(g∗h+i)

        can be written in reverse Polish notation in the following way:
        abc∗d∗+ef−gh∗i+∗+

        The convenience of the reverse Polish notation is, that expressions
        in this form are very easy to evaluate in linear time. We use a stack,
        which is initially empty. We will iterate over the operands and operators
        of the expression in reverse Polish notation. If the current element is a number,
        then we put the value on top of the stack, if the current element is an
        operator, then we get the top two elements from the stack, perform the operation,
        and put the result back on top of the stack. In the end there will be
        exactly one element left in the stack, which will be the value of the expression.

        Obviously this simple evaluation runs in O(n) time.



    Parsing of simple expressions

        For the time being we only consider a
        simplified problem: we assume that all operators
        are binary (i.e. they take two arguments), and all are
        left-associative (if the priorities are equal,
        they get executed from left to right). Parentheses are allowed.

        We will set up two stacks: one for numbers, and one for operators
        and parentheses. Initially both stacks are empty. For the second
        stack we will maintain the condition that all operations are
        ordered by strict descending priority. If there are parenthesis on the stack,
        than each block of operators (corresponding to one pair of parenthesis)
        is ordered, and the entire stack is not necessarily ordered.

        We will iterate over the characters of the expression from left to right.
        If the current character is a digit, then we put the value of
        this number on the stack. If the current character is an
        opening parenthesis, then we put it on the stack. If the current
        character is a closing parenthesis, the we execute all operators on the stack
        until we reach the opening bracket (in other words we perform all
        operations inside the parenthesis). Finally if the current character
        is an operator, then while the top of the stack has
        an operator with the same or higher priority, we will execute
        this operation, and put the new operation on the stack.

        After we processed the entire string, some operators might
        still be in the stack, so we execute them.

        Here is the implementation of this method for the four operators + − ∗ /:

        bool delim(char c) {
            return c == ' ';
        }

        bool is_op(char c) {
            return c == '+' || c == '-' || c == '*' || c == '/';
        }

        int priority (char op) {
            if (op == '+' || op == '-')
                return 1;
            if (op == '*' || op == '/')
                return 2;
            return -1;
        }

        void process_op(stack<int>& st, char op) {
            int r = st.top(); st.pop();
            int l = st.top(); st.pop();
            switch (op) {
                case '+': st.push(l + r); break;
                case '-': st.push(l - r); break;
                case '*': st.push(l * r); break;
                case '/': st.push(l / r); break;
            }
        }

        int evaluate(string& s) {
            stack<int> st;
            stack<char> op;
            for (int i = 0; i < (int)s.size(); i++) {
                if (delim(s[i]))
                    continue;

                if (s[i] == '(') {
                    op.push('(');
                } else if (s[i] == ')') {
                    while (op.top() != '(') {
                        process_op(st, op.top());
                        op.pop();
                    }
                    op.pop();
                } else if (is_op(s[i])) {
                    char cur_op = s[i];
                    while (!op.empty() && priority(op.top()) >= priority(cur_op)) {
                        process_op(st, op.top());
                        op.pop();
                    }
                    op.push(cur_op);
                } else {
                    int number = 0;
                    while (i < (int)s.size() && isalnum(s[i]))
                        number = number * 10 + s[i++] - '0';
                    --i;
                    st.push(number);
                }
            }

            while (!op.empty()) {
                process_op(st, op.top());
                op.pop();
            }
            return st.top();
        }

        Thus we learned how to calculate the value of an expression in O(n),
        at the same time we implicitly used the reverse Polish notation.
        By slightly modifying the above implementation it is also possible
        to obtain the expression in reverse Polish notation in an explicit form.

    Parsing of all expressions include unary and right associative expr:

        Unary operators
            Now suppose that the expression also contains unary operators
            (operators that take one argument). The unary plus and
            unary minus are common examples of such operators.

            One of the differences in this case, is that we need to
            determine whether the current operator is a unary or a binary one.

            You can notice, that before an unary operator, there always is
            another operator or an opening parenthesis, or nothing at
            all (if it is at the very beginning of the expression). On the contrary
            before a binary operator there will always be an operand (number)
            or a closing parenthesis. Thus it is easy to flag
            whether the next operator can be unary or not.

            Additionally we need to execute a unary and a binary operator
            differently. And we need to chose the priority of a binary operator
            higher than all of the binary operations.

            In addition it should be noted, that some unary operators
            (e.g. unary plus and unary minus) are actually right-associative.

        Right-associativity
            Right-associative means, that whenever the priorities are equal,
            the operators must be evaluated from right to left.

            As noted above, unary operators are usually right-associative.
            Another example for an right-associative operator is the
            exponentiation operator (a∧b∧c is usually perceived as a^(b^c) and not as (a^b)^c.

            What difference do we need to make in order to correctly handle
            right-associative operators? It turns out that the changes
            are very minimal. The only difference will be, if the priorities
            are equal we will postpone the execution of the right-associative operation.

            The only line that needs to be replaced is

            while (!op.empty() && priority(op.top()) >= priority(cur_op))

            with:

            while (!op.empty() && (
                    (left_assoc(cur_op) && priority(op.top()) >= priority(cur_op)) ||
                    (!left_assoc(cur_op) && priority(op.top()) > priority(cur_op))
                ))

            where left_assoc is a function that decides if an
            operator is left_associative or not.

        Here is an implementation for the binary
        operators + − ∗ / and the unary operators + and −.

            bool delim(char c) {
                return c == ' ';
            }

            bool is_op(char c) {
                return c == '+' || c == '-' || c == '*' || c == '/';
            }

            bool is_unary(char c) {
                return c == '+' || c=='-';
            }

            int priority (char op) {
                if (op < 0) // unary operator get highest priority
                    return 3; // Negative operators are right associative.
                if (op == '+' || op == '-')
                    return 1;
                if (op == '*' || op == '/')
                    return 2;
                return -1;
            }

            void process_op(stack<int>& st, char op) {
                if (op < 0) {
                    int l = st.top(); st.pop();
                    switch (-op) { // Negative operators are right associative.
                        case '+': st.push(l); break;
                        case '-': st.push(-l); break;
                    }
                } else {
                    int r = st.top(); st.pop();
                    int l = st.top(); st.pop();
                    switch (op) {
                        case '+': st.push(l + r); break;
                        case '-': st.push(l - r); break;
                        case '*': st.push(l * r); break;
                        case '/': st.push(l / r); break;
                    }
                }
            }

            int evaluate(string& s) {
                stack<int> st;
                stack<char> op;
                bool may_be_unary = true;
                for (int i = 0; i < (int)s.size(); i++) {
                    if (delim(s[i]))
                        continue;

                    if (s[i] == '(') {
                        op.push('(');
                        may_be_unary = true;
                    } else if (s[i] == ')') {
                        while (op.top() != '(') {
                            process_op(st, op.top());
                            op.pop();
                        }
                        op.pop();
                        may_be_unary = false;
                    } else if (is_op(s[i])) {
                        char cur_op = s[i];
                        if (may_be_unary && is_unary(cur_op))
                            cur_op = -cur_op;
                        while (!op.empty() && (
                                (cur_op >= 0 && priority(op.top()) >= priority(cur_op)) ||
                                (cur_op < 0 && priority(op.top()) > priority(cur_op))
                            )) {
                            process_op(st, op.top());
                            op.pop();
                        }
                        op.push(cur_op);
                        may_be_unary = true;
                    } else {
                        int number = 0;
                        while (i < (int)s.size() && isalnum(s[i]))
                            number = number * 10 + s[i++] - '0';
                        --i;
                        st.push(number);
                        may_be_unary = false;
                    }
                }

                while (!op.empty()) {
                    process_op(st, op.top());
                    op.pop();
                }
                return st.top();
            }


#########################################
########################################

DYNAMIC PROGRAMMING FOR PRACTICE
https://leetcode.com/discuss/interview-question/1380561/Template-For-Dynamic-programming

    Dynmaic Programming For Practice

    Sharing some topic wise good Dynamic Programming problems and sample solutions to observe on how to approach.

    1.Unbounded Knapsack or Target sum
    Identify if problems talks about finding groups or subset which is equal to given target.

    https://leetcode.com/problems/target-sum/
    https://leetcode.com/problems/partition-equal-subset-sum/
    https://leetcode.com/problems/last-stone-weight-ii/
    https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/

    All the above problems can be solved by 01 Knapsack or Target sum algo with minor tweaks.
    Below is a standard code for 01 knapsack or target sum problems.


    int 01 knacsack(vector<int>& nums,vector<int>& v, int w)  // nums array , w total amount that have to collect 
        {                                                     // v value array
            int n=nums.size();
            
            vector<vector<bool>> d(n+1,vector<bool>(w+1,0));  
            for(int i=1;i<=n;i++)
            {
                for(int j=1;j<=w;j++)
                {
                    if(j<nums[i-1])  
                    {
                        d[i][j]=d[i-1][j];
                    }
                    else if(nums[i-1]<=j)
                    {
                        d[i][j]=max(v[i-1]+d[i-1][j-nums[i-1]],d[i-1][j]);
                    }
                }
            }
            
            return d[n][w];
            
        }
    Funtion for Target sum

    int countsubset(vector<int>& nums, int w) 
        {
            int n=nums.size();
            
            vector<vector<bool>> d(n+1,vector<bool>(w+1));
            for(int i=0;i<=n;i++)
            {
                    d[i][0]=1;
            }
            for(int i=1;i<=w;i++)
            {
                    d[0][i]=0;
            }
            
            for(int i=1;i<=n;i++)
            {
                for(int j=1;j<=w;j++)
                {
                    if(j<nums[i-1])
                    {
                        d[i][j]=d[i-1][j];
                    }
                    else if(nums[i-1]<=j)
                    {
                        d[i][j]=d[i-1][j-nums[i-1]] + d[i-1][j];
                    }
                }
            }
            
            return d[n][w];
        }
    2.Unbounded Knapsack
    Identify if problems talks about finding groups or subset which is equal to given target and repetition is allowed.

    https://leetcode.com/problems/coin-change-2/
    https://leetcode.com/problems/coin-change/

    All the above problems can be solved by unbounded Knapsack algo with minor tweaks.
    Below is a standard code for 01 knapsack or target sum problems.

    int unboundedknacsack(vector<int>& nums,vector<int>& v, int w) 
        {
            int n=nums.size();
            
            vector<vector<bool>> d(n+1,vector<bool>(w+1,0));
            for(int i=1;i<=n;i++)
            {
                for(int j=1;j<=w;j++)
                {
                    if(j<nums[i-1])
                    {
                        d[i][j]=d[i-1][j];
                    }
                    else if(nums[i-1]<=j)
                    {
                        d[i][j]=max(v[i-1]+d[i][j-v[i-1]],d[i-1][j]);
                    }
                }
            }
            
            return d[n][w];
        }
    or

        int change(int amount, vector<int>& coins) 
        {
        vector<vector<int>> d(coins.size()+1,vector<int>(amount+1));
            
        for(int i=0;i<=coins.size();i++)
        {
            d[i][0]=1;
        }
            for(int i=1;i<=amount;i++)
        {
            d[0][i]=0;
        }
            
            for(int i=1;i<=coins.size();i++)
        {
            for(int j=1;j<=amount;j++)
            {
                if(j<coins[i-1])
                {
                    d[i][j]=d[i-1][j];
                }
                
                else if(j>=coins[i-1])
                {
                    d[i][j]=(d[i][j-coins[i-1]]+d[i-1][j]);
                }
            }
        }
            
            return d[coins.size()][amount]; 
        }
    3.Longest Increasing Subsequence (LIS)

    Identify if problems talks about finding longest increasing subset.

    https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
    https://leetcode.com/problems/longest-increasing-subsequence/
    https://leetcode.com/problems/largest-divisible-subset/
    https://leetcode.com/problems/perfect-squares/
    https://leetcode.com/problems/super-ugly-number/

    https://leetcode.com/problems/russian-doll-envelopes/
    https://leetcode.com/problems/maximum-height-by-stacking-cuboids/description/

    @Nam_22 mentioning above two question .

    All the above problems can be solved by longest Increasing subsequence algo with minor tweaks.
    Below is a standard code for LIS problems.




        int lengthOfLIS(vector<int>& nums) 
        {
            vector<int> d(nums.size(),1);
            
            int m=0;
            for(int i=0;i<nums.size();i++)
            {
                for(int j=0;j<i;j++)
                {
                    if(nums[j]<nums[i] && d[i]<d[j]+1)
                    {
                        d[i]=d[j]+1;
                    }
                }
                m=max(d[i],m);
            }
        
            return m;
        }

    longest bitonic subsequence

    int lbs(vector<int> v)
    {
        vector<int> lis(v.size(),1);
        vector<int> lds(v.size(),1);
        
    for(int i=0;i<v.size();i++)
        {
            
            for(int j=0;j<i;j++)
            {
                if(v[j]<v[i] && lis[i]<lis[j]+1)
                {
                lis[i]=lis[j]+1;
                }
            }

        }
        
    for(int i=v.size()-2;i>0;i--)
        {
        
        for(int j=v.size()-1;j>i;j--)
        {
            if(v[j]<v[i] && lds[i]<lds[j]+1)
            { 
                lds[i]=lds[j]+1;
            }
        }
        
        }
        
        int m=0;
        for(int i=0;i<v.size();i++)
        { 
            m=max(m,lis[i]+lds[i]-1);
        }
        
        return m;
    }
    4.Longest Common Subsequence

    Identify if problems talks about finding longest common subset.

    1.subsequence
    https://leetcode.com/problems/longest-common-subsequence/
    https://leetcode.com/problems/distinct-subsequences/
    https://leetcode.com/problems/shortest-common-supersequence/
    https://leetcode.com/problems/distinct-subsequences/
    https://leetcode.com/problems/interleaving-string/

    int longestCommonSubsequence(string text1, string text2) 
        {
            int n1 = text1.size();
            int n2 = text2.size();
            vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
            

            for(int i=1;i<=n1;i++)
            {
                for(int j=1;j<=n2;j++)
                {
                    if(text1[i-1] == text2[j-1])
                        dp[i][j] = 1+dp[i-1][j-1];
                    else
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1]);

                }
            }
            return dp[n1][n2];

        }
    2.substring

    https://leetcode.com/problems/maximum-length-of-repeated-subarray/

    int longestCommonSubstring(string text1, string text2) 
        {
            int n1 = text1.size();
            int n2 = text2.size();
            vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
            
            int r=0;
            for(int i=1;i<=n1;i++)
            {
                for(int j=1;j<=n2;j++)
                {
                    if(text1[i-1] == text2[j-1])
                    { 
                        dp[i][j] = 1+dp[i-1][j-1];
                        r=max(dp[i][j],r);
                    }
                    else
                        dp[i][j] = 0;

                }
            }
            return dp[n1][n2];

        }
    3.palindrome

    https://leetcode.com/problems/longest-palindromic-substring/
    https://leetcode.com/problems/longest-palindromic-subsequence/
    https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
    https://leetcode.com/problems/delete-operation-for-two-strings/

    int lps(string s1)
    {
        int n1=s1.length();
        string s2=s1;
        reverse(s2.begin(),s2.end());
        int n2=s2.length();
        
        vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
        
        for(int i=1;i<=n1;i++)
        {
            for(int j=1;j<=n2;j++)
            {
                if(s1[i-1]==s2[j-1])
                dp[i][j]=1+dp[i-1][j-1];
                
                else
                dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
        
        return dp[n1][n2];
    }
    4.Print

    string longestCommonSubsequence(string a) 
    {
    string b=a;
    reverse(b.begin(),b.end());
    
    int n1=a.size();
    int n2=b.size();
    vector<vector<int>> d(n1+1,vector<int>(n2+1,0));
    
    for(int i=1;i<=n1;i++)
    {
        for(int j=1;j<=n2;j++)
        {
            if(a[i-1]==b[j-1])
            {
                d[i][j]=1+d[i-1][j-1];
            }
            else
            {
                d[i][j]=max(d[i-1][j],d[i][j-1]);
            }
        }
    }
    
    string v;
    int i=n1,j=n2;
    while(i>0 && j>0)
    {
        if(a[i-1]==b[j-1])
        {
            v.push_back(a[i-1]);
            i--;
            j--;
        }
        
        else
        {
            if(d[i-1][j]>d[i][j-1])
            {
                i--;
            }
            else
            {
                j--;
            }
        }
    }
    reverse(v.begin(),v.end());
    return v;
    }
    5.Gap Method Problems

    General Dp problem which is solved by Gap method

    https://leetcode.com/problems/count-different-palindromic-subsequences/
    https://leetcode.com/problems/palindrome-partitioning-ii/
    https://leetcode.com/problems/minimum-score-triangulation-of-polygon/

    And Leetcode stones problem set are also included.

    All the above problems can be solved by gap methodwith minor tweaks.
    Below is a standard code for gap method code.

    count palindromic subsequence

    int countPalindromicSubsequences(string s) 
        {
        int d[s.length()][s.length()];  
        
        for(int g=0;g<s.length();g++)
        {
            for(int i=0,j=g;j<s.length();i++,j++)
            {
                if(g==0)
                {
                    d[i][j]=1;
                }
                else if(g==1)
                {
                    if(s[i]==s[j])
                    {
                        d[i][j]=3;
                    }
                    else
                    {
                        d[i][j]=2;
                    }
                }
                
                else
                {
                    if(s[i]==s[j])
                    {
                        d[i][j]=d[i][j-1]+d[i+1][j]+1;
                    }
                    else
                    {
                        d[i][j]=d[i][j-1]+d[i+1][j]-d[i+1][j-1];
                    }
                }
            }
        }
            
            return d[0][s.length()-1];
        }
    6.Kadans algo

    Identify if problems talks about finding the maximum subarray sum.

    https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
    https://leetcode.com/problems/arithmetic-slices/
    https://leetcode.com/problems/arithmetic-slices-ii-subsequence/
    https://leetcode.com/problems/longest-turbulent-subarray/
    https://leetcode.com/problems/k-concatenation-maximum-sum/
    https://leetcode.com/problems/k-concatenation-maximum-sum/
    https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/
    https://leetcode.com/problems/ones-and-zeroes/
    https://leetcode.com/problems/maximum-sum-circular-subarray/

    All the above problems can be solved by gap method with minor tweaks.
    Below is a standard code for gap method code.

    int kad(vector<int> v)
    {
        int c=v[0],o=v[0];
        
        for(int i=1;i<n;i++)
        {
            if(c >= 0)
            {
                c=c+v[i];
            }
            else
            {
                c=v[i];
            }
            
            if(o<c)
            {
                o=c;
            }
            
        }
        
        return o;
    }
    7.Catalan

    Identify if problems talks about counting the number of something.
    eg node,bracket etc.

    https://leetcode.com/problems/unique-binary-search-trees/

    All the above problems can be solved by catalan with minor tweaks.
    Below is a standard code for catalan code.

    int cat(int n)
    {
        int dp[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i = 2; i < n+1; i++)
        {
            dp[i]=0;
            for(int j = 0; j < i; j++)
            {
                dp[i] += dp[j] * dp[i - 1 - j];
            }
        }
        
        return dp[n];
    }

    Please correct the approach/solution if you find anything wrong.
    And if you like my post then give a thumbs up : ) happy coding



############################
+##########################3
+Linked List Practice!

    DRAW IT OUT TO DO THESE PROBLEMS!!
+
+    24. Swap Nodes in Pairs
+    Medium
+    Given a linked list, swap every two adjacent nodes and return its head. 
+    You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
+    Input: head = [1,2,3,4]
+    Output: [2,1,4,3]
+    Example 2:
+    Input: head = []
+    Output: []
+    Example 3:
+    Input: head = [1]
+    Output: [1]
+    My soln:
+    class Solution:
+        def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
+            # okkk
+            
+            dummy = ListNode()
+            temp = dummy
+            curr = head
+            
+            while curr is not None and curr.next is not None:
+                next_adj_pair = curr.next.next
+                nxt = curr.next
+                temp.next = nxt
+                nxt.next = curr
+                temp = curr
+                curr = next_adj_pair
+                
+            temp.next = curr
+            
+            
+            return dummy.next     