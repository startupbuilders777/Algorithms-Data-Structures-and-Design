
For very quick refresher -> Understand the following. He goes through 300 leetcode problems: https://nishmathcs.wordpress.com/category/leetcode-questions/
Scrape the following -> short and sweet: https://nishmathcs.wordpress.com/category/data-structures-algorithms/page/1/

Post about all string algorithms and hashing types (KMP, Boyer Moore, etc, Rabin Karp)


COMPETITIVE PROGRAMMING GUIDE TO READ: https://cp-algorithms.com/
TOPCODER COMPETITIVE PROGRAMMING GUIDES -> https://www.topcoder.com/community/competitive-programming/tutorials/

REALLY COOL MEDIUM ARTICLE -> https://medium.com/@karangujar43/best-resources-for-competitive-programming-algorithms-and-data-structures-730cb038e11b


TOPICS TO UNDERSTAND: 
        Segment tree (with lazy propagation)
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


THESE ARE HARMAN'S PERSONAL SET OF PARADIGMS:


0)  For problems like parenthesis matching. You can use a stack to solve the matching. But you can also
    do matching by incrementing and decrementing an integer variable. Or you can use colors or 
    other types of INDICATOR VARIABLE TYPE SOLUTIONS that contain meta information on the problem. 

0.05) To solve a difficult 3D problem or 2D problem. Solve the lower dimension first, 
     and then use it to guide your solution for higher dimensions. 
     Such as max area of rectangle of 1's.


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

    



0.27) Memorize 0-1 Knapsack and strategy
      and space efficiency strategy:
    
    # Geeks for Geeks:
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
                    previous_item[w] = item

        cur_weight = capacity
        taken = []
        while cur_weight > 0:
            taken.append(previous_item[cur_weight])
            cur_weight -= previous_item[cur_weight].weight

        return best_value[capacity], taken


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
     USE MATH AND ANALYSIS TO GET THE BEST SOLUTION!



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
        binary searching 2 pointers)

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
    e) Be creative in how you see the DIRECTIONALITY of the solution for a given problem. 

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
        Palindrome Linked List (medium)
        Cycle in a Circular Array (hard)


1.95) Use pointer on the fly construction !!
      Combining running 2 pointers, running 2 container concepts, and space-efficient
      dynamic programming concepts to get O(N) speed, O(1) space except for output container.
      Think in terms of containers to implement fast 2 pointer solution!
      Then think in terms of DP to reduce to a 1 pointer solution!


      which does not count:
        
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



2) Back tracking
    => For permutations, need some intense recursion 
        (recurse on all kids, get all their arrays, and append our chosen element to everyones array, return) 
        and trying all posibilities
    => For combinations, use a binary tree. Make the following choice either: CHOOSE ELEMENT. DONT CHOOSE ELEMENT. 
        Recurse on both cases to get all subsets
    => To get all subsets, count from 0 to 2^n, and use bits to choose elements.

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

2.35) In head recursion , the recursive call, when it happens, comes 
      before other processing in the function (think of it happening at the top, or head, of the function). In tail recursion , it's the opposite—the processing occurs before the recursive call.

2.4) CUT VERTEX AKA ARTICULATION POINT finding:

    '''A recursive function that find articulation points  
    using DFS traversal 
    u --> The vertex to be visited next 
    visited[] --> keeps tract of visited vertices 
    disc[] --> Stores discovery times of visited vertices 
    parent[] --> Stores parent vertices in DFS tree 
    ap[] --> Store articulation points'''
    def APUtil(self,u, visited, ap, parent, low, disc): 
  
        #Count of children in current node  
        children =0
  
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
            elif v != parent[u]:  
                low[u] = min(low[u], disc[v]) 
  
  
    #The function to do DFS traversal. It uses recursive APUtil() 
    def AP(self): 
   
        # Mark all the vertices as not visited  
        # and Initialize parent and visited,  
        # and ap(articulation point) arrays 
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



2.5) CUT EDGE AKA BRIDGE finding. 
    
    '''A recursive function that finds and prints bridges 
    using DFS traversal 
    low[w] is the lowest vertex reachable in a subtree rooted at w

    u --> The vertex to be visited next 
    visited[] --> keeps tract of visited vertices 
    disc[] --> Stores discovery times of visited vertices 
    parent[] --> Stores parent vertices in DFS tree'''
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
  
  
    # DFS based function to find all bridges. It uses recursive 
    # function bridgeUtil() 
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
            for i in xrange(n):
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
                for i in range(n):
                    for j in range(i):
                        A[i][j], A[j][i] = A[j][i], A[i][j]
                for row in A:
                    for j in range(n/2):
                        row[j], row[~j] = row[~j], row[j]


2.57) To find the root nodes in a directed graph (NOT DAG):
      Reverse graph and find nodes with 0 children.
      However, there may not be root nodes!

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



2.7) Common problems solved using DP on broken profile include:

        finding number of ways to fully fill an area (e.g. chessboard/grid) with some figures (e.g. dominoes)
        finding a way to fill an area with minimum number of figures
        finding a partial fill with minimum number of unfilled space (or cells, in case of grid)
        finding a partial fill with the minimum number of figures, such that no more figures can be added
        Problem "Parquet"
        Problem description. Given a grid of size N×M. Find number of ways to 
        fill the grid with figures of size 2×1 (no cell should be left unfilled, 
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
2.7) When you DFS/BACKTRACK, one way to reduce space usage, is using grid itself
     as the visited set, and assigning and reverting it.  
      Additionally, RETURN ASAP. PRUNE, PRUNE PRUNE. 
      Do not aggregrate all the results then return.
      NO UNNECESSARY SEARCHING. Look at Word Search in leet folder. 



2.8) ROLLING HASH USAGE: 
    Consider the string abcd and we have to find the hash values of 
    substrings of this string having length 3 ,i.e.,abc and bcd.
    
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

    H=( c1ak-1 + c2ak-2 + c3ak-3. . . . + cka0 ) % m
    
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
    it is best to operate on index values in your recursion instead of slicing and joining arrays.
    Always operate on the pointers for efficiency purposes.

5) If you need to keep track of a list of values instead of just 1 values 
    such as a list of maxes, instead of 1 max, 
    and pair off them off, use an ordered dictionary! 
    They will keep these values ordered for pairing purposes. 
    pushing and poping an element in an ordered map brings it to the front. 

6)  If you need to do range searches, you need a range tree. 
    if you dont have time to get a range tree, 
    use binary searching as the substitute!
    Also try a segment tree, fenwick tree, etc.

7) if the problem is unsorted, try sorting and if you need to keeping 
    track of indexes, use reverse index map, to do computations. 

7.5) If the problem is already sorted, try binary search. 

8) Do preprocessing work before you start solving problem to improve efficiency

9) Use Counter in python to create a multiset. 

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



11) Know how to write BFS with a deque, and DFS explicitely with a list. 
    Keep tracking of function arguments in tuple for list. 

12) If you need a priority queue, use heapq. Need it for djikistras. 
    Djikstras is general BFS for graphs with different sized edges. 

12.5) Know expand around center to find/count palindroms in a string:
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
        




13) Skip lists are nice

13.5) TOPO SORT -> if you dont know which node to start this from, start from any node.
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

20) B+ Trees + Fenwick Trees


21) 4. Merge Intervals
        The Merge Intervals pattern is an efficient technique to deal with 
        overlapping intervals. In a lot of problems involving intervals, you 
        either need to find overlapping intervals or merge intervals if they overlap. 
        The pattern works like this:
        Given two intervals (‘a’ and ‘b’), there will be six different ways the 
        two intervals can relate to each other: => a consumes b, b consumes a, b after a, a after b, b after a no overlap, a after b no overlap

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
        If the problem asks you to find the missing/duplicate/smallest number in an sorted/rotated array
    
        
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

22.7) Find the start of a loop in a linked list:
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




23) TREE DFS:
    Decide whether to process the current node now (pre-order), 
    or between processing two children (in-order) 
    or after processing both children (post-order).
    Make two recursive calls for both the children 
    of the current node to process them.
    -> You can also just use Tree DFS to process in bfs order

23.5) 2 STACKS == QUEUE TECHNIQUE!
      push onto one, if you pop and empty, dump first one into second one!

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

        Ways to identify the Two Heaps pattern:
        Useful in situations like Priority Queue, Scheduling
        If the problem states that you need to find the smallest/largest/median elements of a set
        Sometimes, useful in problems featuring a binary tree data structure
        Problems featuring
        Find the Median of a Number Stream (medium)


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

27) Top K elements

        Any problem that asks us to find the top/smallest/frequent ‘K’ 
        elements among a given set falls under this pattern.

        The best data structure to keep track of ‘K’ elements is Heap. 
        This pattern will make use of the Heap to solve multiple 
        problems dealing with ‘K’ elements at a time from 
        a set of given elements. The pattern looks like this:

        Insert ‘K’ elements into the min-heap or max-heap based on the problem.

        Iterate through the remaining numbers and if you find one that is 
        larger than what you have in the heap, 
        then remove that number and insert the larger one.

        There is no need for a sorting algorithm because the heap will keep track of the elements for you.
        How to identify the Top ‘K’ Elements pattern:
        If you’re asked to find the top/smallest/frequent ‘K’ elements of a given set
        If you’re asked to sort an array to find an exact element
        Problems featuring Top ‘K’ Elements pattern:
        Top ‘K’ Numbers (easy)
        Top ‘K’ Frequent Numbers (medium)

        # Top K Frequent Elements

        class Solution(object):
            def topKFrequent(self, nums, k):
                """
                :type nums: List[int]
                :type k: int
                :rtype: List[int]
                """

                num_of_items_to_return = k
                m = collections.defaultdict(int)
                
                for i in nums:
                    m[i] += 1

                pq = [] # heapq
                counter = itertools.count()
                
                # entry_finder = {} Used for deleting other elements in heapq!
                
                for k, v in m.items():
                
                    if len(pq) < num_of_items_to_return:
                        count = next(counter)
                        i = [v, count, k] #[priority, count, task]
                        heappush(pq, i)
                    else:
                        top =  pq[0][0] # get priority
                        print("TOP IS", top)

                        if v > top:
                            _ = heappop(pq)
                            
                            
                            count = next(counter)
                            i = [v, count, k] #[priority, count, task]
                            
                            heappush(pq, i)
                            
                return map(lambda x: x[-1], pq)


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

30) Use loop invarients when doing 2 pointer solutions, greedy solutions, etc. to think about, and help
    interviewer realize that your solution works!!!

31) Derieive mathematical relationships between numbers in array, and solve for a solution. Since
    there was a mathematical relationship, xor can prolly be used for speedup. 
    For instance: Find the element that appears once

        Given an array where every element occurs three times, except one element which occurs only once. 

        Soln: Add each number once and multiply the sum by 3, we will get thrice the sum of each 
        element of the array. Store it as thrice_sum. Subtract the sum of the whole array 
        from the thrice_sum and divide the result by 2. The number we get is the required 
        number (which appears once in the array).

32) DP is like traversing a DAG. it can have a parents array, dist, and visited set. SOmetimes you need to backtrack
    to retrieve parents so remember how to do that!!!!. 

33) Do bidirectional BFS search if you know S and T and you are finding the path! 
    (i think its good for early termination in case there is no path)

34) For linked list questions, draw it out. Dont think about it. Then figur eout how you are rearranging the ptrs.
    and how many new variables you need.


35) Linear Algorithms:
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

            3) Finally, root of the MH is the kth smallest element.

            Time complexity of this solution is O(k + (n-k)*Logk)
        METHOD 4(BEST METHOD QUICK SELECT): -> DO MEDIAN OF MEDIANS TO GET O(N) worst time!!!

           The idea is, not to do complete quicksort, but stop at the point where pivot itself is k’th         
            smallest element. Also, not to recur for both left and right sides of pivot, 
            but recur for one of them according to the position of pivot. The worst case time          
            complexity of this method is O(n^2), but it works in O(n) on average.


    Sorting in linear time
        Given array, each int between [0, 100] can we sort in O(n). yeah counting sort.
        What if given arry has range of 32 bit unsigned ints [0, 2^32 -1] => radix sort

    SLiding window
        -> Given an array of n elements, can we find a smallest sub-array size so that the sum of the sub-array is greater 
        than or equal to a constant S in O(n)
        2 pointers both start at index 0. move end pointer 
        to right until you have S. then keep that as current_min_running_length_of_subarray.
        move start pointer to right to remove elements, then fix by extending end pointer if sum falls below S. 
        get new subarrays and update current_min_running_length_of_subarray. 




36) Heapify is cool. Python heapify implementation that is O(N) implemented below: 
    UNDERSTAND IT.
        # Single slash is simple division in python. 2 slashes is floor division in python
        # only the root of the heap actually has depth log2(len(a)). Down at the nodes one above a leaf - 
        # where half the nodes live - a leaf is hit on the first inner-loop iteration.

        def heapify(A):
            for root in xrange(len(A)//2-1, -1, -1):
                rootVal = A[root]
                child = 2*root+1
                while child < len(A):
                    if child+1 < len(A) and A[child] > A[child+1]:
                        child += 1
                    if rootVal <= A[child]:
                        break
                    A[child], A[(child-1)//2] = A[(child-1)//2], A[child]
                    child = child *2 + 1



37) Understand counting sort, radix sort.
        Counting sort is a linear time sorting algorithm that sort in O(n+k) 
        time when elements are in range from 1 to k.        
        What if the elements are in range from 1 to n2? 
        We can’t use counting sort because counting sort will take O(n2) which is worse 
        than comparison based sorting algorithms. Can we sort such an array in linear time?
        Radix Sort is the answer. The idea of Radix Sort is to do digit by digit 
        sort starting from least significant digit to most significant digit. 
        Radix sort uses counting sort as a subroutine to sort.

        The Radix Sort Algorithm
        1) Do following for each digit i where i varies from least significant digit to the most significant digit.
        ………….a) Sort input array using counting sort (or any stable sort) according to the i’th digit.



38) To do post order traversal or inorder traversal 
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


39) When you need to keep a set of running values such as mins, and prev mins, 
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



40) In some questions you can 
    do DFS or BFS from a root node to a specific 
    child node and you end up traversing a tree, 
    either the DFS tree or BFS Tree. 
    HOWEVER, AS AN O(1) space optimization, you might be able 
    to go backward from the child node to the root node,
    and only end up traversing a path rather than a tree!
    FOR INSTANCE THE FOLLOWING PROBLEM:


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

45) Can create queue with 2 stacks


46) Heap
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
    DETECH CYCLE IN UNDIRECTED GRAPH:
    
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

49.5) RANGE MINIMUM QUERY:

    You are given an array A[1..N]. You have to answer incoming queries of the form (L,R), 
    which ask to find the minimum element in array A between positions L and R inclusive.

    RMQ can appear in problems directly or can be applied in some other tasks, e.g. the Lowest Common Ancestor problem.

    Solution
    There are lots of possible approaches and data structures that you can use to solve the RMQ task.

    The ones that are explained on this site are listed below.

    First the approaches that allow modifications to the array between answering queries.

    Sqrt-decomposition - answers each query in O(sqrt(N)), preprocessing done in O(N). Pros: a very simple data structure. Cons: worse complexity.
    
    Segment tree - answers each query in O(logN), preprocessing done in O(N). 
    
                 Pros: good time complexity. Cons: larger amount of code compared to the other data structures.
    Fenwick tree - answers each query in O(logN), preprocessing done in O(NlogN). 
                Pros: the shortest code, good time complexity. 
                Cons: Fenwick tree can only be used for queries with L=1, so it is not applicable to many problems.

    And here are the approaches that only work on static arrays, i.e. it is not possible 
        to change a value in the array without recomputing the complete data structure.

    Sparse Table - answers each query in O(1), preprocessing done in O(NlogN). Pros: simple data structure, excellent time complexity.
    Sqrt Tree - answers queries in O(1), preprocessing done in O(NloglogN). Pros: fast. Cons: Complicated to implement.
    Disjoint Set Union / Arpa's Trick - answers queries in O(1), preprocessing in O(n). 
            Pros: short, fast. Cons: only works if all queries are known in advance, 
            i.e. only supports off-line processing of the queries.
    
    Cartesian Tree and Farach-Colton and Bender algorithm - answers queries in O(1), preprocessing in O(n). 
        Pros: optimal complexity. Cons: large amount of code.
        Note: Preprocessing is the preliminary processing of the given array by building the corresponding data structure for it.

50) Fenwick Tree Structure:
    ->Full binary tree with at least n leaf nodes
    ->kth leaf node stores the value of item k
    
    ->Each internal node(not leaf) stores the sum of value of its children
    
    Main idea: choose the minimal set of nodes whose sum gives the 
    desired value
    -> We will see that
        - at most 1 node is chosen at each level so that 
        the total number of nodes we look at is log(base 2)n
        - and this can be done in O(lgn) time

    Computing Prefix sums:
        Say we want to compute Sum(k)
        Maintain a pointer P which initially points at leaf k.
        Climb the tree using the following procedure:
            If P is pointing to a left child of some node: 
                -> Add the value of P
                -> Set P to the parent node of P's left neighbor
                -> If P has no left neighbor terminate
            Otherwise:
                Set P to the parent node of P
        
        Use an array to implement!

    Updating a value:
    Say we want to do Set(k, x) (set value of leaf k as x)
    -> Start at leaft k, change its val to x
    -> Go to parent, recompute its val. Repeat until you get to root.

    Extension: make the Sum() function work for any interval,
    not just ones that start from item 1
    Can support: Min(i, j), Max(i,j ) (Min/Max element among items i,..j)


51) Lowest Common Ancestor:
    Preprocess tree in O(nlgn) time in order to 
    answer each LCA query in O(lgn) time
    Compute Anc[x][k]  

    Each node stores its depth, as well as the links to every 
    2kth ancestor:
    O(lgn) adiditional storage per node
    - Anc[x][k] denotes the 2kth ancestor of node x
    - Anc[x][0] = x's parent
    - Anc[x][k] = Anc[Anc[x][k-1]][k-1]
    
    Answer query:
    Given two node indices x and y
        Without loss of generality, assume depth(x) ≤ depth(y)
        
        Maintain two pointers p and q, initially pointing at x and y
        If depth(p) < depth(q), bring q to the same depth as p
        – using Anc that we computed before
        
        Now we will assume that depth(p) = depth(q)

        If p and q are the same, return p
        Otherwise, initialize k as ⌈log 2 n⌉ and repeat:
        – If k is 0, return p’s parent node
        – If Anc[p][k] is undefined, or if Anc[p][k] and Anc[q][k]
        point to the same node:
            Decrease k by 1
        – Otherwise:
        
            Set p = Anc[p][k] and q = Anc[q][k] to bring p and q up
            by 2^k levels

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
                
            
            dummy.next.prev = None
            return dummy.next


53) The art of segment trees and monoqueues:


    Previously we saw segment trees.
    That data structure was able to answer the question

    reduce(lambda x,y: operator(x,y), arr[i:j], default)
    and we were able to answer it in O(\log(j - i)) time. 
    Moreover, construction of this data structure only took O(n) time. 
    Moreover, it was very generic as operator and 
    default could assume be any values.

    This obviously had a lot of power, but we can use something a lot 
    simpler if we want to easier problems. 
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
        def push(self, val):
            count = 0
            while self.q and self.op(val, self.q[-1][0]):
                count += 1 + self.q[-1][1]
                self.q.pop()
            self.q.append([val, count])
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
                    
                    # Remove the elements which are  
                    # out of this window 
                    # out of window elements are in the front of the queue. 
                    # indexes are increasing like above -> 2 -> 5
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


#####################################################################################################################
#####################################################################################################################

COOL NOTES PART 1: DYNAMIC PROGRAMMING RECURRENCES EXAMPLES: 
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

################################################################
############################################################
########################################################

COOL NOTES PART 1.6 => DIGIT DP

    How many numbers x are there in the range a to b, where the digit d occurs exactly k times in x? 
    (can solve with combinatorics too.)


    Building a sequence of digits
        Let’s consider the number as a sequence of digits. Let’s name the sequence sq. 
        Initially sq is empty. We’ll try to add new digits from left to right to build the sequence. 
        n each recursive call we’ll place a digit in our current position and 
        will call recursively to add a digit in the next position. 

    Information we need to place a digit at the current position
        Let’s say during the building of the sequence, currently we are at position pos.
        We have already placed some digits in position from 1 to pos-1. So now we are trying to 
        place a digit at current position pos. If we knew the whole sequence 
        we have build so far till position pos-1 then we could 
        easily find out which digits we can place now. But how?

        You can see that, in the sequence sq the left most digit is actually the most significant digit. 
        And the significance get decreased from left to right. So if 
        there exist any position t (1<=t<pos) where sq[t] < b[t] then we can 
        place any digit in our current position. Because the sequence has 
        already become smaller than b no matter which digit we place in the later positions. 
        Note, b[t] means the digit at position t at number b.

        But if there was no t that satisfy that condition then at position pos, 
        we can’t place any digit greater than b[pos]. 
        Because then the number will become larger than b.
        
        using an extra parameter f1(true/false) in our function we can handle that. 
        Whenever we place a digit at position t which is smaller than b[t] 
        we can make f1 = 1 for the next recursive call. So whenever we are at any position later, 
        we don’t actually need the whole sequence. Using the value of f1 
        we can know if the sequence have already become smaller than b.

    Extra condition
        digit d will have to occur exactly k times in sequence sq. We need another parameter cnt. 
        cnt is basically the number of times we have placed digit d so far in 
        our sequence sq. Whenever we place digit d in our sequence sq we just 
        increment cnt in our next recursive call.

        In the base case when we have built the whole sequence we just need to check if 
        cnt is equal to k. If it is then we return 1, 
        otherwise we return 0.

    Final DP States
        If we have understood everything so far then it's easy to see that we need 
        total three states for DP memoization. At which position we are, 
        if the number has already become smaller than b and the frequency of digit d till now.

    Solve for range (a to b)
        Using the above approach we can find the total valid numbers in the range 0 to b. 
        But in the original problem the range was actually a to b. How to handle that? 
        Well, first we can find the result for range 0 to b and then just 
        remove the result for range 0 to a-1. Then what we 
        are left off is actually the result from range a to b.

    How to solve for range a to b in a single recursion?
        In the above approach we used an extra parameter f1 which helped us to make sure 
        the sequence is not getting larger than b. Can’t we do the similar thing so that 
        the sequence does not become smaller than a? Yes of course. For that, 
        we need to maintain an extra parameter f2 which will say if there 
        exist a position t such that sq[t] > a[t]. Depending on the value of 
        f2 we can select the digits in our current position so that the sequence 
        does not become smaller than a. Note: We also have to 
        maintain the condition for f1 parallely so that the sequence remains valid.

        #include <bits/stdc++.h>
        using namespace std;

        vector<int> num;
        int a, b, d, k;
        int DP[12][12][2];
        /// DP[p][c][f] = Number of valid numbers <= b from this state
        /// p = current position from left side (zero based)
        /// c = number of times we have placed the digit d so far
        /// f = the number we are building has already become smaller than b? [0 = no, 1 = yes]

        int call(int pos, int cnt, int f){
            if(cnt > k) return 0;

            if(pos == num.size()){
                if(cnt == k) return 1;
                return 0;
            }

            if(DP[pos][cnt][f] != -1) return DP[pos][cnt][f];
            int res = 0;

            int LMT;

            if(f == 0){
                /// Digits we placed so far matches with the prefix of b
                /// So if we place any digit > num[pos] in the current position, then the number will become greater than b
                LMT = num[pos];
            } else {
                /// The number has already become smaller than b. We can place any digit now.
                LMT = 9;
            }

            /// Try to place all the valid digits such that the number doesn't exceed b
            for(int dgt = 0; dgt<=LMT; dgt++){
                int nf = f;
                int ncnt = cnt;
                if(f == 0 && dgt < LMT) nf = 1; /// The number is getting smaller at this position
                if(dgt == d) ncnt++;
                if(ncnt <= k) res += call(pos+1, ncnt, nf);
            }

            return DP[pos][cnt][f] = res;
        }

        int solve(int b){
            num.clear();
            while(b>0){
                num.push_back(b%10);
                b/=10;
            }
            reverse(num.begin(), num.end());
            /// Stored all the digits of b in num for simplicity

            memset(DP, -1, sizeof(DP));
            int res = call(0, 0, 0);
            return res;
        }

        int main () {

            cin >> a >> b >> d >> k;
            int res = solve(b) - solve(a-1);
            cout << res << endl;

            return 0;
        }



#####################################################################################
####################################################################################3#3
COOL NOTES PART 2: DYANMIC PROGRAMMING WITH DP, CONVEX HULLS, KNUTH OPTIMIZATION


10.6) Divide and Conquer DP:

        Divide and Conquer is a dynamic programming optimization.

        Preconditions
        Some dynamic programming problems have a recurrence of this form:
        dp(i,j) = min k≤j {dp(i−1,k)+C(k,j)}
        where C(k,j) is some cost function.

        Say 1≤i≤n and 1≤j≤m, and evaluating C takes O(1) time. 
        
        Straightforward evaluation of the above recurrence is O(nm^2). 
        
        There are n×m states, and m transitions for each state.

        Let opt(i,j) be the value of k that minimizes the above expression. 
        If opt(i,j) ≤ opt(i,j+1) for all i,j, then we can apply divide-and-conquer DP. 
        This known as the monotonicity condition. 
        The optimal "splitting point" for a fixed i increases as j increases.

        This lets us solve for all states more efficiently. 
        Say we compute opt(i,j) for some fixed i and j. Then for any j′<j we 
        know that opt(i,j′) ≤ opt(i,j). 

        This means when computing opt(i,j′), we don't have to consider as many splitting points!

        To minimize the runtime, we apply the idea behind divide and conquer. 
        First, compute opt(i,n/2). Then, compute opt(i,n/4), knowing that it is less 
        than or equal to opt(i,n/2) and opt(i,3n/4) knowing that it is greater than or 
        equal to opt(i,n/2). By recursively keeping track of the lower and upper bounds on opt, 
        we reach a O(mnlogn) runtime. Each possible value of opt(i,j) only appears in logn different nodes.

        Note that it doesn't matter how "balanced" opt(i,j) is. Across a fixed level,
        each value of k is used at most twice, and there are at most logn levels.

        Generic implementation
        Even though implementation varies based on problem, here's a fairly generic template. 
        The function compute computes one row i of states dp_cur, given the previous row i−1 of states dp_before. 
        It has to be called with compute(0, n-1, 0, n-1).

        int n;
        long long C(int i, int j);
        vector<long long> dp_before(n), dp_cur(n);

        // compute dp_cur[l], ... dp_cur[r] (inclusive)
        void compute(int l, int r, int optl, int optr)
        {
            if (l > r)
                return;
            int mid = (l + r) >> 1;

            pair<long long, int> best = {INF, -1};

            for (int k = optl; k <= min(mid, optr); k++) {
                best = min(best, {dp_before[k] + C(k, mid), k});
            }

            dp_cur[mid] = best.first;
            int opt = best.second;

            compute(l, mid - 1, optl, opt);
            compute(mid + 1, r, opt, optr);
        }

        Things to look out for
        The greatest difficulty with Divide and Conquer DP problems is proving the monotonicity of opt. 
        Many Divide and Conquer DP problems can also be solved with the Convex Hull 
        trick or vice-versa. It is useful to know and understand both!





#############################################################################
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
  
    BINARY SEARCH
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

    INPLACE QUICKSORT:

        def sub_partition(array, start, end, idx_pivot):

            'returns the position where the pivot winds up'
            if not (start <= idx_pivot <= end):
                raise ValueError('idx pivot must be between start and end')

            array[start], array[idx_pivot] = array[idx_pivot], array[start]
            pivot = array[start]
            i = start + 1
            j = start + 1

            while j <= end:
                if array[j] <= pivot:
                    array[j], array[i] = array[i], array[j]
                    i += 1
                j += 1

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

    INPLACE MERGE SORT

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


    MEDIAN OF MEDIANS (THIS IS ALSO KNOWN AS QUICK SELECT, finds the ith smallest element in A):  

        The median-of-medians algorithm is a deterministic linear-time selection algorithm. 
        The algorithm works by dividing a list into sublists and then determines 
        the approximate median in each of the sublists. Then, it takes those medians 
        and puts them into a list and finds the median of that list. It uses that median 
        value as a pivot and compares other elements of the list against the pivot. 
        If an element is less than the pivot value, the element is placed to the left 
        of the pivot, and if the element has a value greater than the pivot, it is 
        placed to the right. The algorithm recurses on the list, honing in on the value it is looking for.

        PYTHON CODE:

        def median_of_medians(A, i):

            #divide A into sublists of len 5
            sublists = [A[j:j+5] for j in range(0, len(A), 5)]
            medians = [sorted(sublist)[len(sublist)/2] for sublist in sublists]
            if len(medians) <= 5:
                pivot = sorted(medians)[len(medians)/2]
            else:
                #the pivot is the median of the medians
                pivot = median_of_medians(medians, len(medians)/2)

            #partitioning step
            low = [j for j in A if j < pivot]
            high = [j for j in A if j > pivot]

            k = len(low)
            if i < k:
                return median_of_medians(low,i)
            elif i > k:
                return median_of_medians(high,i-k-1)
            else: #pivot = k
                return pivot


    MEDIAN OF MEDIANS(QUICKSELECT),  with constant space PARTITION FUNCTION.:

        function partition(list, left, right, pivotIndex)
            pivotValue := list[pivotIndex]
            swap list[pivotIndex] and list[right]  // Move pivot to end
            storeIndex := left
            for i from left to right-1
                if list[i] < pivotValue
                    swap list[storeIndex] and list[i]
                    increment storeIndex
            swap list[right] and list[storeIndex]  // Move pivot to its final place
            return storeIndex

        // Returns the k-th smallest element of list within left..right inclusive
        // (i.e. left <= k <= right).
        // The search space within the array is changing for each round - but the list
        // is still the same size. Thus, k does not need to be updated with each round.
        function select(list, left, right, k)
            if left = right        // If the list contains only one element,
                return list[left]  // return that element
            pivotIndex  := ...     // select a pivotIndex between left and right,
                                    // e.g., left + floor(rand() % (right - left + 1))
            pivotIndex  := partition(list, left, right, pivotIndex)
            // The pivot is in its final sorted position
            if k = pivotIndex
                return list[k]
            else if k < pivotIndex
                return select(list, left, pivotIndex - 1, k)
            else
                return select(list, pivotIndex + 1, right, k)
        
    MEDIAN OF MEDIANS 2 (FULL Implementation)

        def median_of_medians(arr):
            if len(arr) < 6:
                arr = sorted(arr)
                n = len(arr)
                return arr[n/2]
            else:
                list_of_lists = []
                for i in range(0,len(arr), 5):
                    if i + 5 <= len(arr):
                        list_of_lists.append(arr[i:i+5])
                    else:
                        list_of_lists.append(arr[i:])
                medians = []
                for l in list_of_lists:
                    medians.append(sorted(l)[len(l)/2])
                return select_k_smallest(medians, len(medians)/2)
        
        def median_partition(arr, left, right, num):
            random_index = arr.index(num)
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
        
        def select_k_smallest(arr, k):
            if k > len(arr):
                return None
            mid = median_of_medians(arr)
            index = median_partition(arr, 0, len(arr) - 1, mid)
            if index == k - 1:
                return mid
            elif index > k - 1:
                return select_k_smallest(arr[:index], k)
            else:
                return select_k_smallest(arr[index + 1:], k - (index + 1))
        
        def select_k_largest(arr, k):
            if k > len(arr):
                return None
            mid = median_of_medians(arr)
            index = median_partition(arr, 0, len(arr) - 1, mid)
            if index == len(arr) - k:
                return mid
            elif index > len(arr) - k:
                return select_k_largest(arr[:index], k - (len(arr) - index))
            else:
                return select_k_largest(arr[index + 1:], k)



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

    BFS Traversal, Distance And Path Included
        def BFS_distance_path_variation(node):
            q = deque()
            q.append(node)
            visited = {}
            distance = {}
            parent = {}
            distance[node] = 0
            parent[node] = None
            while len(q) > 0:
                current = q.popleft()
                if current not in visited:
                    visited[current] = 1:
                    for neighbor in current.neighbors:
                        if neigbor not in visited:
                            q.append(neighbor)
                            if neighbor not in distance:
                                distance[neighbor] = distance[current] + 1
                                parent[neighbor] = current
                            else:
                                '''
                                TWO TERNARYS BELOW
                                '''
                                parent[neighbor] = current \
                                if distance[current] + 1 < distance[neighbor] else \
                                parent[neighbor]
                                distance[neighbor] = distance[current] + 1 if \
                                distance[current] + 1 < distance[neighbor] else \
                                distance[neighbor]

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

#########################################################################
########################################################################3
Z algorithm (https://www.hackerearth.com/practice/algorithms/string-algorithm/z-algorithm/tutorial/)

COOL NOTES: Z ALGORITHM FOR STRINGS:


    The Z-function for a string S of length N is an array of length N 
    where the i th element is equal to the greatest number of 
    characters starting from the position i 
    that coincide with the first characters of S.

    In other words, z[i] is the length 
    of the longest common prefix between S 
    and the suffix of S starting at i. We assume 0-based indexes; 
    that is, the first character of S has index 0 and the last one has index N-1.

    The first element of Z-functions, z[0], is generally not 
    well-defined. In this article we will assume it is zero.

    z [ 0 ] = 0

    This article presents an algorithm for calculating the Z-function 
    in O(N) time, as well as various of its applications.


    Examples

    For example, here are the values of the Z-function computed for different strings:

    s = 'aaaaa'
    Z[0]	Z[1]	Z[2]	Z[3]	Z[4]
    0	      4	      3	      2	      1

    s = 'aaabaab'
    Z[0]	Z[1]	Z[2]	Z[3]	Z[4]	Z[5]	Z[6]
    0	      2	      1	      0	      2	      1	      0

    s = 'abacaba'
    Z[0]	Z[1]	Z[2]	Z[3]	Z[4]	Z[5]	Z[6]
    0	      0	      1	      0	      3       0	      1

    Trivial algorithm

    The formal definition can be represented in the following elementary implementation.

    vector<int> z_function_trivial(string s) 
    {
        int n = (int) s.length();
        vector<int> z(n);
        for (int i = 1; i < n; ++i)
            while (i + z[i] < n && s[z[i]] == s[i + z[i]])
                ++z[i];
        return z;
    }

    We just iterate through every position and update for each one of them, 
    starting from and incrementing it as long as we do not 
    find a mismatch (and as long as we do not reach the end of the line).

    Efficient algorithm

    The idea is to maintain an interval [L, R] which is the interval with max R
    such that [L,R] is prefix substring (substring which is also prefix). 

    Steps for maintaining this interval are as follows – 

    1) If i > R then there is no prefix substring that starts before i and 
    ends after i, so we reset L and R and compute new [L,R] by comparing 
    str[0..] to str[i..] and get Z[i] (= R-L+1).

    2) If i <= R then let K = i-L,  now Z[i] >= min(Z[K], R-i+1)  because 
    str[i..] matches with str[K..] for atleast R-i+1 characters (they are in
    [L,R] interval which we know is a prefix substring).     
    Now two sub cases arise – 
        a) If Z[K] < R-i+1  then there is no prefix substring starting at 
            str[i] (otherwise Z[K] would be larger)  so  Z[i] = Z[K]  and 
            interval [L,R] remains same.
        b) If Z[K] >= R-i+1 then it is possible to extend the [L,R] interval
            thus we will set L as i and start matching from str[R]  onwards  and
            get new R then we will update interval [L,R] and calculate Z[i] (=R-L+1).

            
    Implementation
    // returns array z[] where z[i] is z-function of s[i]
    int[] zFucntion(String s) {
        int n = s.length();
        int z[] = new int[n];
        int R = 0;
        int L = 0;
        for(int i = 1; i < n; i++) {
            z[i] = 0;
            if (R > i) {
                z[i] = Math.min(R - i, z[i - L]);
            }
            while (i + z[i] < n && s.charAt(i+z[i]) == s.charAt(z[i])) {
                z[i]++;
            }
            if (i + z[i] > R) {
                L = i;
                R = i + z[i];
            }
        }
        z[0] = n;
        return z;
    }

    Complexity
    Worst case time complexity: Θ(N)
    Average case time complexity: Θ(N)
    Best case time complexity: Θ(N)
    Space complexity: Θ(log N)

    Applications

    Applications of Z algorithms are as follows:
    Finding all occurrences of the pattern P inside the text T in O(length(T) + length(P))
    Counting the number of distinct substrings of a string S in O(1)
    Finding a string T of shortest length such that S can be represented as a concatenation of one or more copies of T

#################################################################################
#################################################################################
STRING ALGORITHMS:
    PLEASE COVER:
        Boyer moore good character heuristic/bad char heuristic
        Aho-Corasick Algorithm for Pattern Searching
        Suffix Tree/Suffix Array
        Manachars algorithm (https://www.hackerearth.com/practice/algorithms/string-algorithm/manachars-algorithm/tutorial/)

    Rabin Karp
        So Rabin Karp algorithm needs to calculate 
        hash values for following strings.
        1) Pattern itself.
        2) All the substrings of text of length m.

        Since we need to efficiently calculate hash values for all the substrings 
        of size m of text, we must have a hash function which has following property.
        Hash at the next shift must be efficiently computable from the current hash 
        value and next character in text or we can say hash(txt[s+1 .. s+m]) must 
        be efficiently computable from hash(txt[s .. s+m-1]) and txt[s+m] i.e., 
        hash(txt[s+1 .. s+m])= rehash(txt[s+m], hash(txt[s .. s+m-1])) and rehash must be O(1) operation.
        
        To do rehashing, we need to take off the most significant digit
        and add the new least significant digit for in hash value. Rehashing is done using the following formula.

        hash( txt[s+1 .. s+m] ) = ( d ( hash( txt[s .. s+m-1]) – txt[s]*h ) + txt[s + m] ) mod q

        hash( txt[s .. s+m-1] ) : Hash value at shift s.
        hash( txt[s+1 .. s+m] ) : Hash value at next shift (or shift s+1)
        d: Number of characters in the alphabet
        q: A prime number
        h: d^(m-1)

        How does above expression work?

        This is simple mathematics, we compute decimal value of current window from previous window.
        For example pattern length is 3 and string is “23456”
        You compute the value of first window (which is “234”) as 234.
        How how will you compute value of next window “345”? You will do (234 – 2*100)*10 + 5 and get 345.


        # Rabin Karp Algorithm given in CLRS book 
        # d is the number of characters in the input alphabet 
        d = 256
        
        # pat  -> pattern 
        # txt  -> text 
        # q    -> A prime number 
        
        def search(pat, txt, q): 
            M = len(pat) 
            N = len(txt) 
            i = 0
            j = 0
            p = 0    # hash value for pattern 
            t = 0    # hash value for txt 
            h = 1
        
            # The value of h would be "pow(d, M-1)%q" 
            for i in xrange(M-1): 
                h = (h*d)%q 
        
            # Calculate the hash value of pattern and first window 
            # of text 
            for i in xrange(M): 
                p = (d*p + ord(pat[i]))%q 
                t = (d*t + ord(txt[i]))%q 
        
            # Slide the pattern over text one by one 
            for i in xrange(N-M+1): 
                # Check the hash values of current window of text and 
                # pattern if the hash values match then only check 
                # for characters on by one 
                if p==t: 
                    # Check for characters one by one 
                    for j in xrange(M): 
                        if txt[i+j] != pat[j]: 
                            break
        
                    j+=1
                    # if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1] 
                    if j==M: 
                        print "Pattern found at index " + str(i) 
        
                # Calculate hash value for next window of text: Remove 
                # leading digit, add trailing digit 
                if i < N-M: 
                    t = (d*(t-ord(txt[i])*h) + ord(txt[i+M]))%q 
        
                    # We might get negative values of t, converting it to 
                    # positive 
                    if t < 0: 
                        t = t+q 
        
        # Driver program to test the above function 
        txt = "GEEKS FOR GEEKS"
        pat = "GEEK"
        q = 101 # A prime number 
        search(pat,txt,q) 
        
        # This code is contributed by Bhavya Jain 



    -> Code regex

    isSubstring(), KMP Algorithm
        
        A linear time (!) algorithm that solves the string matching
        problem by preprocessing P in Θ(m) time
        – Main idea is to skip some comparisons by using the previous
        comparison result
        
        LPS[] that will hold the longest prefix suffix  
        
        Uses an auxiliary array π that is defined as the following:
        – π[i] is the largest integer smaller than i such that P 1 . . . P π[i] is
        a suffix of P 1 . . . P i

        Examples:

        Pattern: a a a a a
        LSP    : 0 1 2 3 4

        Pattern: a b a b a b
        LSP    : 0 0 1 2 3 4

        Pattern: a b a c a b a b
        LSP    : 0 0 1 0 1 2 3 2

        Pattern: a a a b a a a a a b
        LSP    : 0 1 2 0 1 2 3 3 3 4

        txt[] = "AAAAABAAABA" 
        pat[] = "AAAA"
        lps[] = {0, 1, 2, 3} 

        i = 0, j = 0
        txt[] = "AAAAABAAABA" 
        pat[] = "AAAA"
        txt[i] and pat[j] match, do i++, j++

        i = 1, j = 1
        txt[] = "AAAAABAAABA" 
        pat[] = "AAAA"
        txt[i] and pat[j] match, do i++, j++

        i = 2, j = 2
        txt[] = "AAAAABAAABA" 
        pat[] = "AAAA"
        pat[i] and pat[j] match, do i++, j++

        i = 3, j = 3
        txt[] = "AAAAABAAABA" 
        pat[] = "AAAA"
        txt[i] and pat[j] match, do i++, j++

        i = 4, j = 4
        Since j == M, print pattern found and reset j,
        j = lps[j-1] = lps[3] = 3

        Here unlike Naive algorithm, we do not match first three 
        characters of this window. Value of lps[j-1] (in above 
        step) gave us index of next character to match.
        i = 4, j = 3
        txt[] = "AAAAABAAABA" 
        pat[] =  "AAAA"
        txt[i] and pat[j] match, do i++, j++

        i = 5, j = 4
        Since j == M, print pattern found and reset j,
        j = lps[j-1] = lps[3] = 3

        Again unlike Naive algorithm, we do not match first three 
        characters of this window. Value of lps[j-1] (in above 
        step) gave us index of next character to match.
        i = 5, j = 3
        txt[] = "AAAAABAAABA" 
        pat[] =   "AAAA"
        txt[i] and pat[j] do NOT match and j > 0, change only j
        j = lps[j-1] = lps[2] = 2

        i = 5, j = 2
        txt[] = "AAAAABAAABA" 
        pat[] =    "AAAA"
        txt[i] and pat[j] do NOT match and j > 0, change only j
        j = lps[j-1] = lps[1] = 1 

        i = 5, j = 1
        txt[] = "AAAAABAAABA" 
        pat[] =     "AAAA"
        txt[i] and pat[j] do NOT match and j > 0, change only j
        j = lps[j-1] = lps[0] = 0

        i = 5, j = 0
        txt[] = "AAAAABAAABA" 
        pat[] =      "AAAA"
        txt[i] and pat[j] do NOT match and j is 0, we do i++.

        i = 6, j = 0
        txt[] = "AAAAABAAABA" 
        pat[] =       "AAAA"
        txt[i] and pat[j] match, do i++ and j++

        i = 7, j = 1
        txt[] = "AAAAABAAABA" 
        pat[] =       "AAAA"
        txt[i] and pat[j] match, do i++ and j++

        We continue this way...

        def KMPSearch(pat, txt): 
            M = len(pat) 
            N = len(txt) 
        
            # create lps[] that will hold the longest prefix suffix  
            # values for pattern 
            lps = [0]*M 
            j = 0 # index for pat[] 
        
            # Preprocess the pattern (calculate lps[] array) 
            computeLPSArray(pat, M, lps) 
        
            i = 0 # index for txt[] 
            while i < N: 
                if pat[j] == txt[i]: 
                    i += 1
                    j += 1
        
                if j == M: 
                    print "Found pattern at index " + str(i-j) 
                    j = lps[j-1] 
        
                # mismatch after j matches 
                elif i < N and pat[j] != txt[i]: 
                    # Do not match lps[0..lps[j-1]] characters, 
                    # they will match anyway 
                    if j != 0: 
                        j = lps[j-1] 
                    else: 
                        i += 1
        
        def computeLPSArray(pat, M, lps): 
            len = 0 # length of the previous longest prefix suffix 
        
            lps[0] # lps[0] is always 0 
            i = 1
        
            # the loop calculates lps[i] for i = 1 to M-1 
            while i < M: 
                if pat[i]== pat[len]: 
                    len += 1
                    lps[i] = len
                    i += 1
                else: 
                    # This is tricky. Consider the example. 
                    # AAACAAAA and i = 7. The idea is similar  
                    # to search step. 
                    if len != 0: 
                        len = lps[len-1] 
        
                        # Also, note that we do not increment i here 
                    else: 
                        lps[i] = 0
                        i += 1
        
        txt = "ABABDABACDABABCABAB"
        pat = "ABABCABAB"
        KMPSearch(pat, txt) 



##########################################
COOL NOTES PART -3: NETWORK FLOW Tutorial: maxflow and mincut
    COMPUTING MAX FLOW:
        Given directed graph, each edge e assocaited with 
        its capacity c(e) > 0. Two special nodes source s and sink t. 

        Problem: Maximize total amount of flow from s to t subject to 2 constraints:

        1) Flow on edge e doesnt exceed c(e)
        2) For every node other than s,t, incoming flow is equal to outgoing.

        Alternate formulation: we want to remove some edges from
        graph such that after removing the edges, there is no path from
        s to t. 
        The cost of removing e is equal to its capacity, c(e)
        The min cut problem is to find a cut with minimum total cost.

        THRM: MAXIMUM FLOW = MINIMUM CUT

        Flow decomposition: any valid flow can be decomposed into flow 
        paths and circulations.

        Ford-Fulkerson Algorithm: Max flow algo. 
        Main idea: find valid flow paths until there is none left, and 
        add them up.

            The intuition goes like this: as long as there is a path from the 
            source to the sink that can take some flow the entire way, we send it. 
            This path is called an augmenting path. We keep doing this until there 
            are no more augmenting paths. In the image above, we could start by 
            sending 2 cars along the topmost path (because only 2 cars can get 
            through the last portion). Then we might send 3 cars along the bottom 
            path for a total of 5 cars. Finally, we can send 2 more cars along the 
            top path for two edges, send them down to bottom path and through to the 
            sink. The total number of cars sent is now 7, and it is the maximum flow.

            Simplest algo:

            Set f total = 0
            Repeat until there is no path from s to t:
                Run DFS from s to find a flow path to t
                Let f be the minimum capacity value on the path
                Add f to f total
                For each edge u → v on the path:
                    Decrease c(u → v) by f
                    Increase c(v → u) by f

            SIMPLIFLIED ALGO:
                initialize flow to 0
                path = findAugmentingPath(G, s, t)
                while path exists:
                    augment flow along path                 #This is purposefully ambiguous for now
                    G_f = createResidualGraph()
                    path = findAugmentingPath(G_f, s, t)
                return flow
            
            More Explained version:
                flow = 0
                for each edge (u, v) in G:
                    flow(u, v) = 0
                while there is a path, p, from s -> t in residual network G_f:
                    residual_capacity(p) = min(residual_capacity(u, v) : for (u, v) in p)
                    flow = flow + residual_capacity(p)
                    for each edge (u, v) in p:
                        if (u, v) is a forward edge:
                            flow(u, v) = flow(u, v) + residual_capacity(p)
                        else:
                            flow(u, v) = flow(u, v) - residual_capacity(p)
                return flow

        Residual graphs are an important middle step in calculating the maximum flow. 
        As noted in the pseudo-code, they are calculated at every step 
        so that augmenting paths can be found from the source to the sink.

        When a residual graph, G_f is created, edges can be created 
        that go in the opposite direction when compared to the original graph. 
        An edge is a 'forward edge' if the edge existed in the original graph, G. 
        If it is a reversal of an original edge, it is called a 'backwards edge.'

        Residual capacity is defined as the new capacity after a given flow has been taken away. 
        In other words, for a given edge (u, v), the residual capacity, c_f is defined as
        1) c_f(u, v) = c(u,v) - f(u, v)

        However, there must also be a residual capacity for the reverse edge as well. 
        The max-flow min-cut theorem states that flow must be preserved 
        in a network. So, the following equality always holds:
        2) f(u, v) = -f(v, u)

        residual capacities are used to make a residual network, G_f
        1) and 2) allow you to operate on residual graph.

        In the forward direction, the edges now have a residual capacity 
        equal to c_f(u, v) = c(u, v) - f(u, v)
        The flow is equal to 2, so the residual capacity of (S, A) and (A, B) is reduced to 2, 
        while the edge (B, T) has a residual capacity of 0.

        In the backward direction, the edges now have a residual capacity equal to 
        c_f(v, u) = c(v, u) - f(v, u) 
        Because of flow preservation, this can be written as c_f(v, u) = c(v, u) + f(u, v)

        And since the capacity of those backward edges was initially 0, 
        all of the backward edges (T, B), (B, A), and (A, S) 
        now have a residual capacity of 2.

        When a new residual graph is constructed with these new edges, 
        any edges with a residual capacity of 0—like (B, T)—are not included. 
        Add all the backward edges to the residual graph! update all the forward edges!
        keep adding augmenting paths until. There are not more paths from the 
        source to the sink, so there can be no more augmenting paths. 

    COMPUTING MIN CUT:

        1) Run Ford-Fulkerson algorithm and consider the final residual graph.

        2) Find the set of vertices that are reachable 
        from the source in the residual graph.

        3) All edges which are from a reachable vertex 
        to non-reachable vertex are minimum cut edges. Print all such edges.




############################################
#########################################3#####

COOL NOTES PART 5: FENWICK TREES: (A VARIENT OF SEGMENT TREES)
        '''
        We have an array arr[0 . . . n-1]. We would like to
        1 Compute the sum of the first i-th elements.
        2 Modify the value of a specified element of the array arr[i] = x where 0 <= i <= n-1.
        Could we perform both the query and update operations in O(log n) time? 
        One efficient solution is to use Segment Tree that performs both operations in O(Logn) time.

        An alternative solution is Binary Indexed Tree, 
        which also achieves O(Logn) time complexity for both operations. 
        Compared with Segment Tree, Binary Indexed Tree 
        requires less space and is easier to implement..
        '''

        # Python implementation of Binary Indexed Tree 
    
        # Returns sum of arr[0..index]. This function assumes 
        # that the array is preprocessed and partial sums of 
        # array elements are stored in BITree[]. 

        def getsum(BITTree,i): 
            s = 0 #initialize result 
        
            # index in BITree[] is 1 more than the index in arr[] 
            i = i+1
        
            # Traverse ancestors of BITree[index] 
            while i > 0: 
        
                # Add current element of BITree to sum 
                s += BITTree[i] 
        
                # Move index to parent node in getSum View 
                i -= i & (-i) 
            return s 
        
        # Updates a node in Binary Index Tree (BITree) at given index 
        # in BITree. The given value 'val' is added to BITree[i] and 
        # all of its ancestors in tree. 
        def updatebit(BITTree , n , i ,v): 
        
            # index in BITree[] is 1 more than the index in arr[] 
            i += 1
        
            # Traverse all ancestors and add 'val' 
            while i <= n: 
        
                # Add 'val' to current node of BI Tree 
                BITTree[i] += v 
        
                # Update index to that of parent in update View 
                i += i & (-i) 
        
        
        # Constructs and returns a Binary Indexed Tree for given 
        # array of size n. 
        def construct(arr, n): 
        
            # Create and initialize BITree[] as 0 
            BITTree = [0]*(n+1) 
        
            # Store the actual values in BITree[] using update() 
            for i in range(n): 
                updatebit(BITTree, n, i, arr[i]) 

            return BITTree 
        
        
        # Driver code to test above methods 
        freq = [2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9] 
        BITTree = construct(freq,len(freq)) 
        print("Sum of elements in arr[0..5] is " + str(getsum(BITTree,5))) 
        freq[3] += 6
        updatebit(BITTree, len(freq), 3, 6) 
        print("Sum of elements in arr[0..5]"+
                            " after update is " + str(getsum(BITTree,5))) 
    

###################################################################################################################

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
###############################################################
################################################################3
COOL NOTES PART 6.5
    BITMASK OPERATIONS FOR BITMASP DP AND HANDLING SETS. 

    1. Representation. You know that shit cause you in CS. 

    2. To multiply/divide an integer by 2: 
        We only need to shift the bits in the integer left/right, respectively.
        Notice that the truncation in the shift right operation automatically rounds the division-by-2 down,
        e.g. 17/2  = 8.

        For example:         A = 34 (base 10)                  = 100010 (base 2)
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


    Dijkstra
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


    Kruskal ------------------------------------------
        In this algorithm, first we sort the edges in ascending order of 
        their weight in an array of edges.

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

    Prim
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
        As Dijkstra you can use std :: priority_queue instead of std :: set.

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
        f = depth_first_search_rec(G)[2]
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


