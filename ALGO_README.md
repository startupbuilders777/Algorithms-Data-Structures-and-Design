
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
                        if key in window:
                            existing_start, existing_end = window[key]
                            best_start = min(existing_start, time)
                            best_end = max(end, existing_end)
                            window[key] = ((best_start, best_end))
                        else:
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
    count += len(differences)
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
                nxt = node.next
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


1)  For problems like parenthesis matching. You can use a stack to solve the matching. But you can also
    do matching by incrementing and decrementing an integer variable. Or you can use colors or 
    other types of INDICATOR VARIABLE TYPE SOLUTIONS that contain meta information on the problem. 
    Also remember that as you see each element, you can push multiple times to stack, not just once
    in case u need to keep count of something before a pop occurs. 
    
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

2) Back tracking
    => For permutations, need some intense recursion 
        (recurse on all kids, get all their arrays, and append our chosen element to everyones array, return) 
        and trying all posibilities
    => For combinations, use a binary tree. Make the following choice either: CHOOSE ELEMENT. DONT CHOOSE ELEMENT. 
        Recurse on both cases to get all subsets
    => To get all subsets, count from 0 to 2^n, and use bits to choose elements.
    => When doing DP, check to see if you are dealing with permutations or combinations type solutions, and 
        adjust your DP CAREFULLY ACCORDING TO THAT -> AKA CHECK COIN CHANGE 2

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






2.31) DFS ANALYSIS START AND END TIMES!
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
    Back edge       | start[u] > start[v] | end[u] < end[v]
    Forward edge    | start[u] < start[v] | end[u] > end[v]
    Cross edge      | start[u] > start[v] | end[u] > end[v]

2.32) Construct the Rooted Tree by using start and finish time of its 
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

2.39) Articulation points and Biconnected graphs:
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

    TO REMEMBER HOW TO DO BOTTOM PROBLEM, FOR DP REMEMBER THAT YOU ARE FILLING 
    A 2D GRID. HOW SHOULD THE GRID BE FILLED WITH PENCIL AND PAPER? 
    THATS HOW YOU FIGURE IT OUT 
    (aka what are the entries in the map if you did topdown?)
    
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


1)  Know how to write BFS with a deque, and DFS explicitely with a list. 
    Keep tracking of function arguments in tuple for list. 

2)  If you need a priority queue, use heapq. Need it for djikistras. 
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
        -> CAN BE SOLVED IN O(N) WITH BUCKET SORT, AND QUICK SELECT. CHECK IT OUT
        -> TOP K MOST FREQUENT ELEMENTS QUESTION TO SEE THIS. 

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




1)  Use loop invarients when doing 2 pointer solutions, greedy solutions, etc. to think about, and help
    interviewer realize that your solution works!!!

2)  Derieive mathematical relationships between numbers in array, and solve for a solution. Since
    there was a mathematical relationship, xor can prolly be used for speedup. 
    For instance: Find the element that appears once

        Given an array where every element occurs three times, except one element which occurs only once. 

        Soln: Add each number once and multiply the sum by 3, we will get thrice the sum of each 
        element of the array. Store it as thrice_sum. Subtract the sum of the whole array 
        from the thrice_sum and divide the result by 2. The number we get is the required 
        number (which appears once in the array).
        How do we add each number once though? we cant use a set. 
        XOr? wtf?

3)  DP is like traversing a DAG. it can have a parents array, dist, and visited set. SOmetimes you need to backtrack
    to retrieve parents so remember how to do that!!!!. 

4)  Do bidirectional BFS search if you know S and T and you are finding the path! 
    (i think its good for early termination in case there is no path)

5)  For linked list questions, draw it out. Dont think about it. Then figur eout how you are rearranging the ptrs.
    and how many new variables you need. ALSO USE DUMMY POINTERS to not deal with modifying head pointer case. 


6)  Linear Algorithms:
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




7)  Heapify is cool. Python heapify implementation that is O(N) implemented below: 
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

8)  Understand counting sort, radix sort.
        Counting sort is a linear time sorting algorithm that sort in O(n+k) 
        time when elements are in range from 1 to k.        
        What if the elements are in range from 1 to n2? 
        We can’t use counting sort because counting sort will take O(n2) which is worse 
        than comparison based sorting algorithms. Can we sort such an array in linear time?
        Radix Sort is the answer. The idea of Radix Sort is to do digit by digit 
        sort starting from least significant digit to most significant digit. 
        Radix sort uses counting sort as a subroutine to sort.
        Look at section below for impls.


9)  To do post order traversal or inorder traversal 
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


10) When you need to keep a set of running values such as mins, and prev mins, 
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


11) In some questions you can 
    do DFS or BFS from a root node to a specific 
    child node and you end up traversing a tree, 
    either the DFS tree or BFS Tree. 
    HOWEVER, AS AN O(1) space optimization, you might be able 
    to go backward from the child node to the root node,
    and only end up traversing a path rather than a tree!
    Go up the tree for SPEED. 


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
       Number of pairs is 3 x 2  _ _ <- using space theory. 
       Lets binary search the space. 
       min -> difference between 1st and 2nd pair.
           -> largest diff -> 1st and last pair

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
        Count is 4-0-1 = 3 pairs
        
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
                count += j - i - 1  # count pairs
                i += 1  # move slow pointer
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




###################################################################################
###################################################################################
COOL NOTES PART 0.90: DYNAMIC PROGRAMMING PATTERNS, ILLUSTRATIONS, AND EXAMPLES: 

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


