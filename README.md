
THESE ARE HARMANS PERSONAL SET OF PARADIGMS:
0) Branch and Bound algos vs backtracking algos:

    Backtracking
    [1] It is used to find all possible solutions available to the problem.
    [2] It traverse tree by DFS(Depth First Search).
    [3] It realizes that it has made a bad choice & undoes the last choice by backing up.
    [4] It search the state space tree until it found a solution.
    [5] It involves feasibility function.

    Branch-and-Bound
    [1] It is used to solve optimization problem.
    [2] It may traverse the tree in any manner, DFS or BFS.
    [3] It realizes that it already has a better optimal solution 
        that the pre-solution leads to so it abandons that pre-solution.
    [4] It completely searches the state space tree to get optimal solution.
    [5] It involves bounding function.

    Backtracking

    Backtracking is a general algorithm for finding all (or some) solutions to some computational problems, notably constraint satisfaction problems, that incrementally builds candidates to the solutions, and abandons each partial candidate c ("backtracks") as soon as it determines that c cannot possibly be completed to a valid solution.
    It enumerates a set of partial candidates that, in principle, could be completed in various ways to give all the possible solutions to the given problem. The completion is done incrementally, by a sequence of candidate extension steps.
    Conceptually, the partial candidates are represented as the nodes of a tree structure, the potential search tree. Each partial candidate is the parent of the candidates that differ from it by a single extension step, the leaves of the tree are the partial candidates that cannot be extended any further.
    It traverses this search tree recursively, from the root down, in depth-first order (DFS). 
    It realizes that it has made a bad choice & undoes the last choice by backing up.
    For more details: Sanjiv Bhatia's presentation on Backtracking for UMSL.

    Branch And Bound

    A branch-and-bound algorithm consists of a systematic enumeration of candidate solutions by means of state space search: the set of candidate solutions is thought of as forming a rooted tree with the full set at the root.
    The algorithm explores branches of this tree, which represent subsets of the solution set. Before enumerating the candidate solutions of a branch, the branch is checked against upper and lower estimated bounds on the optimal solution, and is discarded if it cannot produce a better solution than the best one found so far by the algorithm.
    It may traverse the tree in any following manner:
        BFS (Breath First Search) or (FIFO) Branch and Bound
        D-Search or (LIFO) Branch and Bound
        Least Count Search or (LC) Branch and Bound



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


1.5) LOOK AT PROBLEM IN ALL POSSIBLE DIRECTIONS to apply your techniques, whether its 2 pointer, 
    sliding window, or Dynamic programm
    a) think about left to right
    b) right to left
    c) 2 pointer on either side and you close into the middle
    d) 2 pointers, one that traverses even indexes, and the other that traverses odd indexes
    e) Be creative in how you see the DIRECTIONALITY of the solution for a given problem. 


1.7) Sliding window: Common problems you use the sliding window pattern with:
        -> Maximum sum subarray of size ‘K’ (easy)
        -> Longest substring with ‘K’ distinct characters (medium)
        -> String anagrams (hard)
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
    -> Cycle detection -> use visited map
    -> shortest path = BFS
    -> Do parent stuff with parents map (such as common ancestors).
    -> Cool graph techniques is coloring nodes, or flagging nodes 
        if you are trying to get multiple paths in graph and looking for path intersections. 
    -> To do topological sort, USE DFS and get start end times. also can count in-nodes and out-nodes to sort them !
    -> Reverse the graph to get the shortest path starting from all other nodes to your node!

    -> Sometimes in a problem, you cant bfs/dfs once. you need to bfs/dfs every vertex!
    -> Minimum spanning tree -> use prims algorithm or kruskals algorithm
    -> Find strongly connected components => use kosarju's algo which does dfs on graph and the reverse of the graph from a vertex.


2.6) LRU Cache learnings and techniques=>
    Circular Doubly linked lists are better than doubly linked lists if you set up dummy nodes
    so you dont have to deal with edge cases regarding changing front and back pointers
    -> With doubly linked lists and maps, You can remove any node in O(1) time as well as append to front and back in O(1) time 
       which enables alot of efficiency

    -> You can also use just an ordered map for this question to solve it fast!! 
    (pop items and put them back in to bring them to the front technique to do LRU)

4) To do things inplace, such as inplace quick sort, or even binary search, 
    it is best to operate on index values in your recursion instead of slicing and joining arrays.
    Always operate on the pointers for efficiency purposes.

5) If you need to keep track of a list of values instead of just 1 values such as a list of maxes, instead of 1 max, 
    and pair off them off, use an ordered dictionary! 
    They will keep these values ordered for pairing purposes. 
    pushing and poping an element in an ordered map brings it to the front. 

6)  If you need to do range searches, you need a range tree. 
    if you dont have time to get a range tree, 
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


11) Know how to write BFS with a deque, and DFS explicitely with a list. 
    Keep tracking of function arguments in tuple for list. 

12) If you need a priority queue, use heapq. Need it for djikistras. 
    Djikstras is general BFS for graphs with different sized edges. 


13) Skip lists are nice

14) Use stacks/queues to take advantage of push/pop structure in problems 
    such as parentheses problems. Or valid expression problems.

15) When you have a problem and it gives you a Binary search tree, 
    make sure to exploit that structure and not treat it as a normal
    binary tree!!!


16) Diviide and conquer

17) Greedy algorithms => requires smart way of thinking about things


18) Bit magic -> Entire section below

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
        Problems featuring cyclic sort pattern:
        Find the Missing Number (easy)
        Find the Smallest Missing Positive Number (medium)


22) Know in-place reverse linked list (MEMORIZE)
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


41) FORD FULKERSON ALGORITHM PYTHON:
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

        A Binary heap is by definition a complete binary tree ,that is, all levels of the       tree, except possibly the last one (deepest) are fully filled, and, if the last     level of the tree is not complete, the nodes of that level are filled from left to        right.
        
        It is by definition that it is never unbalanced. The maximum difference in balance      of the two subtrees is 1, when the last level is partially filled with nodes only       in the left subtree.

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

#############################################
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


############################################
COOL NOTES PART -1: SORTING, SEARCHING, Quick selecting

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



BUCKET SORT:

Bucket Sort

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

###############################################
COOL NOTES PART 0: String matching algorithms

-> Rabin Karp
-> Code regex



###################################################33

Cool Notes Part 0.5: Sliding Window with a deque

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
                    
                # Process rest of the elements, i.e.  
                # from arr[k] to arr[n-1] 
                for i in range(k, n): 
                    
                    # The element at the front of the 
                    # queue is the largest element of 
                    # previous window, so print it 
                    print(str(arr[Qi[0]]) + " ", end = "") 
                    
                    # Remove the elements which are  
                    # out of this window 
                    while Qi and Qi[0] <= i-k: 
                        
                        # remove from front of deque 
                        Qi.popleft()  
                    
                    # Remove all elements smaller than 
                    # the currently being added element  
                    # (Remove useless elements) 
                    while Qi and arr[i] >= arr[Qi[-1]] : 
                        Qi.pop() 
                    
                    # Add current element at the rear of Qi 
                    Qi.append(i) 
                
                # Print the maximum element of last window 
                print(str(arr[Qi[0]])) 
        
        -> YOU CAN ALSO ANSWER THIS QUESTION WITH A SEGMENT TREE. 



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
    
    for i in range(1, n):
        for j in range(1, n):
            if(x[i] == y[j])
                D[i, j] = D[i-1, j-1] + 1
            else:
                D[i, j] = max(D[i-1, j], D[i, j-1])
    
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


#######################################################################################################################

-> Tries
-> Fibonacci heaps
-> Splay trees

-> Range Tree
-> Balanced BSTs -> AVL/REDBLACK, Find a python one. 

