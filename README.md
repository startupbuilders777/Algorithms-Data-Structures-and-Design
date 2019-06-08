

THESE ARE HARMANS PARADIGMS:

1) Exploit problem structure

    a) This means using a sliding window
    b) This means keeping track of running values, and updating permanent values such as 
        keeping tracking of curr_max_from_left to update overall_max when you are running through an array
        -> running variables, running maps, running sets
    c) Use 2 pointer solutions. Two pointers can be nodes or indexes in an array.

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
    -> Know floyd warshall, bellman ford, djikstra wellll


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

6) If you need to do range searches, you need a range tree. if you dont have time to get a range tree, 
    use binary searching as the substitute!

7) if the problem is unsorted, try sorting and if you need to keeping track of indexes (reverse index map) to do computations. 

7.5) If the problem is already sorted, try binary search. 

8) Do preprocessing work before you start solving problem to improve efficiency. This follows the abuse 
   dictionaries guide.

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

    -> LOOK BELOW FOR DYNAMIC PROGRAMMING RECURRENCE EXAMPLES!!!

11) Know how to write BFS with a deque, and DFS explicitely with a list. Keep tracking of function arguments in tuple for list. 

12) If you need a priority queue, use heapq. Need it for djikistras. Djikstras is general BFS for graphs with different sized edges. 


13) Skip lists are nice

14) Use stacks/queues to take advantage of push/pop structure in problems such as parentheses problems. Or valid expression problems.

15) When you have a problem and it gives you a Binary search tree, make sure to exploit that structure and not treat it as a normal
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


22) Know in-place reverse linked list
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



#####################################################################################################################

COOL NOTES PART 1: DYNAMIC PROGRAMMING RECURRENCES EXAMPLES:











#################################################################################################################

COOL Notes PART 2: Bit Magic

1) The left shift and right shift operators should not be used for negative numbers 
If any of the operands is a negative number, it results in undefined behaviour. 
For example results of both -1 << 1 and 1 << -1 is undefined. Also, if the number 
is shifted more than the size of integer, the behaviour is undefined. For example, 
1 << 33 is undefined if integers are stored using 32 bits. See this for more details.

2) The bitwise XOR operator is the most useful operator from technical interview perspective. 
It is used in many problems. A simple example could be “Given a set of numbers where all 
elements occur even number of times except one number, find the odd occurring number” 
This problem can be efficiently solved by just doing XOR of all numbers.

3) The bitwise operators should not be used in place of logical operators.

4) The left-shift and right-shift operators are equivalent to multiplication and division by 2 respectively.
As mentioned in point 1, it works only if numbers are positive.

5) The & operator can be used to quickly check if a number is odd or even
The value of expression (x & 1) would be non-zero only if x is odd, otherwise the value would be zero.

6) The ~ operator should be used carefully
The result of ~ operator on a small number can be a big number if the result is stored in an
unsigned variable. And result may be negative number if result is stored in signed 
variable (assuming that the negative numbers are stored in 2’s complement 
form where leftmost bit is the sign bit)


-> X ^ 0s = x
-> X ^ 1s = ~x
x ^ x = 0

7) Compute XOR from 1 to n (direct method) :

        // Direct XOR of all numbers from 1 to n 
        int computeXOR(int n) 
        { 
        	if (n % 4 == 0) 
        		return n; 
        	if (n % 4 == 1) 
        		return 1; 
        	if (n % 4 == 2) 
        		return n + 1; 
        	else
        		return 0; 
        } 

8) We can quickly calculate the total number of combinations with numbers smaller than or
   equal to with a number whose sum and XOR are equal. Instead of using looping 
   (Brute force method), we can directly find it by a mathematical trick i.e.

    // Refer Equal Sum and XOR for details.
    Answer = pow(2, count of zero bits)

9) How to know if a number is a power of 2?
        // Function to check if x is power of 2 
        bool isPowerOfTwo(int x) 
        { 
        	// First x in the below expression is 
        	// for the case when x is 0 
        	return x && (!(x & (x - 1))); 
        } 

10) Find XOR of all subsets of a set. We can do it in O(1) time. 
    The answer is always 0 if given set has more than one elements. 
    For set with single element, the answer is value of single element. 

11) We can quickly find number of leading, trailing zeroes and number of 1’s 
    in a binary code of an integer in C++ using GCC. 
    It can be done by using inbuilt function i.e.
  
    Number of leading zeroes: builtin_clz(x)
    Number of trailing zeroes : builtin_ctz(x)
    Number of 1-bits: __builtin_popcount(x) 
    Refer GCC inbuilt functions for details.

12) Convert binary code directly into an integer in C++.
        // Conversion into Binary code// 
        #include <iostream> 
        using namespace std; 
        
        int main() 
        { 
            auto number = 0b011; 
            cout << number; 
            return 0; 
        } 
        Output: 3




COOL NOTES PART 3: USING XOR TO SOLVE PROBLEMS EXAMPLES: ##################################################################################3

1) You are given a list of n-1 integers and these integers are in the range of 1 to n. 
There are no duplicates in list. One of 
the integers is missing in the list. Write an efficient code to find the missing integer.

METHOD 2(Use XOR)

  1) XOR all the array elements, let the result of XOR be X1.
  2) XOR all numbers from 1 to n, let XOR be X2.
  3) XOR of X1 and X2 gives the missing number.


2) How to swap two numbers without using a temporary variable?
   Given two variables, x and y, swap two variables without using a third variable.

   int main() 
{ 
    int x = 10, y = 5; 
  
    // Code to swap 'x' and 'y' 
    x = x + y; // x now becomes 15 
    y = x - y; // y becomes 10 
    x = x - y; // x becomes 5 
    cout << "After Swapping: x =" << x << ", y=" << y; 
} 

OR:  (WITH XOR)

int main() 
{ 
    int x = 10, y = 5; 
    // Code to swap 'x' (1010) and 'y' (0101) 
    x = x ^ y; // x now becomes 15 (1111) 
    y = x ^ y; // y becomes 10 (1010) 
    x = x ^ y; // x becomes 5 (0101) 
    cout << "After Swapping: x =" << x << ", y=" << y; 
    return 0; 
} 

3) XOR Linked List – A Memory Efficient Doubly Linked List | Set 1

An ordinary Doubly Linked List requires space for two address 
fields to store the addresses of previous and next nodes. 
A memory efficient version of Doubly Linked List can be 
created using only one space for address field with every node. 
This memory efficient Doubly Linked List is called XOR Linked List 
or Memory Efficient as the list uses bitwise XOR operation to save space 
for one address. In the XOR linked list, instead of storing actual memory 
addresses, every node stores the XOR of addresses of previous and next nodes.

Traversal of XOR Linked List:
We can traverse the XOR list in both forward 
and reverse direction. While traversing the 
list we need to remember the address of 
the previously accessed node in order to 
calculate the next node’s address. 
For example when we are at node C, we must have address of 
B. XOR of add(B) and npx of C gives us the add(D). The reason 
is simple: npx(C) is “add(B) XOR add(D)”. If we do xor of npx(C) 
with add(B), we get the result as “add(B) XOR add(D) XOR add(B)” 
which is “add(D) XOR 0” which is “add(D)”. So we have the address of next node. 
Similarly we can traverse the list in backward direction.

4) Find the two non-repeating elements in an array of repeating elements

    Let x and y be the non-repeating elements we are looking for and arr[] be the input array.
    First, calculate the XOR of all the array elements.

        xor = arr[0]^arr[1]^arr[2].....arr[n-1]
    All the bits that are set in xor will be set in one non-repeating 
    element (x or y) and not in others. So if we take any set bit of xor 
    and divide the elements of the array in two sets – one set of elements
    with same bit set and another set with same bit not set. By doing so, 
    we will get x in one set and y in another set. Now if we do XOR of all 
    the elements in the first set, we will get the first non-repeating element,
    and by doing same in other sets we will get the second non-repeating element.

    Let us see an example.
    arr[] = {2, 4, 7, 9, 2, 4}
    1) Get the XOR of all the elements.
        xor = 2^4^7^9^2^4 = 14 (1110)
    2) Get a number which has only one set bit of the xor.   
    Since we can easily get the rightmost set bit, let us use it.
        set_bit_no = xor & ~(xor-1) = (1110) & ~(1101) = 0010
    Now set_bit_no will have only set as rightmost set bit of xor.
    3) Now divide the elements in two sets and do xor of         
    elements in each set and we get the non-repeating 
    elements 7 and 9. Please see the implementation for this step.
    /* Now divide elements in two sets by comparing rightmost set 
   bit of xor with bit at same position in each element. */
    for(i = 0; i < n; i++) 
    { 
        if(arr[i] & set_bit_no) 
        *x = *x ^ arr[i]; /*XOR of first set */
        else
        *y = *y ^ arr[i]; /*XOR of second set*/
    } 

5)   Find the two numbers with odd occurrences in an unsorted array
        # Python3 program to find the 
        # two odd occurring elements 
        
        # Prints two numbers that occur odd 
        # number of times. The function assumes 
        # that the array size is at least 2 and 
        # there are exactly two numbers occurring 
        # odd number of times. 
        def printTwoOdd(arr, size): 
            
            # Will hold XOR of two odd occurring elements  
            xor2 = arr[0]  
            
            # Will have only single set bit of xor2 
            set_bit_no = 0  
            n = size - 2
            x, y = 0, 0
        
            # Get the xor of all elements in arr[].  
            # The xor will basically be xor of two 
            # odd occurring elements  
            for i in range(1, size): 
                xor2 = xor2 ^ arr[i] 
        
            # Get one set bit in the xor2. We get  
            # rightmost set bit in the following  
            # line as it is easy to get  
            set_bit_no = xor2 & ~(xor2 - 1) 
        
            # Now divide elements in two sets:  
            # 1) The elements having the corresponding bit as 1.  
            # 2) The elements having the corresponding bit as 0.  
            for i in range(size): 
            
                # XOR of first set is finally going to   
                # hold one odd  occurring number x  
                if(arr[i] & set_bit_no): 
                    x = x ^ arr[i] 
        
                # XOR of second set is finally going  
                # to hold the other odd occurring number y  
                else: 
                    y = y ^ arr[i]  
        
            print("The two ODD elements are", x, "&", y) 
        
        # Driver Code 
        arr = [4, 2, 4, 5, 2, 3, 3, 1] 
        arr_size = len(arr) 
        printTwoOdd(arr, arr_size) 
    

6) Add two numbers without using arithmetic operators

        Write a function Add() that returns sum of two integers. 
        The function should not use any of the arithmetic operators (+, ++, –, -, .. etc).
        Sum of two bits can be obtained by performing XOR (^) of the two bits. 
        Carry bit can be obtained by performing AND (&) of two bits.
        Above is simple Half Adder logic that can be used to add 2 single bits.
        We can extend this logic for integers. If x and y don’t have set bits at same position(s), 
        then bitwise XOR (^) of x and y gives the sum of x and y. To incorporate common set bits also, 
        bitwise AND (&) is used. Bitwise AND of x and y 
        gives all carry bits. We calculate (x & y) << 1 and 
        add it to x ^ y to get the required result.

        # Python3 Program to add two numbers 
        # without using arithmetic operator 
        def Add(x, y): 
        
            # Iterate till there is no carry  
            while (y != 0): 
            
                # carry now contains common 
                # set bits of x and y 
                carry = x & y 
        
                # Sum of bits of x and y where at 
                # least one of the bits is not set 
                x = x ^ y 
        
                # Carry is shifted by one so that    
                # adding it to x gives the required sum 
                y = carry << 1
            
            return x 
        
        print(Add(15, 32)) 
  
7. Count number of bits to be flipped to convert A to B

        Given two numbers ‘a’ and b’. Write a program to count number 
        of bits needed to be flipped to convert ‘a’ to ‘b’.

        1. Calculate XOR of A and B.      
                a_xor_b = A ^ B
        2. Count the set bits in the above 
            calculated XOR result.
                countSetBits(a_xor_b)

        # Function that count set bits 
        def countSetBits( n ): 
            count = 0
            while n: 
                count += n & 1
                n >>= 1
            return count 
            
        # Function that return count of 
        # flipped number 
        def FlippedCount(a , b): 
        
            # Return count of set bits in 
            # a XOR b 
            return countSetBits(a^b) 
        
        # Driver code 
        a = 10
        b = 20
        print(FlippedCount(a, b)) 

8.      Find the element that appears once. Given an array where every element occurs three times, 
        except one element which occurs only once. Find the element that occurs once.

        Run a loop for all elements in array. At the end of every iteration, maintain following two values.

        ones: The bits that have appeared 1st time or 4th time or 7th time .. etc.

        twos: The bits that have appeared 2nd time or 5th time or 8th time .. etc.

        Finally, we return the value of ‘ones’

        How to maintain the values of ‘ones’ and ‘twos’?
        ‘ones’ and ‘twos’ are initialized as 0. For every new element in array, 
        find out the common set bits in the new element and previous value of ‘ones’. 
        These common set bits are actually the bits that should be added to ‘twos’. 
        So do bitwise OR of the common set bits with ‘twos’. ‘twos’ also gets 
        some extra bits that appear third time. These extra bits are removed later.
        Update ‘ones’ by doing XOR of new element with previous value of ‘ones’.
        There may be some bits which appear 3rd time. These extra bits are also removed later.

        Both ‘ones’ and ‘twos’ contain those extra bits which appear 3rd time. 
        Remove these extra bits by finding out common set bits in ‘ones’ and ‘twos’.



        Below is the implementation of above approach:

        # Python3 code to find the element that  
        # appears once 

        def getSingle(arr, n): 
            ones = 0
            twos = 0

            for i in range(n): 
                # one & arr[i]" gives the bits that 
                # are there in both 'ones' and new 
                # element from arr[]. We add these 
                # bits to 'twos' using bitwise OR 
                twos = twos | (ones & arr[i]) 

                # one & arr[i]" gives the bits that 
                # are there in both 'ones' and new 
                # element from arr[]. We add these 
                # bits to 'twos' using bitwise OR 
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

        # driver code 
        arr = [3, 3, 2, 3] 
        n = len(arr) 
        print("The element with single occurrence is ", 
                getSingle(arr, n)) 

9. Detect if two integers have opposite signs

        Given two signed integers, write a function that returns true if the signs of 
        given integers are different, otherwise false. For example, the function 
        should return true -1 and +100, and should return false for -100 and -200. 
        The function should not use any of the arithmetic operators.

        Let the given integers be x and y. The sign bit is 1 in negative numbers, 
        and 0 in positive numbers. The XOR of x and y will have the sign bit as 1 
        iff they have opposite sign. In other words, XOR of x and y will be negative 
        number number iff x and y have opposite signs. The following code use this logic.

        # Python3 Program to Detect  
        # if two integers have  
        # opposite signs. 
        def oppositeSigns(x, y): 
            return ((x ^ y) < 0); 

        x = 100
        y = 1

        if (oppositeSigns(x, y) == True): 
            print "Signs are opposite"
        else: 
            print "Signs are not opposite"

10. Return the rightmost 1 in the binary representation of a number.
        Example: For 1010, you should perform some operations to give 0010 as the output. 
        For 1100, you should give 0100. Similarly for 0001, you should return 0001.
    
        For this problem, you need to know a property of binary subtraction. 
        Check if you can find out the property in the examples below,

        1000 – 0001 = 0111
        0100 – 0001 = 0011
        1100 – 0001 = 1011

        The property is, the difference between a binary number n and n-1 
        is all the bits on the right of the rightmost 1 are flipped 
        including the rightmost 1.  
        Using this amazing property, we can get our solution as
        x ^ (x & (x - 1))
        
        You now already know 80% about a powerful data structure called 
        Fenwick Tree or Binary Indexed Tree. You can look up on it to 
        learn the 20% or let me know if you want my next article to be about it. )
