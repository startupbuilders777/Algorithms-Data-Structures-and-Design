######################################################################
MAX FLOW MIN CUT:

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

############################################################################################
########################################################################################


49.5) RANGE MINIMUM QUERY:

    You are given an array A[1..N]. You have to answer incoming queries of the form (L,R), 
    which ask to find the minimum element in array A between positions L and R inclusive.

    RMQ can appear in problems directly or can be applied in some other tasks, 
    e.g. the Lowest Common Ancestor problem.

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

############################################################################
############################################################################
MEDIAN OF MEDIANS:


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


    MEDIAN OF MEDIANS(QUICKSELECT) with constant space PARTITION FUNCTION.:

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
