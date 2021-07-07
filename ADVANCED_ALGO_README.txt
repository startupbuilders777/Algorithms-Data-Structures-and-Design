1) Binary Trees And when to process null nodes: 
    AKA VERY WIERD RECURSION TREE

   Pseudo-Palindromic Paths in a Binary Tree

   Given a binary tree where node values are digits from 1 to 9. 
   A path in the binary tree is said to be pseudo-palindromic if 
   at least one permutation of the node values in the path is a palindrome.

   Return the number of pseudo-palindromic paths going from the root node to leaf nodes.

   Example 1:

   Input: root = [2,3,1,3,1,null,1]
   Output: 2 
   Explanation: The figure above represents the given binary tree. 
   There are three paths going from the root node to leaf nodes: the 
   red path [2,3,3], the green path [2,1,1], and the path [2,3,1]. 
   Among these paths only red path and green path are pseudo-palindromic 
   paths since the red path [2,3,3] can be rearranged in [3,2,3] (palindrome) 
   and the green path [2,1,1] can be rearranged in [1,2,1] (palindrome).
   Example 2:

   Input: root = [2,1,1,1,3,null,null,null,null,null,1]
   Output: 1 
   Explanation: The figure above represents the given binary tree. 
   There are three paths going from the root node to leaf nodes: 
   the green path [2,1,1], the path [2,1,3,1], and the path [2,1]. 
   Among these paths only the green path is pseudo-palindromic 
   since [2,1,1] can be rearranged in [1,2,1] (palindrome).
   Example 3:

   Input: root = [9]
   Output: 1

    SOLUTION:
    
    
    In the following problem, we have to process entire paths of binary tree,
    so we can't recurse on every child, except the child that exists:
    aka the normal check only for nullptr at top cant be used.
    you have to look 1 level ahead and check for children in the base case
    and only recurse on the side with children: 


    class Solution {
    public:
        int pseudoPalindromicPaths (TreeNode* root) {
            // count it up!
            unordered_map<int, int> m;
            int count = helper(root, m);
            return count; 
        }
        
        int palinCheck(const unordered_map<int, int>& m) {
            bool odd = false; // only  
            for(const auto & [key, value] : m) {
                if(value % 2 != 0 && odd) {
                    return false;
                }  else if(value % 2 != 0 ) {
                    odd = true;
                }
            }
            return true;    
        }
        
        int helper(TreeNode* node, unordered_map<int, int>& m) {
            m[node->val] = m.find(node->val) != m.end() ? m[node->val] + 1 : 1;
            int res = 0;
            if(node->left == nullptr && node->right == nullptr) {            
            res = palinCheck(m); 
            }
            
            if(node->left) {
                res += helper(node->left, m);
            }
            
            if(node->right) {
                res += helper(node->right, m);
            }
            
            m[node->val] -= 1; 
            return res; 
        }
        
    };

    BETTER SOLUTION: USE XOR AND CONSTANT SPACE TO KEEP TRACK OF VALUES WE VISITED: 
    // THEN CHECK IF ONE BIT IS ACTIVE BUT CHECKING IF ITS A POWER OF 2!!


    class Solution:
        def pseudoPalindromicPaths (self, root: TreeNode) -> int:
            def preorder(node, path):
                nonlocal count
                if node:
                    # compute occurences of each digit 
                    # in the corresponding register
                    path = path ^ (1 << node.val)
                    # if it's a leaf, check if the path is pseudo-palindromic
                    if node.left is None and node.right is None:
                        # check if at most one digit has an odd frequency
                        if path & (path - 1) == 0:
                            count += 1
                    else:                    
                        preorder(node.left, path)
                        preorder(node.right, path) 
            
            count = 0
            preorder(root, 0)
            return count

1.2) Morris Traversal Review:

    To do post order traversal or inorder traversal 
    on a binary tree iteratively (or doing any dfs, 
    where you want to vist root node last). 
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


1.6) MORRIS INORDER TRAVERSAL:
    (https://www.youtube.com/watch?v=wGXB9OWhPTg) 2 times speed 

    DONT USE A STACK OR ANYTHING ELSE. IT RUINS THE O(1) CONSTANT SPACE 
    WE ARE TRYING TO ACHIEVE WITH MORRIS TRAVERSAL

    basically we keep going to the left node, until we reach the leftmost node.
    while we do this, we set the backlinks for the node that ends our 
    inorder  traversal at the current level we are processing
    and we go deeper in the levels and continue to process left and save backlinks 
    we create the backlinks for the rightmost node on the left side because we 
    want a way to go back to the root node once we are on the very left side. 
    on the very left side we have processed left, then we process current node, 
    then when we process the right node the magic happens, the right node either 
    causes us to process it, then we go right, and that will take us to the root 
    which is a level above where we need to be to continue to do the inorder traversals. 

    we also have to do cleanup after we have jumped up the level, so we check its left, its 
    rightmost points to us because we were set as a backlink, delete the backlink

    class Solution {
    public:
        vector<int> inorderTraversal(TreeNode* root) {
            vector<int> nodes;
            while (root) {
                if (root -> left) {
                    TreeNode* pre = root -> left;
                    while (pre -> right && pre -> right != root) {
                        // go to rightmost node on left side
                        // sometimes the rightmost node leads back to root!
                        pre = pre -> right;
                    }
                    if (!pre -> right) {
                        // rightmost node doesnt have back connection to root of left, set it. 
                        pre -> right = root;
                        // go left and continue to set the backlinks, until you cant go left no more!
                        root = root -> left;
                    } else {
                        // revert back the changes of the back ptr. 
                        pre -> right = NULL;
                        nodes.push_back(root -> val); // insert rightmost node on left side. 
                        root = root -> right;
                    }
                } else {
                    // okay insert current node then go to right side to continue inorder
                    nodes.push_back(root -> val);
                    // below statement either goes right of root, or traverses a backlink back to root!
                    root = root -> right;
                }
            }
            return nodes;
        }
    };



1.7) Morris Traversal INORDER -> Review Continued:

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


1.8 Morris Traversal Preorder:

    public class Solution {
        public List<Integer> preorderTraversal(TreeNode root) {
            List<Integer> res = new ArrayList<>();
            if (root == null)
                return res;
            TreeNode cur = root;
            while (cur != null) {
                if (cur.left == null) {
                    res.add(cur.val);
                    cur = cur.right;
                } else {
                    TreeNode prev = cur.left;
                    while (prev.right != null && prev.right != cur)
                        prev = prev.right;
                    if (prev.right == null) {
                        res.add(cur.val);
                        prev.right = cur;
                        cur = cur.left;
                    } else {
                        prev.right = null;
                        cur = cur.right;
                    }
                }
            }
            return res;
        }
    }
    A good explanation here gives implementation of Morris Traversal InOrder algorithm. 
    The PreOrder algorithm can be achieved by tweaking the code very slightly: rather than 
    visit cur when we restore the tree(which is generally the second time we are at node cur), 
    we visit it the first time we encounter it, which is when we have to modify.

    The space complexity for Morris Traversal is O(1) obviously. The time complexity is actually 
    O(N) which is a little more subtle. The outer while loop is executed O(N) iterations obviously, 
    depending on the position of root in the InOrder serialization of the tree. 
    In each such iteration, we have to find the left predecessor for cur, which 
    is the costly part. Trivially you would think that gives us O(NlgN) per height 
    of the tree. But if you think aggregately, you will see these:

    For each node, we do find left predecessor on it only twice.
    Throughout these two find left predecessor inner while loops, each edge of 
    the tree get traversed at most twice.
    You might have to draw a tree and doodle some trace to convince yourself this. 
    Since a tree has N-1 edges, with N as the number of nodes, we 
    know that the time complexity is O(N).

1.9 Morris Preorder Traversal:

    class Solution {
    public:
        vector<int> preorderTraversal(TreeNode* root) {
            
            vector<int>v;
            if (!root) return v;
            
            TreeNode* curr = root;
            
            while (curr) {
                if (!curr->left) {
                    v.push_back(curr->val);
                    curr = curr->right;
                } else {
                    TreeNode* pre = curr->left;

                    while (pre->right && pre->right != curr)
                        pre = pre->right;

                    if (pre->right == curr) {
                        pre->right = NULL;
                        curr = curr->right;
                    } else {
                        v.push_back(curr->val);
                        pre->right = curr;
                        curr = curr->left;
                    }
                }
            }
            
            return v;
        }
    };




2) Morris Preorder Traversal (CONSTANT SPACE TREE TRAVERSAL!!): 

   Sum Root to Leaf Numbers
   
   Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

   An example is the root-to-leaf path 1->2->3 which represents the number 123.

   Find the total sum of all root-to-leaf numbers.

   Note: A leaf is a node with no children.

   Example:

    3 Solutions: 
    Iterative Preorder Traversal -> With stack O(N) time, O(H) space complexity 
    Recursive Preorder Traversal -> O(N) and O(H)
    
    Best approach: Morris Preorder Traversal (Time O(N) space O(1)): 
    Trade in performance to save space. 

    The idea of Morris algorithm is to set the temporary link between the node and its 
    predecessor: predecessor.right = root. So one starts from the node, computes 
    its predecessor and verifies if the link is present.

    There is no link? Set it and go to the left subtree.
    There is a link? Break it and go to the right subtree.

    There is one small issue to deal with : what if there is no left child, i.e. 
    there is no left subtree? Then go straightforward to the right subtree.

    class Solution:
        def sumNumbers(self, root: TreeNode):
            root_to_leaf = curr_number = 0
            
            while root:  
                # If there is a left child,
                # then compute the predecessor.
                # If there is no link predecessor.right = root --> set it.
                # If there is a link predecessor.right = root --> break it.
                if root.left: 
                    # Predecessor node is one step to the left 
                    # and then to the right till you can.
                    predecessor = root.left 
                    steps = 1
                    while predecessor.right and predecessor.right is not root: 
                        predecessor = predecessor.right 
                        steps += 1

                    # Set link predecessor.right = root
                    # and go to explore the left subtree
                    if predecessor.right is None:
                        curr_number = curr_number * 10 + root.val                    
                        predecessor.right = root  
                        root = root.left  
                    # Break the link predecessor.right = root
                    # Once the link is broken, 
                    # it's time to change subtree and go to the right
                    else:
                        # If you're on the leaf, update the sum
                        if predecessor.left is None:
                            root_to_leaf += curr_number
                        # This part of tree is explored, backtrack
                        for _ in range(steps):
                            curr_number //= 10
                        predecessor.right = None
                        root = root.right 
                        
                # If there is no left child
                # then just go right.        
                else: 
                    curr_number = curr_number * 10 + root.val
                    # if you're on the leaf, update the sum
                    if root.right is None:
                        root_to_leaf += curr_number
                    root = root.right
                            
            return root_to_leaf


2.5) Morris Post Order Traversal (not really post order, just reversed preorder, garbage soln): 

    In pre-order Morris Traverse, we always check whether the current 
    pointer can go left, finding out the predecessor, 
    linking its right pointer to current pointer.

    In post-order Morris Traverse, we just flip the logic (do not need 
    to a bunch of reverse). Instead of trying go left, we tries to go right.

    Moreover, we use a deque to collect the answer.

    class Solution:
        def go_right(self, curr, ans):
            if not curr.right:
                ans.appendleft(curr.val)
                return False
            
            after = curr.right
            
            while after.left and after.left != curr:
                after = after.left
                
            if after.left == curr:
                after.left = None
                return False
            else:
                ans.appendleft(curr.val)
                after.left = curr
                return True
            
            
        def postorderTraversal(self, root: TreeNode) -> List[int]:
            if not root:
                return []
            
            from collections import deque
            ans = deque()
            curr = root
            
            while curr:
                if self.go_right(curr, ans):
                    curr = curr.right
                else:
                    curr = curr.left
                    
            return list(ans)

2.6) Morris True Post Order:
    class Solution {
    public:
        vector<int> postorderTraversal(TreeNode* root) {
            vector<int> nodes;
            TreeNode* dummy = new TreeNode(0);
            dummy -> left = root;
            TreeNode* cur = dummy;
            while (cur) {
                if (cur -> left) {
                    TreeNode* pre = cur -> left;
                    while (pre -> right && (pre -> right != cur)) {
                        pre = pre -> right;
                    }
                    if (!(pre -> right)) {
                        pre -> right = cur;
                        cur = cur -> left;
                    } else {
                        reverseAddNodes(cur -> left, pre, nodes);
                        pre -> right = NULL;
                        cur = cur -> right;
                    }
                } else {
                    cur = cur -> right;
                }
            }
            return nodes;
        }
    private:
        void reverseNodes(TreeNode* start, TreeNode* end) {
            if (start == end) {
                return;
            }
            TreeNode* x = start;
            TreeNode* y = start -> right;
            TreeNode* z;
            while (x != end) {
                z = y -> right;
                y -> right = x;
                x = y;
                y = z;
            }
        }
        void reverseAddNodes(TreeNode* start, TreeNode* end, vector<int>& nodes) {
            reverseNodes(start, end);
            TreeNode* node = end;
            while (true) {
                nodes.push_back(node -> val);
                if (node == start) {
                    break;
                }
                node = node -> right;
            }
            reverseNodes(end, start);
        }
    };

2.7) Morris Post Order:

    public class Solution {
        // Important, when you pop a node, ensure its children are traversed.
        public List<Integer> postorderTraversal(TreeNode root) {
            Stack<TreeNode> s = new Stack();
            List<Integer> ans = new ArrayList<Integer>();
            TreeNode cur = root;
            
            while (cur != null || !s.empty()) {
                while (!isLeaf(cur)) {
                    s.push(cur);
                    cur = cur.left;
                }
                
                if (cur != null) ans.add(cur.val);
                
                while (!s.empty() && cur == s.peek().right) {
                    cur = s.pop();
                    ans.add(cur.val);
                }
                
                if (s.empty()) cur = null; else cur = s.peek().right;
            }
            
            return ans;
        }
        
        private boolean isLeaf(TreeNode r) {
            if (r == null) return true;
            return r.left == null && r.right == null;
        }
    }



3) Morris Post Order Traversal:

    GET ALL THEE CORRECT POST ORDERS, AND WRITE MORE NOTES FROM HERE:
    https://leetcode.com/problems/binary-tree-postorder-traversal/discuss


    How do we perform Postorder traversal. We'll use above concept with minor changes to 
    achieve postorder traversal. First lets have a dummy node and make whole tree as left 
    child of dummy node and make right child empty. [ why? Bec if we assume there is no right 
    child of root then prinitng left child and then root become postorder traversal, Right ;) ] 
    Now what next? Are we finished, No... only performing inorder on new tree does not make any 
    sense, it still printing inorder traversal of original tree followed by dummy node.


    //This is Post Order :children before node( L ,R , N)
    void morrisPostorderTraversal(Node *root){

        // Making our tree left subtree of a dummy Node
        Node *dummyRoot = new Node(0);
        dummyRoot->left = root;

        //Think of P as the current node 
        Node *p = dummyRoot, *pred, *first, *middle, *last;
        while(p!=NULL){        

            if(p->left == NULL){
                p = p->right;
            } else{
                /* p has a left child => it also has a predeccessor
                make p as right child predeccessor of p    
                */
                pred = p->left;
                while(pred->right!=NULL && pred->right != p){
                    pred = pred->right;
                }

                if(pred->right == NULL){ 

                    // predeccessor found for first time
                    // modify the tree

                    pred->right = p;    
                    p = p->left;

                }else {                          

                // predeccessor found second time
                // reverse the right references in chain from pred to p
                    first = p;
                    middle = p->left;              
                    while(middle!=p){            
                        last = middle->right;
                        middle->right = first;
                        first = middle;
                        middle = last;
                    }

                    // visit the nodes from pred to p
                    // again reverse the right references from pred to p    
                    first = p;
                    middle = pred;
                    while(middle!=p){

                        cout<<" "<<middle->data;  
                        last = middle->right;          
                        middle->right = first;
                        first = middle;

                        middle = last;
                    }

                    // remove the pred to node reference to restore the tree structure
                    pred->right = NULL;    
                    p = p-> right;
                }
            }
        }    
    }   




##############################################################################
RESERVOIR SAMPLING:

    Reservoir sampling is a family of randomized algorithms for randomly choosing k 
    samples from a list of n items, where n is either a very large or unknown number. 
    Typically n is large enough that the list doesn’t fit into main memory. 
    For example, a list of search queries in Google and Facebook.

    So we are given a big array (or stream) of numbers (to simplify), and we need 
    to write an efficient function to randomly select k numbers where 1 <= k <= n.
    Let the input array be stream[].

    A simple solution is to create an array reservoir[] of maximum 
    size k. One by one randomly select an item from stream[0..n-1]. 
    If the selected item is not previously selected, then put it in 
    reservoir[]. To check if an item is previously selected or not, we need 
    to search the item in reservoir[]. The time complexity of this algorithm 
    will be O(k^2). This can be costly if k is big. Also, this is not efficient 
    if the input is in the form of a stream.

    It can be solved in O(n) time. The solution also suits well for input 
    in the form of stream. The idea is similar to this post. 
    Following are the steps.

    1) Create an array reservoir[0..k-1] and copy first k items of stream[] to it.
    2) Now one by one consider all items from (k+1)th item to nth item.
    …a) Generate a random number from 0 to i where i is index of current item 
        in stream[]. Let the generated random number is j.
    …b) If j is in range 0 to k-1, replace reservoir[j] with arr[i]


    # An efficient Python3 program  
    # to randomly select k items 
    # from a stream of items 
    import random 
    # A utility function  
    # to print an array 
    def printArray(stream,n): 
        for i in range(n): 
            print(stream[i],end=" "); 
        print(); 
    
    # A function to randomly select 
    # k items from stream[0..n-1]. 
    def selectKItems(stream, n, k): 
            i=0;  
            # index for elements 
            # in stream[] 
            
            # reservoir[] is the output  
            # array. Initialize it with 
            # first k elements from stream[] 
            reservoir = [0]*k; 
            for i in range(k): 
                reservoir[i] = stream[i]; 
            
            # Iterate from the (k+1)th 
            # element to nth element 
            while(i < n): 
                # Pick a random index 
                # from 0 to i. 
                j = random.randrange(i+1); 
                
                # If the randomly picked 
                # index is smaller than k, 
                # then replace the element 
                # present at the index 
                # with new element from stream 
                if(j < k): 
                    reservoir[j] = stream[i]; 
                i+=1; 
            
            print("Following are k randomly selected items"); 
            printArray(reservoir, k); 
        
    # Driver Code 
    
    if __name__ == "__main__": 
        stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; 
        n = len(stream); 
        k = 5; 
        selectKItems(stream, n, k); 
    
    # This code is contributed by mits 




    How does this work?
    Imagine the following "dating" game show. The contestant, a bachelorette, is seated 
    at a table with an empty chair. The host introduces the first suitor; the bachelorette has 
    to invite him to sit with her and be her current "date". Next, the host introduces the second 
    suitor. Now the girl gets the choice of whether she will keep her current "date" or replace him 
    with the new suitor. She can use a variety of means to make her decision, such as asking 
    questions, or making the two suitors compete in some way. After that, the host introduces 
    the third suitor, and again the girl can choose to either keep or replace her current "date." 
    After showing n suitors this way, the game show ends, and the girl goes on a real date 
    with the suitor she kept at the end, the "winner" of the show.

    Imagine one contestant who simply flips a coin to decide whether or not to swap her current 
    "date". Is this "fair" to the suitors, i.e., is the probability distribution of the winner 
    uniform over all the suitors? The answer is no, because it is much more likely for the last 
    few suitors to win than the first few suitors. The very first suitor is most unfortunate of all, 
    since if he wants to go on a date with the girl, he has to survive n-1 coin flips. The last 
    suitor has the best chances--he only needs to win a single coin flip.

    So, what kind of procedure should the girl use if she wants to give all the suitors an equal
    chance, 1/n? First of all, she has to swap to the last suitor with probability 1/n, 
    since the last suitor wins if and only if she decides to swap on the last step. Now 
    what about the second-to-last suitor? In order for him to win, she has to decide to 
    swap on the (n-1)th step, and then also decide to not swap on the nth step. This happens 
    with probability  (pn−1) * (n-1)/n, where  pn−1  is the probability that she swaps on the (n-1)th 
    step and (n-1)/n is the probability she does not swap on the nth step. Solving for

    pn−1 (n-1)/n = 1/n

    we get  pn−1  = 1/(n-1). Continuing the pattern, we see that her swap 
    probability on the kth step is 1/k.

    That is the algorithm for reservoir sampling.




    To prove that this solution works perfectly, we must prove that 
    the probability that any item stream[i] where 0 <= i < n will be in 
    final reservoir[] is k/n. Let us divide the proof in two cases 
    as first k items are treated differently.


    Case 1: For last n-k stream items, i.e., for stream[i] where k <= i < n
    For every such stream item stream[i], we pick a random index from 0 to i and 
    if the picked index is one of the first k indexes, we replace 
    the element at picked index with stream[i]

    To simplify the proof, let us first consider the last item. The probability that the 
    last item is in final reservoir = The probability that one of the first k indexes is 
    picked for last item = k/n (the probability of picking one of the k items from a list of size n)

    Let us now consider the second last item. The probability that the second last 
    item is in final reservoir[] = 
    [Probability that one of the first k indexes is picked in iteration for stream[n-2]] X 
    [Probability that the index picked in iteration for stream[n-1] is not same as 
    index picked for stream[n-2] ] = [k/(n-1)]*[(n-1)/n] = k/n.

    Similarly, we can consider other items for all stream items from 
    stream[n-1] to stream[k] and generalize the proof.

    Case 2: For first k stream items, i.e., for stream[i] where 0 <= i < k
    The first k items are initially copied to reservoir[] and may be removed 
    later in iterations for stream[k] to stream[n].
    The probability that an item from stream[0..k-1] is in final array = Probability that 
    the item is not picked when items stream[k], stream[k+1], …. stream[n-1] are 
    considered = [k/(k+1)] x [(k+1)/(k+2)] x [(k+2)/(k+3)] x … x [(n-1)/n] = k/n














##########################################################
LCA - Intro (https://cp-algorithms.com/graph/lca.html)
    (USE THE IMAGE IN THE TUTORIAL TO HELP SEE EXAMPLE)


    Given a tree G. Given queries of the form (v1,v2), for 
    each query you need to find the lowest common ancestor (or least common ancestor), 
    i.e. a vertex v that lies on the path from the root to v1 
    and the path from the root to v2, and the vertex should be 
    the lowest. In other words, the desired vertex v is the most 
    bottom ancestor of v1 and v2. It is obvious that their 
    lowest common ancestor lies on a shortest path from v1 and v2. 
    Also, if v1 is the ancestor of v2, v1 is their lowest common ancestor.


    The Idea of the Algorithm
    Before answering the queries, we need to preprocess the tree.
    We make a DFS traversal starting at the root and we build a list 
    euler which stores the order of the vertices that we visit (a vertex 
    is added to the list when we first visit it, and after the return 
    of the DFS traversals to its children). This is also called an Euler tour 
    of the tree. It is clear that the size of this list will be O(N). 
    We also need to build an array first[0..N−1] which stores for each 
    vertex i its first occurrence in euler. That is, the first position 
    in euler such that euler[first[i]]=i. Also by using the DFS we 
    can find the height of each node (distance from root to it) 
    and store it in the array height[0..N−1].

    So how can we answer queries using the Euler tour and the additional 
    two arrays? Suppose the query is a pair of v1 and v2. Consider 
    the vertices that we visit in the Euler tour between the first 
    visit of v1 and the first visit of v2. It is easy to see, that the 
    LCA(v1,v2) is the vertex with the lowest height on this path. 
    We already noticed, that the LCA has to be part of the shortest 
    path between v1 and v2. Clearly it also has to be the vertex 
    with the smallest height. And in the Euler tour we essentially 
    use the shortest path, except that we additionally visit all 
    subtrees that we find on the path. But all vertices in these 
    subtrees are lower in the tree than the LCA and therefore have a
    larger height. So the LCA(v1,v2) can be uniquely determined 
    by finding the vertex with the smallest height in the Euler tour 
    between first(v1) and first(v2).

    To recap: to answer a query we just need to find the vertex 
    with smallest height in the array euler in the range from 
    first[v1] to first[v2]. Thus, the LCA problem is reduced to 
    the RMQ problem (finding the minimum in an range problem).
    
    Using Sqrt-Decomposition, it is possible to obtain
    a solution answering each query in O(√N) with preprocessing in O(N) time.

    Using a Segment Tree you can answer each query in O(logN) 
    with preprocessing in O(N) time.

    Since there will almost never be any update to the stored values, 
    a Sparse Table might be a better choice, allowing O(1) 
    query answering with O(NlogN) build time.

LCA - IMPLEMENTATION OF ABOVE IDEA: 

    struct LCA {
        vector<int> height, euler, first, segtree;
        vector<bool> visited;
        int n;

        LCA(vector<vector<int>> &adj, int root = 0) {
            n = adj.size();
            height.resize(n);
            first.resize(n);
            euler.reserve(n * 2);
            visited.assign(n, false);
            dfs(adj, root);
            int m = euler.size();
            segtree.resize(m * 4);
            build(1, 0, m - 1);
        }

        void dfs(vector<vector<int>> &adj, int node, int h = 0) {
            visited[node] = true;
            height[node] = h;
            first[node] = euler.size();
            euler.push_back(node);
            for (auto to : adj[node]) {
                if (!visited[to]) {
                    dfs(adj, to, h + 1);
                    euler.push_back(node);
                }
            }
        }

        void build(int node, int b, int e) {
            if (b == e) {
                segtree[node] = euler[b];
            } else {
                int mid = (b + e) / 2;
                build(node << 1, b, mid);
                build(node << 1 | 1, mid + 1, e);
                int l = segtree[node << 1], r = segtree[node << 1 | 1];
                segtree[node] = (height[l] < height[r]) ? l : r;
            }
        }

        int query(int node, int b, int e, int L, int R) {
            if (b > R || e < L)
                return -1;
            if (b >= L && e <= R)
                return segtree[node];
            int mid = (b + e) >> 1;

            int left = query(node << 1, b, mid, L, R);
            int right = query(node << 1 | 1, mid + 1, e, L, R);
            if (left == -1) return right;
            if (right == -1) return left;
            return height[left] < height[right] ? left : right;
        }

        int lca(int u, int v) {
            int left = first[u], right = first[v];
            if (left > right)
                swap(left, right);
            return query(1, 0, euler.size() - 1, left, right);
        }
    };



##########################################################
LCA - Lowest Common Ancestor - Binary Lifting 
    (https://cp-algorithms.com/graph/lca_binary_lifting.html)
    (USE THE IMAGE IN THE TUTORIAL TO HELP SEE EXAMPLE)


    Algorithm
    For each node we will precompute its ancestor above him, 
    its ancestor two nodes above, its ancestor four above, etc.
    Let's store them in the array up, i.e. up[i][j] is the 2^j-th ancestor 
    above the node i with i=1...N, j=0...ceil(log(N)). These information 
    allow us to jump from any node to any ancestor above
    it in O(logN) time. We can compute this array using
    a DFS traversal of the tree.

    For each node we will also remember the time of the first 
    visit of this node (i.e. the time when the DFS discovers the node), 
    and the time when we left it (i.e. after we visited all 
    children and exit the DFS function). We can use this information 
    to determine in constant time if a node is an ancestor of another node.

    Suppose now we received a query (u, v). We can immediately 
    check whether one node is the ancestor of the other. In this case 
    this node is already the LCA. If u is not the ancestor of v, 
    and v not the ancestor of u, we climb the ancestors of u 
    until we find the highest (i.e. closest to the root) node, 
    which is not an ancestor of v (i.e. a node x, such that x is not an ancestor of v, 
    but up[x][0] is). We can find this node x in O(logN) 
    time using the array up.

    We will describe this process in more detail. Let L = ceil(log(N)). 
    Suppose first that i = L. If up[u][i] is not an ancestor of v, 
    then we can assign u = up[u][i] and decrement i. If up[u][i] is an ancestor, 
    then we just decrement i. Clearly after doing this for all non-negative i 
    the node u will be the desired node - i.e. u is still not an 
    ancestor of v, but up[u][0] is.

    Now, obviously, the answer to LCA will be up[u][0] - i.e., 
    the smallest node among the ancestors of the node u, 
    which is also an ancestor of v.

    So answering a LCA query will iterate i from ceil(log(N)) to 0 a
    nd checks in each iteration if one node is the ancestor of the other. 
    Consequently each query can be answered in O(logN).

    Implementation
    int n, l;
    vector<vector<int>> adj;

    int timer;
    vector<int> tin, tout;
    vector<vector<int>> up;

    void dfs(int v, int p)
    {
        tin[v] = ++timer;
        up[v][0] = p;
        for (int i = 1; i <= l; ++i)
            up[v][i] = up[up[v][i-1]][i-1];

        for (int u : adj[v]) {
            if (u != p)
                dfs(u, v);
        }

        tout[v] = ++timer;
    }

    bool is_ancestor(int u, int v)
    {
        return tin[u] <= tin[v] && tout[u] >= tout[v];
    }

    int lca(int u, int v)
    {
        if (is_ancestor(u, v))
            return u;
        if (is_ancestor(v, u))
            return v;
        for (int i = l; i >= 0; --i) {
            if (!is_ancestor(up[u][i], v))
                u = up[u][i];
        }
        return up[u][0];
    }

    void preprocess(int root) {
        tin.resize(n);
        tout.resize(n);
        timer = 0;
        l = ceil(log2(n));
        up.assign(n, vector<int>(l + 1));
        dfs(root, root);
    }


##########################################################
RMQ PRACTICAL IMPLEMENTATIONS:


-44) SEGMENT TREES USE CASE IN RMQ: 

    Segment tree can be used to do preprocessing and query in moderate time. 
    With segment tree, preprocessing time is O(n) and time to for range 
    minimum query is O(Logn). The extra space required is O(n) to store the segment tree.

    Representation of Segment trees
    1. Leaf Nodes are the elements of the input array.
    2. Each internal node represents minimum of all leaves under it.

    An array representation of tree is used to represent Segment Trees. 
    For each node at index i, the left child is at index 2*i+1, 
    right child at 2*i+2 and the parent is at (i-1)/2


    Construction of Segment Tree from given array
    We start with a segment arr[0 . . . n-1]. and every time we divide 
    the current segment into two halves(if it has not yet become 
    a segment of length 1), and then call the same procedure on both halves, 
    and for each such segment, we store the minimum value in a segment tree node.

    All levels of the constructed segment tree will be completely 
    filled except the last level. Also, the tree will be a Full Binary Tree 
    because we always divide segments in two halves at every level. 
    Since the constructed tree is always full binary tree with n leaves, 
    there will be n-1 internal nodes. So total number of nodes will be 2^n – 1.

    Height of the segment tree will be logn. Since the tree is represented 
    using array and relation between parent and child indexes must 
    be maintained, size of memory allocated for segment tree will be 2*2^(ciel(logn)) - 1.

    Query for minimum value of given range
    Once the tree is constructed, how to do range minimum 
    query using the constructed segment tree. Following is algorithm to get the minimum.

    // qs --> query start index, qe --> query end index
    int RMQ(node, qs, qe) 
    {
    if range of node is within qs and qe
            return value in node
    else if range of node is completely outside qs and qe
            return INFINITE
    else
        return min( RMQ(node's left child, qs, qe), RMQ(node's right child, qs, qe) )
    }


-43.5) SEGMENT TREE IMPLEMENTATION FOR RMQ
    PYTHON SEGMENT TREE SOLUTION:

    # Python3 program for range minimum  
    # query using segment tree  
    import sys; 
    from math import ceil,log2; 
    
    INT_MAX = sys.maxsize; 
    
    # A utility function to get  
    # minimum of two numbers  
    def minVal(x, y) : 
        return x if (x < y) else y;  
    
    # A utility function to get the  
    # middle index from corner indexes.  
    def getMid(s, e) : 
        return s + (e - s) // 2;  
    
    """ A recursive function to get the  
    minimum value in a given range  
    of array indexes. The following  
    are parameters for this function.  
    
        st --> Pointer to segment tree  
        index --> Index of current node in the  
            segment tree. Initially 0 is  
            passed as root is always at index 0  
        ss & se --> Starting and ending indexes  
                    of the segment represented  
                    by current node, i.e., st[index]  
        qs & qe --> Starting and ending indexes of query range """
    
    def RMQUtil( st, ss, se, qs, qe, index) : 
    
        # If segment of this node is a part  
        # of given range, then return  
        # the min of the segment  
        if (qs <= ss and qe >= se) : 
            return st[index];  
    
        # If segment of this node  
        # is outside the given range  
        if (se < qs or ss > qe) : 
            return INT_MAX;  
    
        # If a part of this segment  
        # overlaps with the given range  
        mid = getMid(ss, se);  
        return minVal(RMQUtil(st, ss, mid, qs,  
                            qe, 2 * index + 1),  
                    RMQUtil(st, mid + 1, se, 
                            qs, qe, 2 * index + 2));  
    
    # Return minimum of elements in range  
    # from index qs (query start) to  
    # qe (query end). It mainly uses RMQUtil()  
    def RMQ( st, n, qs, qe) :  
    
        # Check for erroneous input values  
        if (qs < 0 or qe > n - 1 or qs > qe) : 
        
            print("Invalid Input");  
            return -1;  
        
        return RMQUtil(st, 0, n - 1, qs, qe, 0);  
    
    # A recursive function that constructs  
    # Segment Tree for array[ss..se].  
    # si is index of current node in segment tree st  
    def constructSTUtil(arr, ss, se, st, si) : 
    
        # If there is one element in array,  
        # store it in current node of  
        # segment tree and return  
        if (ss == se) : 
    
            st[si] = arr[ss];  
            return arr[ss];  
    
        # If there are more than one elements,  
        # then recur for left and right subtrees  
        # and store the minimum of two values in this node  
        mid = getMid(ss, se);  
        st[si] = minVal(constructSTUtil(arr, ss, mid, 
                                        st, si * 2 + 1), 
                        constructSTUtil(arr, mid + 1, se, 
                                        st, si * 2 + 2));  
        
        return st[si];  
    
    """Function to construct segment tree  
    from given array. This function allocates  
    memory for segment tree and calls constructSTUtil() 
    to fill the allocated memory """
    def constructST( arr, n) : 
    
        # Allocate memory for segment tree  
    
        # Height of segment tree  
        x = (int)(ceil(log2(n)));  
    
        # Maximum size of segment tree  
        max_size = 2 * (int)(2**x) - 1;  
    
        st = [0] * (max_size);  
    
        # Fill the allocated memory st  
        constructSTUtil(arr, 0, n - 1, st, 0);  
    
        # Return the constructed segment tree  
        return st;  
    
    # Driver Code 
    if __name__ == "__main__" :  
    
        arr = [1, 3, 2, 7, 9, 11];  
        n = len(arr);  
    
        # Build segment tree from given array  
        st = constructST(arr, n);  
    
        qs = 1; # Starting index of query range  
        qe = 5; # Ending index of query range  
    
        # Print minimum value in arr[qs..qe]  
        print("Minimum of values in range [", qs,  
            ",", qe, "]", "is =", RMQ(st, n, qs, qe));  
    
    # This code is contributed by AnkitRai01  



-43) STATIC RMQ Solutions:

    Input:  arr[]   = {7, 2, 3, 0, 5, 10, 3, 12, 18};
            query[] = [0, 4], [4, 7], [7, 8]

    Output: Minimum of [0, 4] is 0
            Minimum of [4, 7] is 3
            Minimum of [7, 8] is 12

    A simple solution is to run a loop from L to R and find minimum element in 
    given range. This solution takes O(n) time to query in worst case.

    Another approach is to use Segment tree. With segment tree, preprocessing 
    time is O(n) and time to for range minimum query is O(Logn). The extra space 
    required is O(n) to store the segment tree. Segment tree allows updates also in O(Log n) time.

    Can we do better if we know that array is static?
    How to optimize query time when there are no update operations 
    and there are many range minimum queries?

    Create 2D lookup table. Described in Algo point below -42)

    This approach supports query in O(1), but preprocessing takes O(n^2) time. 
    Also, this approach needs O(n^2) extra space which may become huge for large input arrays.

    Method 2 (Square Root Decomposition -> BTW YOU CAN MODIFY ARRAY BETWEEN QUERIES WITH THIS METHOD) 
    We can use Square Root Decompositions to reduce space required in above method.

    Preprocessing:
    1) Divide the range [0, n-1] into different blocks of √n each.
    2) Compute minimum of every block of size √n and store the results.

    Preprocessing takes O(√n * √n) = O(n) time and O(√n) space.

    Query:
    1) To query a range [L, R], we take minimum of all blocks that lie 
    2) in this range. For left and right corner blocks which may partially 
    3) overlap with given range, we linearly scan them to find minimum.

    Time complexity of query is O(√n). Note that we have minimum of middle block 
    directly accessible and there can be at most O(√n) middle blocks. There can be 
    at most two corner blocks that we may have to scan, so we may have to scan 2*O(√n) 
    elements of corner blocks. Therefore, overall time complexity is O(√n).


-42.9) STATIC RMQ Solutions WITH SPARSE TABLES :

    Method 3 (Sparse Table Algorithm)
    The above solution (sqrt decomp)) requires only O(√n) space, but takes O(√n) time to query. 
    Sparse table method supports query time O(1) with extra space O(n Log n).

    The idea is to pre-compute minimum of all subarrays of size 2^j where j varies 
    from 0 to Log n. Like method 1, we make a lookup table. 
    Here lookup[i][j] contains minimum of range starting from i and of size 2^j. 
    For example lookup[0][3] contains minimum of range [0, 7] (starting with 0 and of size 23)

    Preprocessing:
    How to fill this lookup table? The idea is simple, 
    fill in bottom up manner using previously computed values.

    For example, to find minimum of range [0, 7], we can use minimum of following two.
    a) Minimum of range [0, 3]
    b) Minimum of range [4, 7]

    Based on above example, below is formula,

    // If arr[lookup[0][2]] <=  arr[lookup[4][2]], 
    // then lookup[0][3] = lookup[0][2]
    If arr[lookup[i][j-1]] <= arr[lookup[i+2^(j-1)-1][j-1]]
        lookup[i][j] = lookup[i][j-1]

    // If arr[lookup[0][2]] >  arr[lookup[4][2]], 
    // then lookup[0][3] = lookup[4][2]
    Else 
        lookup[i][j] = lookup[i+2^(j-1)-1][j-1] 


    Query:
    For any arbitrary range [l, R], we need to use ranges which are in powers of 2. 
    The idea is to use closest power of 2. We always need to do at most one 
    comparison (compare minimum of two ranges which are powers of 2). One 
    range starts with L and and ends with “L + closest-power-of-2”. The other range 
    ends at R and starts with “R – same-closest-power-of-2 + 1”. For example, 
    if given range is (2, 10), we compare minimum of two ranges (2, 9) and (3, 10).

    Based on above example, below is formula,

    // For (2,10), j = floor(Log2(10-2+1)) = 3
    j = floor(Log(R-L+1))

    // If arr[lookup[0][3]] <=  arr[lookup[3][3]], 
    // then RMQ(2,10) = lookup[0][3]
    If arr[lookup[L][j]] <= arr[lookup[R-(int)pow(2,j)+1][j]]
        RMQ(L, R) = lookup[L][j]

    // If arr[lookup[0][3]] >  arr[lookup[3][3]], 
    // then RMQ(2,10) = lookup[3][3]
    Else 
        RMQ(L, R) = lookup[R-(int)pow(2,j)+1][j]

    Since we do only one comparison, time complexity of query is O(1).

    SPARSE TABLE SOLUTION:
    # Python3 program to do range minimum query  
    # in O(1) time with O(n Log n) extra space 
    # and O(n Log n) preprocessing time 
    from math import log2 
    
    MAX = 500
    
    # lookup[i][j] is going to store index of  
    # minimum value in arr[i..j].  
    # Ideally lookup table size should  
    # not be fixed and should be determined  
    # using n Log n. It is kept constant 
    # to keep code simple. 
    lookup = [[0 for i in range(500)]  
                for j in range(500)] 
    
    # Structure to represent a query range 
    class Query: 
        def __init__(self, l, r): 
            self.L = l 
            self.R = r 
    
    # Fills lookup array lookup[][] 
    # in bottom up manner. 
    def preprocess(arr: list, n: int): 
        global lookup 
    
        # Initialize M for the  
        # intervals with length 1 
        for i in range(n): 
            lookup[i][0] = i 
    
        # Compute values from  
        # smaller to bigger intervals 
        j = 1
        while (1 << j) <= n: 
    
            # Compute minimum value for 
            # all intervals with size 2^j 
            i = 0
            while i + (1 << j) - 1 < n: 
    
                # For arr[2][10], we compare  
                # arr[lookup[0][3]] and  
                # arr[lookup[3][3]] 
                if (arr[lookup[i][j - 1]] <  
                    arr[lookup[i + (1 << (j - 1))][j - 1]]): 
                    lookup[i][j] = lookup[i][j - 1] 
                else: 
                    lookup[i][j] = lookup[i + 
                                (1 << (j - 1))][j - 1] 
    
                i += 1
            j += 1
    
    # Returns minimum of arr[L..R] 
    def query(arr: list, L: int, R: int) -> int: 
        global lookup 
    
        # For [2,10], j = 3 
        j = int(log2(R - L + 1)) 
    
        # For [2,10], we compare  
        # arr[lookup[0][3]] and  
        # arr[lookup[3][3]], 
        if (arr[lookup[L][j]] <= 
            arr[lookup[R - (1 << j) + 1][j]]): 
            return arr[lookup[L][j]] 
        else: 
            return arr[lookup[R - (1 << j) + 1][j]] 
    
    # Prints minimum of given  
    # m query ranges in arr[0..n-1] 
    def RMQ(arr: list, n: int, q: list, m: int): 
    
        # Fills table lookup[n][Log n] 
        preprocess(arr, n) 
    
        # One by one compute sum of all queries 
        for i in range(m): 
    
            # Left and right boundaries 
            # of current range 
            L = q[i].L 
            R = q[i].R 
    
            # Print sum of current query range 
            print("Minimum of [%d, %d] is %d" % 
                    (L, R, query(arr, L, R))) 
    
    # Driver Code 
    if __name__ == "__main__": 
        a = [7, 2, 3, 0, 5, 10, 3, 12, 18] 
        n = len(a) 
        q = [Query(0, 4), Query(4, 7),  
                        Query(7, 8)] 
        m = len(q) 
    
        RMQ(a, n, q, m) 
    
    So sparse table method supports query operation in O(1) time with 
    O(n Log n) preprocessing time and O(n Log n) space.




-42.5) Sqrt Decomposition Python Example:

    # Python 3 program to demonstrate working of Square Root 
    # Decomposition. 
    from math import sqrt 
    
    MAXN = 10000
    SQRSIZE = 100
    
    arr = [0]*(MAXN)         # original array 
    block = [0]*(SQRSIZE)     # decomposed array 
    blk_sz = 0                 # block size 
    
    # Time Complexity : O(1) 
    def update(idx, val): 
        blockNumber = idx // blk_sz 
        block[blockNumber] += val - arr[idx] 
        arr[idx] = val 
    
    # Time Complexity : O(sqrt(n)) 
    def query(l, r): 
        sum = 0
        while (l < r and l % blk_sz != 0 and l != 0): 
            # traversing first block in range 
            sum += arr[l] 
            l += 1
        
        while (l + blk_sz <= r): 
            # traversing completely overlapped blocks in range 
            sum += block[l//blk_sz] 
            l += blk_sz 
        
        while (l <= r): 
            # traversing last block in range 
            sum += arr[l] 
            l += 1
        
        return sum
        
    # Fills values in input[] 
    def preprocess(input, n): 
        
        # initiating block pointer 
        blk_idx = -1
    
        # calculating size of block 
        global blk_sz 
        blk_sz = int(sqrt(n)) 
    
        # building the decomposed array 
        for i in range(n): 
            arr[i] = input[i]; 
            if (i % blk_sz == 0): 
                
                # entering next block 
                # incementing block pointer 
                blk_idx += 1; 
            
            block[blk_idx] += arr[i] 
    
    # Driver code 
    
    # We have used separate array for input because 
    # the purpose of this code is to explain SQRT 
    # decomposition in competitive programming where 
    # we have multiple inputs. 
    input= [1, 5, 2, 4, 6, 1, 3, 5, 7, 10] 
    n = len(input) 
    
    preprocess(input, n) 
    
    print("query(3,8) : ",query(3, 8)) 
    print("query(1,6) : ",query(1, 6)) 
    update(8, 0) 
    print("query(8,8) : ",query(8, 8)) 




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
    There are lots of possible approaches and data 
    structures that you can use to solve the RMQ task.

    The ones that are explained on this site are listed below.

    First the approaches that allow modifications to the array between answering queries.

    Sqrt-decomposition - answers each query in O(sqrt(N)), preprocessing done in O(N). 
    Pros: a very simple data structure. Cons: worse complexity.
    
    Segment tree - answers each query in O(logN), preprocessing done in O(N). 
                 Pros: good time complexity. Cons: larger amount of code compared to the other data structures.

    Fenwick tree - answers each query in O(logN), preprocessing done in O(NlogN). 
                Pros: the shortest code, good time complexity. 
                Cons: Fenwick tree can only be used for queries with L=1, 
                      so it is not applicable to many problems.

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





49.7 SQRT DECOMPOSITION  (https://cp-algorithms.com/data_structures/sqrt_decomposition.html)

    Sqrt Decomposition is a method (or a data structure) that allows 
    you to perform some common operations (finding sum of the elements 
    of the sub-array, finding the minimal/maximal element, etc.) in O(√n) 
    operations, which is much faster than O(n) for the trivial algorithm.

    The basic idea of sqrt decomposition is preprocessing. 
    We'll divide the array a into blocks of length approximately √n, and 
    for each block i we'll precalculate the sum of elements in it b[i].

    Let's start with the simplest implementation:

    // input data
    int n;
    vector<int> a (n);

    // preprocessing
    int len = (int) sqrt (n + .0) + 1; // size of the block and the number of blocks
    vector<int> b (len);
    for (int i=0; i<n; ++i)
        b[i / len] += a[i];

    // answering the queries
    for (;;) {
        int l, r;
    // read input data for the next query
        int sum = 0;
        for (int i=l; i<=r; )
            if (i % len == 0 && i + len - 1 <= r) {
                // if the whole block starting at i belongs to [l, r]
                sum += b[i / len];
                i += len;
            }
            else {
                sum += a[i];
                ++i;
            }
    }

    This implementation has unreasonably many division operations (which are 
    much slower than other arithmetical operations). Instead, we can calculate 
    the indices of the blocks cl and cr which contain indices l and r, and loop 
    through blocks cl+1…cr−1 with separate processing of the "tails" in blocks cl 
    and cr. This approach corresponds to the last formula in the description, 
    and makes the case cl=cr a special case.


    int sum = 0;
    int c_l = l / len,   c_r = r / len;
    if (c_l == c_r)
        for (int i=l; i<=r; ++i)
            sum += a[i];
    else {
        for (int i=l, end=(c_l+1)*len-1; i<=end; ++i)
            sum += a[i];
        for (int i=c_l+1; i<=c_r-1; ++i)
            sum += b[i];
        for (int i=c_r*len; i<=r; ++i)
            sum += a[i];
    }

    So far we were discussing the problem of finding the sum of elements of a 
    continuous subarray. This problem can be extended to allow to update individual 
    array elements. If an element a[i] changes, it's sufficient to update the value of 
    b[k] for the block to which this element belongs (k=i/s) in one operation:

    b[k]+=anew[i]−aold[i]


    On the other hand, the task of finding the sum of elements can be replaced with 
    the task of finding minimal/maximal element of a subarray. If this problem has to 
    address individual elements' updates as well, updating the value of b[k] is also 
    possible, but it will require iterating through all values of block k in O(s)=O(√n) operations.

    Sqrt decomposition can be applied in a similar way to a whole class of other problems: 
    finding the number of zero elements, finding the first non-zero element, counting 
    elements which satisfy a certain property etc.

    Another class of problems appears when we need to update array elements on intervals: 
    increment existing elements or replace them with a given value.

    For example, let's say we can do two types of operations on an array: add a given 
    value δ to all array elements on interval [l,r] or query the value of element a[i]. 
    Let's store the value which has to be added to all elements of block k in b[k] 
    (initially all b[k]=0). During each "add" operation we need to add δ to b[k] for 
    all blocks which belong to interval [l,r] and to add δ to a[i] for all elements 
    which belong to the "tails" of the interval. The answer a query i is simply a[i]+b[i/s]. 
    This way "add" operation has O(n√n) complexity, and answering a query has O(1) complexity.

    Finally, those two classes of problems can be combined if the task requires 
    doing both element updates on an interval and queries on an interval. Both 
    operations can be done with O(√n) complexity. This will require two block 
    arrays b and c: one to keep track of element updates and another to keep track of answers to the query.

    There exist other problems which can be solved using sqrt decomposition, for 
    example, a problem about maintaining a set of numbers which would allow 
    adding/deleting numbers, checking whether a number belongs to the set and 
    finding k-th largest number. To solve it one has to store numbers in 
    increasing order, split into several blocks with √n numbers in each. 
    Every time a number is added/deleted, the blocks have to be rebalanced 
    by moving numbers between beginnings and ends of adjacent blocks.


49.7 SQRT DECOMPOSITION  - MO'S Algorithm (https://cp-algorithms.com/data_structures/sqrt_decomposition.html)

    A similar idea, based on sqrt decomposition, can be used to answer range 
    queries (Q) offline in O((N+Q)N−−√). This might sound like a lot worse than 
    the methods in the previous section, since this is a slightly worse complexity 
    than we had earlier and cannot update values between two queries. But in a lot of 
    situations this method has advantages. During a normal sqrt decomposition, we have 
    to precompute the answers for each block, and merge them during answering queries. 
    In some problems this merging step can be quite problematic. E.g. when each queries 
    asks to find the mode of its range (the number that appears the most often). For this 
    each block would have to store the count of each number in it in some sort of data 
    structure, and we cannot longer perform the merge step fast enough any more. Mo's algorithm 
    uses a completely different approach, that can answer these kind of queries fast, because 
    it only keeps track of one data structure, and the only operations with it are easy and fast.

    The idea is to answer the queries in a special order based on the indices. We will 
    first answer all queries which have the left index in block 0, then answer all queries 
    which have left index in block 1 and so on. And also we will have to answer the 
    queries of a block is a special order, namely sorted by the right index of the queries.

    As already said we will use a single data structure. This data structure will 
    store information about the range. At the beginning this range will be empty. 
    When we want to answer the next query (in the special order), we simply extend 
    or reduce the range, by adding/removing elements on both sides of the current range, 
    until we transformed it into the query range. This way, we only need to add or 
    remove a single element once at a time, which should be pretty easy operations in our data structure.

    Since we change the order of answering the queries, this is only possible 
    when we are allowed to answer the queries in offline mode.

    Implementation
    In Mo's algorithm we use two functions for adding an index and for 
    removing an index from the range which we are currently maintaining.

    void remove(idx);  // TODO: remove value at idx from data structure
    void add(idx);     // TODO: add value at idx from data structure
    int get_answer();  // TODO: extract the current answer of the data structure

    int block_size;

    struct Query {
        int l, r, idx;
        bool operator<(Query other) const
        {
            return make_pair(l / block_size, r) <
                make_pair(other.l / block_size, other.r);
        }
    };

    vector<int> mo_s_algorithm(vector<Query> queries) {
        vector<int> answers(queries.size());
        sort(queries.begin(), queries.end());

        // TODO: initialize data structure

        int cur_l = 0;
        int cur_r = -1;
        // invariant: data structure will always reflect the range [cur_l, cur_r]
        for (Query q : queries) {
            while (cur_l > q.l) {
                cur_l--;
                add(cur_l);
            }
            while (cur_r < q.r) {
                cur_r++;
                add(cur_r);
            }
            while (cur_l < q.l) {
                remove(cur_l);
                cur_l++;
            }
            while (cur_r > q.r) {
                remove(cur_r);
                cur_r--;
            }
            answers[q.idx] = get_answer();
        }
        return answers;
    }

    Based on the problem we can use a different data structure and modify the 
    add/remove/get_answer functions accordingly. For example if we are asked to 
    find range sum queries then we use a simple integer as data structure, which 
    is 0 at the beginning. The add function will simply add the value of the 
    position and subsequently update the answer variable. On the other hand remove 
    function will subtract the value at position and subsequently update the answer 
    variable. And get_answer just returns the integer.

    For answering mode-queries, we can use a binary search tree (e.g. map<int, int>) 
    for storing how often each number appears in the current range, and a second binary 
    search tree (e.g. set<pair<int, int>>) for keeping counts of the numbers (e.g. as count-number pairs) in order. 
    The add method removes the current number from the second BST, increases the count in the first one, 
    and inserts the number back into the second one. remove does the same thing, it only decreases the 
    count. And get_answer just looks at second tree and returns the best value in O(1).

    Complexity
    Sorting all queries will take O(QlogQ).

    How about the other operations? How many times will the add and remove be called?

    Let's say the block size is S.

    If we only look at all queries having the left index in the same block, 
    the queries are sorted by the right index. Therefore we will call add(cur_r) 
    and remove(cur_r) only O(N) times for all these queries combined. 
    This gives O((N/S)N) calls for all blocks.

    The value of cur_l can change by at most O(S) during between two queries. 
    Therefore we have an additional O(SQ) calls of add(cur_l) and remove(cur_l).

    For S≈N√N this gives O((N+Q)√N) operations in total. Thus the 
    complexity is O((N+Q)F√N) where O(F) is the complexity of add and remove function.

    Tips for improving runtime
    Block size of precisely √N doesn't always offer the best runtime. 
    For example, if √N=750 then it may happen that block size of 700 or 800 may run better. 
    More importantly, don't compute the block size at runtime - make it const. 
    Division by constants is well optimized by compilers.

    In odd blocks sort the right index in ascending order and in even blocks 
    sort it in descending order. This will minimize the movement of right pointer, 
    as the normal sorting will move the right pointer from the end back to 
    the beginning at the start of every block. With the 
    improved version this resetting is no more necessary.

    bool cmp(pair<int, int> p, pair<int, int> q) {
        if (p.first / BLOCK_SIZE != q.first / BLOCK_SIZE)
            return p < q;
        return (p.first / BLOCK_SIZE & 1) ? (p.second < q.second) : (p.second > q.second);
    }

    You can read about faster sorting here: https://codeforces.com/blog/entry/61203




1)  Fenwick Tree Structure:
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


2)  Lowest Common Ancestor:
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
