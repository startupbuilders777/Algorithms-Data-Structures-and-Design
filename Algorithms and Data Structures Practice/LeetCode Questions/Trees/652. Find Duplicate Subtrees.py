'''
652. Find Duplicate Subtrees
Medium

1261

198

Add to List

Share
Given a binary tree, return all duplicate subtrees. 
For each kind of duplicate subtrees, you only need to 
return the root node of any one of them.

Two trees are duplicate if they have the same structure with same node values.

Example 1:

        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
      
The following are two duplicate subtrees:

      2
     /
    4
and

    4
Therefore, you need to return above trees' root in the form of a list.
'''


'''
O(N) SOLUTION BETTER THAN STRING SERIALZIATION SOLUTIONS BELOW WHICH ARE O(N^2)
'''

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

'''


@StefanPochmann, I'm not sure about the time complexity of string or your tuple hash. 
For each time we wanna get the hash value of a string, should we regard the 
action as O(1) or O(length of string)? If it is the latter, then the total 
time complexity would be O(n ^ 2), given tree is a linked list, which 
conflicts with the time of actual execution of those solutions; 
otherwise, in what way that Python makes the operation of 
string with different length has the same time complexity? Thanks!


STEFAN: To answer your questions: Python strings cache their hashes, 
Python tuples do not. See this for reasons. My post (really a 
full-blown article by now :-) goes into detail about tuples. 
For strings, well, the one time they do calculate their hash, 
that takes O(length) time. Just like their creation takes anyway. 
So even if string hashes magically always were O(1), then the 
string serialization solutions would still only be O(n2) 
time already just for creating the strings. 




How can type_id represent a tree signature without node.val involved? thanks.


Sorry for replying late, here is a recursive explanation:
Base case:
For a nil node, the type_id would be always -1;

Recursive step:
For a non-nil node, suppose we get the left and right subtree signature 
as type_id_left, type_id_right. Then the signature of the tree rooted 
in the node should be 

(value of the node as the root, type_id_left, type_id_right). If it shows 
before in hashmap, we find the type_id of this tree, otherwise we give 
it a new id and record it.

As you can see, a structure of a tree would be decided by three parts: 
root, left sub-tree and right sub-tree. we use type_id to represent 
the structure of a tree, so that the node.val has been included 
in the type_id already, as a type_id corresponds 

(node.val, left type_id, right type_id), which means the type_id 
includes all values of the nodes in the tree, as you can expand 
the type_id and all type_ids shows up. For each expand step, 
you will get one node value, and finally you will get a whole tree without type_id.

'''






# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        
        # Well when you process each node, you build a merkel hash for it. 
        # then you see if the hashes match, and return the roots!
        # Problem: hash collisions.
        
        tree_m = {}
        result = set()
        
        def dfs(node):
            
            if node is None:
                return "#"
            
            
            l = dfs(node.left)
            r = dfs(node.right)
            
            h = str(node.val) + "," + l + "," + r 
            
            if tree_m.get(h):
                result.add(tree_m.get(h))
            else:
                tree_m[h] = node
            
            return h
        
        dfs(root)
        return result
    
# FASTEST SOLUTION AVOID USING SET PLEASE!
# Well it uses counter as a set! 
# by only tracking the second time we see something. 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        counter = collections.Counter()
        res = [] 
        def dfs(node):
            if not node:
                return ' '
            serialize = str(node.val) + ',' + dfs(node.left) + ',' + dfs(node.right)
            if counter[serialize] == 1:
                res.append(node)
            counter[serialize] += 1    
            return serialize
        dfs(root)
        return res


def findDuplicateSubtrees(self, root):
        def trv(root):
            if not root: return "null"
            struct = "%s,%s,%s" % (str(root.val), trv(root.left), trv(root.right))
            nodes[struct].append(root)
            return struct
        
        nodes = collections.defaultdict(list)
        trv(root)
        return [nodes[struct][0] for struct in nodes if len(nodes[struct]) > 1]


'''
To create hash values we can do pre and post but not inorder:

For those who're confused why pre and post order works but not in order. I think this clears my confusion: https://leetcode.com/problems/find-duplicate-subtrees/discuss/106011/Java-Concise-Postorder-Traversal-Solution/108467. To be brief, 
it's because inorder can create same serialization for symmetric trees.
'''

