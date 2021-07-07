'''
1367. Linked List in Binary Tree
Medium

314

12

Add to List

Share
Given a binary tree root and a linked list with head as the first node. 

Return True if all the elements in the linked list starting from the head correspond to some downward path connected in the binary tree otherwise return False.

In this context downward path means a path that starts at some node and goes downwards.

 

Example 1:



Input: head = [4,2,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: true
Explanation: Nodes in blue form a subpath in the binary Tree.  
Example 2:



Input: head = [1,4,2,6], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: true
Example 3:

Input: head = [1,4,2,6,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
Output: false
Explanation: There is no path in the binary tree that contains all the elements of the linked list from head.
 

Constraints:

1 <= node.val <= 100 for each node in the linked list and binary tree.
The given linked list will contain between 1 and 100 nodes.
The given binary tree will contain between 1 and 2500 nodes.

'''

# MY SOLUTION IS PRETTY FAST

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        
        '''
        its similar to boyer moore,
        when you mess up, you need to identify
        which string you should start searching with again. 
        dont just restart the search where u messed up the search,
        you have to go back up the tree and find longest matched sequence of characters,
        and see if they match a shifted version of the string. 
        
        you should save the tree nodes that match the head value of linked list, 
        store in map, and restart all searches from the map if there is a failure. 
        
        But yeah there is a string algorithm for bad partial matches, and skipping checks!
        optimization
        '''
        
        partial = []
        
        # Memoize failure!
        m = {}
        def find(tn, ln):
            nonlocal partial
            nonlocal m 
            
            if(m.get((tn, ln)) != None):
                return m[(tn, ln)]
            
            
            if ln == None:
                # found
                return True
            
            if(tn == None):
                return False
            # print("head val, ln val", head.val, ln.val)
            
            
            if ln != head and tn.val == head.val:
                # we found another search candidate. 
                partial.append(tn)
            
            
            if tn.val == ln.val:
                # recurse on both left and right side. 
                m[(tn, ln)]  = find(tn.right, ln.next) or find(tn.left, ln.next)
                return m[(tn, ln)] 
            
            else: 
                # nodes further up will find it first?
                # should start with those first!!
                partialMatch = partial
                partial = [] 
                # its important to empty it, otherwise, your recursive calls will use this
                
                
                # CAN WE memoize below, because it depends on how many partial matches, you find
                # i guess, youll find the same number of partial matches each time you revisit a node.
                # no you wont, because it depends on where you started. 
                # you dont have to memoize true. only false. 
                
                while partialMatch:
                    partialNodeMatch = partialMatch.pop()
                    
                    if(find(partialNodeMatch.left, head.next) or find(partialNodeMatch.right, head.next)):
                        return True
                    
                m[(tn, ln)] = find(tn.right, head) or find(tn.left, head)
                return m[(tn, ln)]
        
        return find(root, head)


# ELEGANT:

def isSubPath(self, head, root, is_consecutive = False):
    if not head: return True
    if not root: return False
    if is_consecutive:
        if head.val != root.val: return False
        return self.isSubPath(head.next, root.left, True) or self.isSubPath(head.next, root.right, True)
    return self.isSubPath(head, root, True) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)