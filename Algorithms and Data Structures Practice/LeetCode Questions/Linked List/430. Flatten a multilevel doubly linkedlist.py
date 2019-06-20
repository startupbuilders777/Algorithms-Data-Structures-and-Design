'''
430. Flatten a Multilevel Doubly Linked List
Medium

356

58

Favorite

Share
You are given a doubly linked list which in addition to the next and previous pointers, it could have a child pointer, 
which may or may not point to a separate doubly linked list. These child lists may have one or more children of 
their own, and so on, to produce a multilevel data structure, as shown in the example below.

Flatten the list so that all the nodes appear in a single-level, doubly linked list. 
You are given the head of the first level of the list.

Example:

Input:
 1---2---3---4---5---6--NULL
         |
         7---8---9---10--NULL
             |
             11--12--NULL

Output:
1-2-3-7-8-11-12-9-10-4-5-6-NULL

'''

"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution:
    
    def print_list(self, node, result=[]):
        if(node is None):
            print("nodes are: ", result)
        else:
            result.append(node.val)
            self.print_list(node.next, result)

    def flatten(self, head: 'Node') -> 'Node':
        
        
        
        def flat(node, rest):
            if(node is None and rest is not None):
                # print("REST WAS NOT USED, AND NODE BECAME NONE!")
                return None
            
            if(node is None):
                return None
            
            if(node.next is None and rest):
                flat_rest = flat(rest, None)
                
                node.next = flat_rest
                flat_rest.prev = node
                return node
                    

            if(node.child):
                # print("NODE v HAS CHILD, children are: ", node.val)
                # self.print_list(node.child)
                flat_child = flat(node.child, node.next)
                node.child = None
                node.next = flat_child 
                flat_child.prev = node
                return flat(node, rest)
            
            node.next =  flat(node.next, rest)
            # node.next = flatten_next
            # flat.prev = node
            
            return node
        
        return flat(head, None)

# USE EXPLICIT STACK FOR THIS SOLUTION:

class Solution(object):
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


