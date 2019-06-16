# DONE


'''
138. Copy List with Random Pointer
Medium

1574

425

Favorite

Share
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

 

Example 1:



Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.


'''


"""
# Definition for a Node.
class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        m = {}
        t = {}
        
        if(head is None):
            return None        
        new_head = Node(head.val, None, None)
        nn = new_head
        n = head.next
        i = 1
        
        old_addr_to_position = {}
        position_to_new_node = {}
        
        # take old address, return position 
        old_addr_to_position[head] = 0  
        
        # take counter, return new_node at that position
        position_to_new_node[0] = new_head
        
        while n is not None:
                        
            new_node = Node(n.val, None, None)
            
            nn.next = new_node 
            new_node.val = n.val
            
            position_to_new_node[i] = new_node
            old_addr_to_position[n] = i 
            
            nn = nn.next
            n = n.next
            i += 1
        
        n = head
        nn = new_head
        
        while n is not None:
            pos = old_addr_to_position.get(n.random)
            new_random = None
            if(pos is not None):
                new_random = position_to_new_node[pos]
                
            nn.random = new_random
                                              
            nn = nn.next
            n = n.next
        
        
        return new_head

