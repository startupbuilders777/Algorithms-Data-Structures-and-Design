# DONE


'''
138. Copy List with Random Pointer
Medium

1574

425

Favorite

Share
A linked list is given such that each node contains an 
additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

Example 1:



Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.


'''

'''
CONSTANT SPACE SOLUTION:
An intuitive solution is to keep a hash table for each node in the list, 
via which we just need to iterate the list in 2 rounds respectively 
to create nodes and assign the values for their random pointers. 
As a result, the space complexity of this solution is O(N), 
although with a linear time complexity.

Note: if we do not consider the space reversed for the output, 
then we could say that the algorithm does not consume any 
additional memory during the processing, i.e. O(1) space complexity

As an optimised solution, we could reduce the space complexity 
into constant. The idea is to associate the original node with 
its copy node in a single linked list. In this way, we don't
need extra space to keep track of the new nodes.

The algorithm is composed of the follow three steps which are also 3 iteration rounds.

Iterate the original list and duplicate each node. The duplicate
of each node follows its original immediately.
Iterate the new list and assign the random pointer for each
duplicated node.
Restore the original list and extract the duplicated nodes.
The algorithm is implemented as follows:


'''


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

# SIMPLER WAY:
'''
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

# Alternatively, we can modify 'random' field instead of the 'next' field in the copied list:

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # create new nodes
        node = head
        while node:
            node.random, node = Node(node.val, node.random, None), node.next
        # populate random field of the new node
        node = head
        while node:
            node.random.random, node = node.random.next.random if node.random.next else None, node.next 
        # restore original list and build new list 
        head_copy, node = head.random if head else None, head
        while node:
            node.random.next, node.random, node = node.next.random if node.next else None, node.random.next, node.next
        return head_copy






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

