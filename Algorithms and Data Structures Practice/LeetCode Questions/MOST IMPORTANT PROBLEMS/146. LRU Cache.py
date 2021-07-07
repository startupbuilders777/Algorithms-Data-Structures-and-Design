'''

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
'''

#DONEEEE

'''
For a fixed-size queue, where the oldest entry is to be automatically discarded,
 a better data structure 
would be a circular-buffer, which in Python is provided by the 
collections.deque class. You should make a deque(maxlen=capacity) 
instead of your LinkedList.

If this is not an exercise to create an LRU Cache from scratch, an
OrderedDict could be an even better data structure. It is a dictionary
and ordered list in one. It has built in methods .move_to_end(key) which
would mark the entry most recently used, and .popitem() which 
will remove the oldest entry. – AJNeufeld Aug 23 '18 at 15:41
'''

class Node(object):
    def __init__(self, data):
        self.prev = None
        self.data = data
        self.next = None
        
class LRUCache(object):
        
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.m = {}

        self.key_to_node = {}
        
        self.capacity = capacity
        self.curr_cap = 0
        self.front = None # Pointer to end of LRU cache. 
                         # When get is called, grab that node, 
                         # remove O(1), then put in front O(1)
        self.back = None
        
    def hit(self, key):
        hit_node = self.key_to_node[key]
        
        if(hit_node != self.front):
            if(hit_node.prev):
                hit_node.prev.next = hit_node.next
            if(hit_node.next):
                hit_node.next.prev = hit_node.prev
            if(hit_node == self.back): # NEED TO UPDATE BACK !
                self.back = hit_node.next 
            curr_front_node = self.front
            curr_front_node.next = hit_node
            hit_node.prev = curr_front_node
            hit_node.next = None
            self.front = hit_node
            
        
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        item = self.m.get(key)
        if(item is None):
            return -1
        self.hit(key)
        return item 

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """            
        if(self.m.get(key) is not None):
            self.m[key] = value
            self.hit(key)
            return 
            
        if(self.curr_cap < self.capacity):
            self.m[key] = value
            self.curr_cap += 1
            
            node = Node(key)
            
            if(self.front is None): #Either way, update both!
                self.front = node
                self.key_to_node[key] = node
                self.back  = node
                
            else:
                self.key_to_node[key] = node
                
                currFront = self.front
                # Update front node
                currFront.next = node
                node.prev = currFront                    
                self.front = node
                
                
        else:
            # evict item in cache
            # delete prev node. 
            if(self.back):
                
                back_key = self.back.data
                print("EVICTION OF key (backey, newkeytoadd)", (back_key, key) )
                
                newPrevNode = self.back.next
                self.key_to_node.pop(back_key)
                self.m.pop(back_key)
                self.back = newPrevNode
                self.curr_cap -= 1
                
            return self.put(key, value) # TRY AGAIN AFTER EVICT!
            
        
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


#################### Someone elses solution!!!!

# THEY MADE CODE MUCH EASIER BY USIGN HEAD TAIL NODES. LEARN HOW TO DO THIS!!!!
# ALSO CIRCULAR DOUBLY LINKED LIST TO MAKE CODE EASIER!!!
# ALSO MEMORY EFFICIENT BECAUSE STORE VALUE IN NODE, 
# AND MAP GOES KEY TO NODE. INSTEAD OF 2 MAPS!!!

class Node:
    def __init__(self, k, v):
    self.key = k
    self.val = v
    self.prev = None
    self.next = None

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


# BETTER ONE WITH ORDERED DICT

from collections import OrderedDict

class LRUCache(object):
    def __init__(self, capacity):
        self.dic = collections.OrderedDict()
        self.remain = capacity

    def get(self, key):
        if key not in self.dic:
            return -1
        v = self.dic.pop(key) 
        self.dic[key] = v   # set key as the newest one
        return v

    def set(self, key, value):
        if key in self.dic:    
            self.dic.pop(key)
        else:
            if self.remain > 0:
                self.remain -= 1  
            else:  # self.dic is full
                self.dic.popitem(last=False) 
        self.dic[key] = value



        

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)